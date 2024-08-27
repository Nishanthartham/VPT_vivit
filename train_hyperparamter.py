from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, lr_scheduler, Adam
import resnet
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
import torch.nn as nn
import numpy as np
import os
import random
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from PIL import Image
from ray import tune
from tqdm import tqdm

FRAC_TRAIN = 1
WEIGHT_DECAY = 0
# STEP_SIZE = 40
SEED = 30
NUM_CLASSES = 7
SAMPLE_DURATION = 28
MODEL_PATH = "/local/scratch/v_sabhay_jain/finetuning_3d_resnet/resnet-34-kinetics-cpu.pth"
DATA_TYPE = "real_0.05"
SAVE_DIR = f"models/models_rgb_{DATA_TYPE}"
MODEL_NAME = f"resnet_{FRAC_TRAIN}.pth"
TRAIN_PATH = f"/local/scratch/v_sabhay_jain/training_3d_resnet/finetuning/global_data/train_real_split_2_0.05.csv"
VAL_PATH = f"/local/scratch/v_sabhay_jain/training_3d_resnet/finetuning/global_data/test_real_split_2.csv"
TEST_PATH = f"/local/scratch/v_sabhay_jain/training_3d_resnet/finetuning/global_data/test_real_split_2.csv"
DATASET = "experimental_data"
MEAN = 0
STD = 1

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

size = (128, 128)
transforms_list = [
    transforms.RandomResizedCrop(size=size, scale=(0.5, 1)), 
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.9, 1.1))
]

class CustomDataset(Dataset):
    def __init__(self, csv_path, size, transforms_list):
        self.size = size
        df = pd.read_csv(csv_path)
        self.image_paths = list(df["path"])
        self.labels = list(df["label"])
        self.transforms = transforms_list

    def __getitem__(self, index):
        path = self.image_paths[index]
        label = torch.from_numpy((np.array(self.labels[index])))
        img = Image.open(path, 'r')
        video = []
        count = 0
        for i in range(5):
            for j in range(6):
                left = 29*j
                top = 29*i
                right = 29*j+28
                bottom = 29*i+28
                video.append(np.array(img.crop((left, top, right, bottom)).resize(self.size)))
                count += 1
                if count == 28:
                    break
            if count == 28:
                break
        video = np.array(video)
        video = (video - video.min())/(video.max() - video.min())
        video = np.stack((video,)*3, axis=0)
        video = torch.from_numpy(video)
        for t in self.transforms:
            s = np.random.uniform()
            if s > 0.5:
                video = t(video)
        return video.float(), label

    def __len__(self):
        return len(self.labels)

class resnet_model(nn.Module):
    def __init__(self, model, hidden_size, num_classes):
        super(resnet_model, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.model(x)
        out = self.fc1(out)
        return out


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def load_resnet_video_pretrained(model_path):
    model = resnet.resnet34(num_classes=400, shortcut_type='A',
                            sample_size=128, sample_duration=SAMPLE_DURATION,
                            last_fc=False)

    if model_path != None and model_path != "":
        weights = torch.load(model_path, map_location='cpu')
        if "state_dict" in weights:
            model.load_state_dict(weights["state_dict"])
        else:
            model.load_state_dict(weights)
        print(f"loaded weight from {model_path}")
    return model

set_seed(SEED)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("Loading the data\n")
print(TRAIN_PATH)
print(SAVE_DIR)

train_dataset = CustomDataset(TRAIN_PATH, size, transforms_list)
val_dataset = CustomDataset(VAL_PATH, size, [])
test_dataset = CustomDataset(TEST_PATH, size, [])

def train(config):
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    gamma = config["gamma"]
    batch_size = config["batch_size"]
    print(f"epochs:{epochs}, learning_rate: {learning_rate}, gamma: {gamma}, batch_size: {batch_size}")
    epochs = int(epochs)
    batch_size = int(batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, pin_memory=True)

    val_loader = DataLoader(val_dataset, 32,
                            shuffle=False, pin_memory=True)
    video_pretrained_model = load_resnet_video_pretrained(MODEL_PATH)
    resnet_model_train = resnet_model(video_pretrained_model, 512, NUM_CLASSES)

    # test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    optimizer = Adam(resnet_model_train.parameters(),
                     lr=learning_rate,
                     weight_decay=0)
    milestones = [int(epochs*0.5), int(epochs*0.75)]
    criterion = CrossEntropyLoss()
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma)

    accelerator = Accelerator()
    device = accelerator.device

    train_loader, val_loader, resnet_model_train, optimizer, scheduler = accelerator.prepare(
        train_loader, val_loader, resnet_model_train, optimizer, scheduler
    )

    # df = pd.DataFrame(columns = ["train_loss", "val_loss", "val_auc", "val_acc"])

    print("training the model\n")
    prev_auc_score = 0
    for epoch in range(epochs):
        resnet_model_train.train()
        epoch_loss = 0
        pbar = tqdm(train_loader)
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = resnet_model_train(inputs.to(torch.float32))
            loss = criterion(outputs, targets)

            accelerator.backward(loss)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_description(f"steps: loss {loss_val}")
        resnet_model_train.eval()
        val_loss = 0
        val_accuracy = 0
        out = []
        out_pred = []
        for inputs, targets in tqdm(val_loader):
            with torch.no_grad():
                outputs = resnet_model_train(inputs.to(torch.float32))
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_accuracy += calculate_accuracy(outputs, targets)
                out.extend(targets.detach().cpu().tolist())
                out_pred.extend(torch.softmax(
                    outputs, dim=-1).detach().cpu().tolist())
        if NUM_CLASSES == 2:
            auc_score = roc_auc_score(np.array(out), np.array(out_pred)[:, 1])
        else:
            auc_score = roc_auc_score(
                np.array(out), np.array(out_pred), multi_class='ovr')

        if prev_auc_score <= auc_score:
            # torch.save(resnet_model_train.state_dict(),
            #            os.path.join(SAVE_DIR, MODEL_NAME))
            prev_auc_score = auc_score
        scheduler.step()
        # df.loc[len(df.index)] = [epoch_loss/len(train_loader), val_loss/len(val_loader), auc_score, val_accuracy/len(val_loader)]
        print(f"{epoch} epochs train_loss: {epoch_loss/len(train_loader)} val_loss: {val_loss/len(val_loader)} val_accuracy: {val_accuracy/len(val_loader)} val_auc_score: {auc_score}")
    tune.report(auc=prev_auc_score)


config ={
    'learning_rate':tune.loguniform(1e-5, 1e-3),
    'batch_size':tune.choice([4, 8, 16, 32]),
    'epochs':tune.choice([i*10 for i in range(2, 5)]),
    'gamma': tune.loguniform(5e-2, 5e-1)
}
analysis = tune.run(train, config=config)
print("Best config: ", analysis.get_best_config(metric="auc"), resources_per_trial={"cpu": 8, "gpu": 1})
df_analysis = analysis.dataframe()
df_analysis.to_csv("rgb_0.5_analysis.csv", index=False)
