from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, lr_scheduler, Adam
import resnet
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
import torch.nn as nn
from config2 import *
import numpy as np
import os
import random
from sklearn.metrics import roc_auc_score, roc_curve
from collections import OrderedDict


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


class CustomDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __getitem__(self, index):
        data = np.load(os.path.join(self.path, f"{index}.npz"))
        img_3d, label = data["img_3d"], data["label"]
        img_3d_norm = (img_3d - MEAN)/STD
        return torch.from_numpy(img_3d_norm), torch.from_numpy(label)

    def __len__(self):
        return len(os.listdir(self.path))


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
    model = resnet.resnet50(num_classes=400, shortcut_type='B',
                            sample_size=128, sample_duration=SAMPLE_DURATION,
                            last_fc=False)

    if model_path != None and model_path != "":
        weights = torch.load(model_path, map_location='cpu')
        if "state_dict" in weights:
            new_state_dict = OrderedDict()
            for k, v in weights["state_dict"].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(weights)
        print(f"loaded weight from {model_path}")
    return model


def test(test_loader, video_pretrained_model):
    resnet_model_test = resnet_model(video_pretrained_model, 2048, NUM_CLASSES)
    weights = torch.load(os.path.join(SAVE_DIR, MODEL_NAME))
    resnet_model_test.load_state_dict(weights)

    test_loader, resnet_model_test = accelerator.prepare(
        test_loader, resnet_model_test
    )
    test_loss = 0
    test_accuracy = 0
    out = []
    out_pred = []
    resnet_model_test.eval()
    for i, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            outputs = resnet_model_test(inputs.to(torch.float32))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(outputs, targets)
            out.extend(targets.detach().cpu().tolist())
            out_pred.extend(torch.softmax(
                outputs, dim=-1).detach().cpu().tolist())
    test_accuracy = test_accuracy/len(test_loader)
    if NUM_CLASSES == 2:
        auc_score = roc_auc_score(np.array(out), np.array(out_pred)[:, 1])
        fpr, tpr, thresholds = roc_curve(
            np.array(out), np.array(out_pred)[:, 1])
        max_acc = 0
        for thresh in thresholds:
            acc = np.sum(np.array(out) == (
                np.array(out_pred)[:, 1] > thresh))/len(out)
            max_acc = max(acc, max_acc)
        test_accuracy = max_acc
    else:
        auc_score = roc_auc_score(
            np.array(out), np.array(out_pred), multi_class='ovr')
        test_accuracy = np.sum(np.array(out) == (
            np.argmax(out_pred, axis=-1)))/len(out)
    return test_accuracy, auc_score


if __name__ == "__main__":
    set_seed(SEED)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    print("Loading the data\n")
    print(TRAIN_PATH)

    train_dataset = CustomDataset(TRAIN_PATH)
    val_dataset = CustomDataset(VAL_PATH)
    test_dataset = CustomDataset(TEST_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    video_pretrained_model = load_resnet_video_pretrained(MODEL_PATH)
    resnet_model_train = resnet_model(video_pretrained_model, 2048, NUM_CLASSES)
    # for p in resnet_model_train.model.parameters():
    #     p.requires_grad = False
    
    # non_frozen_parameters = [p for p in resnet_model_train.parameters() if p.requires_grad]
    # print(len(non_frozen_parameters))

    # weights = torch.load("/local/scratch/v_sabhay_jain/finetuning_3d_resnet/models_rgb_2_exp/resnet_1.pth")
    # del weights["fc1.weight"]
    # del weights["fc1.bias"]
    # resnet_model_train.load_state_dict(weights, strict=False)

    epochs = EPOCHS
    optimizer = Adam(resnet_model_train.parameters(),
                     lr=LEARNING_RATE,
                     weight_decay=WEIGHT_DECAY)
    criterion = CrossEntropyLoss()
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=GAMMA)

    accelerator = Accelerator()
    device = accelerator.device

    train_loader, val_loader, resnet_model_train, optimizer, scheduler = accelerator.prepare(
        train_loader, val_loader, resnet_model_train, optimizer, scheduler
    )

    print("training the model\n")
    prev_auc_score = 0
    for epoch in range(epochs):
        resnet_model_train.train()
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = resnet_model_train(inputs.to(torch.float32))
            loss = criterion(outputs, targets)

            accelerator.backward(loss)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            print(f"{i+1} steps: loss {loss_val}")
        resnet_model_train.eval()
        val_loss = 0
        val_accuracy = 0
        out = []
        out_pred = []
        for i, (inputs, targets) in enumerate(val_loader):
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
            torch.save(resnet_model_train.state_dict(),
                       os.path.join(SAVE_DIR, MODEL_NAME))
            prev_auc_score = auc_score
        scheduler.step()

        print(f"{epoch} epochs train_loss: {epoch_loss/len(train_loader)} val_loss: {val_loss/len(val_loader)} val_accuracy: {val_accuracy/len(val_loader)} val_auc_score: {auc_score}")
    print("Testing the model\n")
    test_accuracy, auc_score = test(test_loader, video_pretrained_model)
    print(f"Test ACC: {test_accuracy}")
    print(f"Test AUC: {auc_score}")
