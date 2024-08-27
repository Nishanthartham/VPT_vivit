from transformers import VivitImageProcessor, VivitForVideoClassification
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, lr_scheduler, Adam, AdamW
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
import torch.nn as nn
import numpy as np
import os
import sys
import random
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from PIL import Image
from torch.nn import Parameter
import matplotlib.pyplot as plt

## Use 1 GPU card

plot_name = sys.argv[1]
# pre_trained = "video"
pre_trained = "10-class-SNR005"
print(f"plot_name = {plot_name} and pretrained = {pre_trained}")
if pre_trained == 'random':
    pass
elif pre_trained == 'video':
    ckpt = 'google/vivit-b-16x2-kinetics400'
    print(f"cktp ={ckpt}")
elif pre_trained == '10-class-SNR005':
    ckpt = 'ckpt-10-class-SNR005'
    print(f"cktp ={ckpt}")
else:
    raise 'pre-trained parameter not supported'

num_gpu = torch.cuda.device_count()
print('use %d gpu cards' % num_gpu)

# Use Noble dataset
num_classes = 7
train_path = "./data_split/train_real_split.csv"
# train_path = "./data_split/train_real_3_shot.csv"
# val_path = "./data_split/val_real_split.csv"
test_path = "./data_split/test_real_split.csv"
# test_path = "./data_split/test_real_3_shot.csv"

video_init = True  
# vpt = False
size = (224, 224)
transforms_list = [
    transforms.RandomResizedCrop(size=size, scale=(0.5, 1)), 
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.9, 1.1))
]

class CustomDataset(Dataset):
    def __init__(self, csv_path, size, transforms_list,num_prompts):
        self.size = size
        df = pd.read_csv(csv_path)
        self.image_paths = list(df["path"])
        self.labels = list(df["label"])
        self.transforms = transforms_list
        self.num_prompts = num_prompts
        self.prompts = Parameter(torch.randn(num_prompts, 3, *size))  # Learnable prompts of size (224, 224)

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
                # if not vpt:
                for j in range(4):
                    video.append(np.array(img.crop((left, top, right, bottom)).resize(self.size)))
                break
        video = np.array(video)
        # print(f" video shape = {video.shape}")#28,224,224
        video = (video - video.min())/(video.max() - video.min())
        video = np.stack((video,)*3, axis=1)#(32,3,224,224) -> (35,3,224,224) 
        # print(f"video.shape = {video.shape}")
        video = torch.from_numpy(video)

        # if vpt:
        #     # print("Applied VPT")
        #     prompts = self.prompts
        #     video = torch.cat((video,prompts),dim=0)#(32,3,224,224)
        # print(f" video shape after concat = {video.shape}")#28,224,224

        for t in self.transforms:
            s = np.random.uniform()
            if s > 0.5:
                video = t(video)
        # print(f"final video.shape = {video.shape}")
        
        return video.float(), label

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_path, size, transforms_list,100)
# val_dataset = CustomDataset(val_path, size, [],0)
test_dataset = CustomDataset(test_path, size, [],0)
def collate_fn(examples):
    pixel_values = []
    labels = []
    for data, l in examples:
        # print(f"collate size = {data.shape}")
        pixel_values.append(data)
        labels.append(l)
    # print(f"torch.stack(pixel_values) = {torch.stack(pixel_values).shape} and {torch.stack(labels)}")
    return {"pixel_values": torch.stack(pixel_values), "labels": torch.stack(labels)}

model = VivitForVideoClassification.from_pretrained(ckpt, num_labels = num_classes, ignore_mismatched_sizes=True)
# print(f"model config = {model.config}")
# print(f"type of model config = {type(model.config)}")
if pre_trained == 'random':
    model = VivitForVideoClassification(model.config)

from transformers import TrainingArguments, Trainer

metric_name = "accuracy"

args = TrainingArguments(
    f"vivit_rgb",
    save_strategy="no",#######The checkpoint save strategy to adopt during training
    evaluation_strategy="no",
    do_eval=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5 if video_init else 2e-4,
    num_train_epochs=20 if video_init else 100,
    load_best_model_at_end=False,########To check
    metric_for_best_model=metric_name,
    log_level='info',
    logging_dir='logs',
    logging_steps=10,
    logging_strategy="epoch",
    remove_unused_columns=False,
    fp16=True,
    bf16=False,
    gradient_accumulation_steps=8/num_gpu,
    max_grad_norm=1.0
)

from sklearn.metrics import accuracy_score
import numpy as np

eval_acc_list = []
def compute_metrics(eval_pred):
    global eval_acc_list
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    eval_acc = accuracy_score(predictions, labels)
    eval_acc_list.append(eval_acc)
    return dict(accuracy=eval_acc)

# opt = Adam(model.parameters(),
#                      lr=1e-4)
# schd = lr_scheduler.MultiStepLR(opt, milestones=[9375, 18750, 28125], gamma=0.1)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=None
    # optimizers=(opt, schd)
)

# train_outputs = trainer.train()
trainer.train()
# print(f"train outputs = {train_outputs.metrics}")
print(f"accuracy outputs = {eval_acc_list}")
# plt.plot(eval_acc_list)
# plt.savefig(f"/shared/home/v_nishanth_artham/local_scratch/CryoET/3d_resnet/plots/prompting_complete_training/{plot_name}.png")
# plt.close()
outputs = trainer.predict(test_dataset)
print(f"output = {outputs.metrics}")
