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
import json
import mrcfile 
import random


## Use 1 GPU card
a = os.getenv("CUDA_VISIBLE_DEVICES")
print(f"devices = {a}")
pre_trained = sys.argv[1]
print(pre_trained)
if pre_trained == 'random':
    pass
elif pre_trained == 'video':
    ckpt = 'google/vivit-b-16x2-kinetics400'
elif pre_trained == '10-class-SNR005':
    ckpt = 'ckpt-10-class-SNR005'
else:
    raise 'pre-trained parameter not supported'

num_gpu = torch.cuda.device_count()
print('use %d gpu cards' % num_gpu)

# Use Noble dataset
num_classes = 7
data_path = "/shared/home/v_nishanth_artham/local_scratch/CryoET/2d_to_3d_transformer_subtomogram/real_data/Noble_7_class.json"
# val_path = "./data_split/val_real_split.csv"
# test_path = "./data_split/test_real_split.csv"

video_init = True  

size = (224, 224)
transforms_list = [
    transforms.RandomResizedCrop(size=size, scale=(0.5, 1)), 
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.9, 1.1))
]
Noble_7_class = ["31","72","43","35","33","73","69"]

class CustomDataset(Dataset):
    def __init__(self, train_path, split="all"):
        # self.size = size
        # df = pd.read_csv(csv_path)
        # self.image_paths = list(df["path"])
        with open(train_path,'r') as file:
            # print(f"image paths = {json.load(file)}")
            self.data_dict =  json.load(file)
            print(f"total dataset size: {len(self.data_dict)}")
            self.image_paths = list(self.data_dict.keys())#not using this
        # self.labels = list(df["label"])
            self.labels = list(x[0] for x in self.data_dict.values())
        # self.transforms = transforms_list
        np.random.seed(54321)
        random.seed(54321)
        # print(f"image paths = {self.image_paths}")
        # print(f"lables  = {self.labels}")
        self.mrc_data = []
        for label in Noble_7_class:
            paths = [p for p, l in self.data_dict.items() if l[0] == label]
            random.shuffle(paths)
            if (split == "train"):
                self.mrc_data += paths[:int(0.60*len(paths))]
            elif (split == "eval"):
                self.mrc_data += paths[int(0.60*len(paths)):int(0.80*len(paths))]
            elif (split == "test"):
                self.mrc_data += paths[int(0.80*len(paths)):]
            else:
                raise Exception("Invalid data split")
        print(f"{split} dataset size: {len(self.mrc_data)}")

    def __len__(self):
        return len(self.mrc_data)

    def __getitem__(self, index):
        print(index)
        path = self.mrc_data[index]
        # label = torch.from_numpy((np.array(self.labels[index])))
        print(f"image path = {path}")

        # img = Image.open(path, 'r')
        # video = []
        # count = 0
        # for i in range(5):
        #     for j in range(6):
        #         left = 29*j
        #         top = 29*i
        #         right = 29*j+28
        #         bottom = 29*i+28
        #         video.append(np.array(img.crop((left, top, right, bottom))))
        #         count += 1
        #         if count == 28:
        #             break
        #     if count == 28:
        #         for j in range(4):
        #             video.append(np.array(img.crop((left, top, right, bottom))))
        #         break
        # video = np.array(video)
        # video = (video - video.min())/(video.max() - video.min())
        # video = np.stack((video,)*3, axis=1)
        # video = torch.from_numpy(video)
        # print(f"size = {video.shape}")
        # for t in self.transforms:
        #     s = np.random.uniform()
        #     if s > 0.5:
        #         video = t(video)
        video = mrcfile.open(path).data
        label = self.data_dict[path][0]
        print(f"mrc dim = {video.shape} and label = {label}")
        
        return torch.tensor(np.expand_dims(video, axis=0), dtype=torch.float32),torch.tensor(int(label))

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(data_path,"train")
val_dataset = CustomDataset(data_path, "eval")
test_dataset = CustomDataset(data_path, "test")

def collate_fn(examples):
    pixel_values = []
    labels = []
    for data, l in examples:
        pixel_values.append(data)
        labels.append(l)
        print(f"pixel_values = {pixel_values} and labels={labels}")
    print(f"pixel_values: {torch.stack(pixel_values).shape}, labels: {torch.stack(labels).shape}")
    return {"pixel_values": torch.stack(pixel_values), "labels": torch.stack(labels)}

model = VivitForVideoClassification.from_pretrained(ckpt, num_labels = num_classes, ignore_mismatched_sizes=True)
model.config.image_size = (28, 28, 28)
if pre_trained == 'random':
    model = VivitForVideoClassification(model.config)

from transformers import TrainingArguments, Trainer

metric_name = "accuracy"

args = TrainingArguments(
    f"vivit_rgb",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5 if video_init else 2e-4,
    num_train_epochs=20 if video_init else 100,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    logging_steps=10,
    remove_unused_columns=False,
    fp16=True,
    bf16=False,
    gradient_accumulation_steps=8/num_gpu,
)

from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

# opt = Adam(model.parameters(),
#                      lr=1e-4)
# schd = lr_scheduler.MultiStepLR(opt, milestones=[9375, 18750, 28125], gamma=0.1)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=None,
    # optimizers=(opt, schd)
)

trainer.train()
print("Training done")
outputs = trainer.predict(test_dataset)
print(outputs.metrics)