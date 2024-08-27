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
import random
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from PIL import Image

## Use 2 GPU cards

train_path = "data_split/simulated_SNR005_train.csv"
val_path = "data_split/simulated_SNR005_val.csv"
test_path = "data_split/simulated_SNR005_test.csv"

video_init = True  

size = (224, 224)
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
        for i in range(6):
            for j in range(6):
                left = 33*j
                top = 33*i
                right = 33*j+32
                bottom = 33*i+32
                video.append(np.array(img.crop((left, top, right, bottom)).resize(self.size)))
                count += 1
                if count == 32:
                    break
            if count == 32:
                break
        video = np.array(video)
        video = (video - video.min())/(video.max() - video.min())
        video = np.stack((video,)*3, axis=1)
        video = torch.from_numpy(video)
        for t in self.transforms:
            s = np.random.uniform()
            if s > 0.5:
                video = t(video)
        return video.float(), label

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_path, size, transforms_list)
val_dataset = CustomDataset(val_path, size, [])
test_dataset = CustomDataset(test_path, size, [])

def collate_fn(examples):
    pixel_values = []
    labels = []
    for data, l in examples:
        pixel_values.append(data)
        labels.append(l)
    return {"pixel_values": torch.stack(pixel_values), "labels": torch.stack(labels)}

model = VivitForVideoClassification.from_pretrained("./ckpt-10-class-SNR005", num_labels = 10, ignore_mismatched_sizes=False)
#model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400", num_labels = 10, ignore_mismatched_sizes=True)
if not video_init:
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
    num_train_epochs=10 if video_init else 100,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    logging_steps=10,
    remove_unused_columns=False,
    fp16=False,
    bf16=True,
    gradient_accumulation_steps=4,
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

#trainer.train()

outputs = trainer.predict(test_dataset)
print(outputs.metrics)
