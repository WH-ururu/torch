import sys
import string
from ast import literal_eval
import pandas as pd
from imutils import paths
import math
from pandas.core.frame import DataFrame
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import DataLoader
import torchvision
import copy
import json
import repackage

repackage.up(1)
from modules.neuralnet import train_model, visualize_model


# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("../configuation/config.json") as f:
    CONFIG = json.load(f)

epochs = int(input("Enter the epochs(default: 25) : ") or "25")
lr = float(input("Optimizer learning rate(default: 0.001): ") or "0.001")
momentum = float(input("Optimizer momentum(default: 0.09): ") or "0.9")
step_size = int(input("scheduler Stepsize(default: 7): ") or "7")
gamma = float(input("scheduler gamma(default: 0.1): ") or "0.1")


args = {
    "epochs": epochs,
    "learning_rate": lr,
    "momentum": momentum,
    "step_size": step_size,
    "gamma": gamma,
}

print(args)

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "validation": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

data_dir = CONFIG["imageDir"]
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "validation"]
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=8)
    for x in ["train", "validation"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
class_names = image_datasets["train"].classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다림


# 학습 데이터의 배치를 얻음
inputs, classes = next(iter(dataloaders["train"]))

# 배치로부터 격자 형태의 이미지를 만듬
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])
# 합성곱 신경망 미세조정(finetuning)
# 미리 학습한 모델을 불러온 후 마지막의 완전히 연결된 계층을 초기화합니다.


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# 여기서 각 출력 샘플의 크기는 2로 설정합니다.
# 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(model_ft.parameters(), lr=args["learning_rate"], momentum=args["momentum"])

# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_ft, step_size=args["step_size"], gamma=args["gamma"]
)
# 학습 및 평가하기
# CPU에서는 15-25분 가량, GPU에서는 1분도 이내의 시간이 걸립니다.

model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=args["epochs"],
    dataloaders=dataloaders,
    dataset_sizes=dataset_sizes,
)
