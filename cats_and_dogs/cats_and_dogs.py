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
# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화

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
    "gamma": gamma
}

print(args)

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([
            0.485, 0.456, 0.406
        ],
            [
            0.229, 0.224, 0.225
        ])
    ]),
    "validation": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([
            0.485, 0.456, 0.406
        ],
            [
            0.229, 0.224, 0.225
        ])
    ])
}
1+1

math.pow(2, 38)

data_dir = "../../../datasets/cats_and_dogs_small"
image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ["train", "validation"]}

dataloaders = {x: DataLoader(
    image_datasets[x], batch_size=4, shuffle=True, num_workers=8) for x in ["train", "validation"]}

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

# 모델 학습
# 학습률(learning rate) 관리(scheduling)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model
# 모델 예측값 시각화하기
# 일부 이미지에 대한 예측값을 보여주는 일반화된 함수입니다.


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
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
args = {
    "epochs": epochs,
    "learning_rate": lr,
    "momentum": momentum,
    "step_size": step_size,
    "gamma": gamma
}
optimizer_ft = optim.SGD(model_ft.parameters(),
                         lr=args["learning_rate"], momentum=args["momentum"])

# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_ft, step_size=args["step_size"], gamma=args["gamma"])
# 학습 및 평가하기
# CPU에서는 15-25분 가량, GPU에서는 1분도 이내의 시간이 걸립니다.

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=args["epochs"])
