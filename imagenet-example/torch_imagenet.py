import torch
import torchvision
import os

# os.chdir("/home/woohyeon/study/imagenet-example")
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import json
from modules.dataloader import CustomDataset
from imutils import paths
from PIL import Image
import argparse
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
args = vars(ap.parse_args())
# args = {"image": "./images/space_shuttle.png"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./configuration/config.json", "r") as f:
    CONFIG = json.load(f)

with open("./configuration/imagenet_class_index.json", "r") as f:
    classes = json.load(f)

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ]
)

model_ft = models.vgg16(pretrained=True)
# num_ftrs = model_ft.classifier[-1].in_features
# model_ft.fc = nn.Linear(num_ftrs, 1000)

model_ft = model_ft.to(device)


test_loader = DataLoader(CustomDataset(args["image"], transform=trans), shuffle=False, batch_size=1)

with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model_ft(inputs)

        prob = F.softmax(outputs, dim=1)
        top_p, top_class = prob.topk(1, dim=1)


orig = cv2.imread(args["image"])
label, prob = classes[str(top_class.item())][1], top_p.item()
cv2.putText(
    orig,
    "Label: {}, {:.2f}%".format(label, prob * 100),
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)
cv2.imshow("Classification", orig)
cv2.waitKey(0)