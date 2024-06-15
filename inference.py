import os.path

import torch
import torch.nn as nn
from datasets import AnimalDataset
from models import AdvancedCNN
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import shutil
from torchsummary import summary
from torchvision.models import resnet18, ResNet18_Weights
import cv2
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser("Test arguments")
    parser.add_argument("--image_path", "-s", type=str)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint-path", "-t", type=str, default="trained_models/animals/best.pt")
    args = parser.parse_args()
    return args

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
    ])
    image = cv2.imread(args.image_path)
    image = cv2.convert
    # model = AdvancedCNN(num_classes=10).to(device)   # Our own model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model.to(device)

 


if __name__ == '__main__':
    args = get_args()
    inference(args)
