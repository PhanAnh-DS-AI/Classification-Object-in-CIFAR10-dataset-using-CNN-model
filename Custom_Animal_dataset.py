import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import transforms, ToTensor, Compose, Resize
from torchvision.datasets import ImageFolder
import os
import pickle
import numpy as np
import pandas as pd
import cv2
from PIL import Image


class AnimalDataset(Dataset):
    def __init__(self, root, is_train, transform=None):
        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]

        self.all_image_paths = []
        self.all_labels = []
        # Truy cập vào file hình ảnh của từng categories
        for index, category in enumerate(self.categories):
            category_path = os.path.join(data_path, category)
            for item in os.listdir(category_path):
                image_path = os.path.join(category_path, item)
                # print(os.path.isfile(image_path)) # Kiểm tra có phải là file hay không Đúng trả về True, và ngược lại
                # Cần tạo một list để lưu các ảnh của category
                self.all_image_paths.append(image_path)
                self.all_labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, item):
        image_path = self.all_image_paths[item]
        # image = cv2.imread(image_path) # có thể chọn 2 cách. 
        # Tuy nhiên cách Image thì tốt hơn vì Hàm resize chỉ chấp nhận PIL image và Tensor. Và nên để ToTensor trước Resize
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.all_labels[item]
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])
    dataset = AnimalDataset(root="D:/DL4CV/animals", is_train=True, transform=transform)
    #image, label = dataset[5000]
    #cv2.imshow(str(label),image)
    #cv2.waitKey(0)
    # DataLoader dùng để lấy nhiều bức ảnh cùng lúc và nhiều chức năng khác
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )

    for images, labels in dataloader:
        print(images.shape)
        print(labels)
        print("------------")