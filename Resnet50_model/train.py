import os
import sys
import json
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from model import resnet50


class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []

        df = pd.read_csv('./OCT/label_oct.csv')
        hashMap = {}
        for k, v in df.values:
            hashMap[k] = float(v / 500)

        filenames = os.listdir(image_folder)
        for filename in filenames:
            self.image_paths.append(os.path.join(image_folder, filename))
            self.labels.append(hashMap[filename])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_dataset = CustomImageDataset('./OCT_images', transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = CustomImageDataset('./OCT_images', transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet50()
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth

    # 为了快速跑通，我们先注释掉这三行。这就相当于让 AI 从“零基础”开始学，虽然预测会不准，但程序能跑起来才是我们的目的。
    # model_weight_path = "./resnet50-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 1)
    net.to(device)

    # define loss function
    loss_function = nn.MSELoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 10
    min_loss = float('inf')
    save_path = 'model.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            labels = labels.to(torch.float32).unsqueeze(1)  # 增加这一段 .unsqueeze(1)
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        val_loss = 0.0  # accumulate validate loss
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_labels = val_labels.to(torch.float32).unsqueeze(1) # 增加这一段 .unsqueeze(1)
                outputs = net(val_images.to(device))
                val_loss += loss_function(outputs, val_labels.to(device))

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        average_val_loss = val_loss / val_num
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f' %
              (epoch + 1, running_loss / train_steps, average_val_loss))

        if average_val_loss < min_loss:
            min_loss = average_val_loss
            torch.save(net.state_dict(), save_path)

        print('--------------------------------------------')

    print('Finished Training')


if __name__ == '__main__':
    main()
