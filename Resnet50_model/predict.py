import os
import json
import csv
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    filenames = os.listdir(r'D:\pythonProject\08\Black_Clefts\images\train')
    results = []
    for filename in filenames:
        img_path = os.path.join(r'D:\pythonProject\08\Black_Clefts\images\train', filename)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # create model
        model = resnet50(num_classes=1).to(device)

        # load model weights
        weights_path = "model_OSI.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # prediction
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            results.append({'文件名': filename, '预测结果': int(output.item() * 500)})


    # Output results to CSV file
    csv_file = 'predictions_train.csv'
    with open(csv_file, mode='w', newline='') as file:
        fieldnames = ['文件名', '预测结果']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        print(f"预测结果已保存到文件：{csv_file}")


if __name__ == '__main__':
    main()
