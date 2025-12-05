import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import resnet50
import csv
import torch.nn as nn

# Function to save features to CSV
def save_features_to_csv(features, filename, image_name):
    features = features.cpu().detach().numpy().flatten()
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([image_name] + features.tolist())

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self.model.eval()

        # Hook the activations and gradients
        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        # Ensure gradients and activations are reset for this image
        self.gradients = []
        self.activations = []

        # Forward pass
        output = self.model(input_image)
        pred = output.argmax(dim=1).item()

        # Zero the gradients and perform backpropagation
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradients and activations
        gradients = self.gradients[0].cpu().detach().numpy()
        activations = self.activations[0].cpu().detach().numpy()

        # Compute the mean of the gradients over each channel
        weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        # Create the heatmap
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

def visualize_gradcam(image_paths, model, target_layer, output_dir, csv_file):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Create Grad-CAM object
    gradcam = GradCAM(model, target_layer)

    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')  # Ensure image is RGB
        img_tensor = data_transform(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

        # Forward pass to get the features from avgpool
        avgpool_features = None
        def avgpool_hook(module, input, output):
            nonlocal avgpool_features
            avgpool_features = output

        hook_handle = model.avgpool.register_forward_hook(avgpool_hook)

        with torch.no_grad():
            output = model(img_tensor)
            target_class = output.argmax().item()

        hook_handle.remove()

        # Save avgpool features to CSV
        save_features_to_csv(avgpool_features.squeeze(), csv_file, os.path.basename(image_path))

        cam = gradcam.generate_cam(img_tensor, target_class)

        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        original_img = np.array(img.resize((224, 224)))  # Resize to match the model input size
        if original_img.max() > 1:
            original_img = np.float32(original_img) / 255

        cam_image = heatmap + original_img
        cam_image = cam_image / np.max(cam_image)

        # Plot images side by side
        fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
        axarr[0].imshow(original_img)
        axarr[0].set_title('Original Image')
        axarr[0].axis('off')

        axarr[1].imshow(cam, cmap='jet')
        axarr[1].set_title('Grad-CAM')
        axarr[1].axis('off')

        axarr[2].imshow(cam_image)
        axarr[2].set_title('Overlay')
        axarr[2].axis('off')

        # Save the plot
        filename = os.path.basename(image_path)
        base_filename, _ = os.path.splitext(filename)
        save_path = os.path.join(output_dir, f'gradcam_{base_filename}.jpg')
        plt.savefig(save_path)
        plt.close()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 1. 保持和 train.py 一模一样的模型初始化逻辑
    model = resnet50()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 1)  # 手动替换最后一层
    model.to(device)

    # 2. 指向你刚才训练好的 model.pth (就在当前目录下)
    weights_path = './model.pth'
    # 加上 weights_only=True 告诉 PyTorch 我很安全，别唠叨了
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    # Assume target layer is the last convolutional layer
    target_layer = model.layer4[2].conv2

    image_dir = './OCT_images'
    output_dir = './GradCAM_Result'
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, 'features.csv')
    # Write header for CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image_Name'] + [f'Feature_{i}' for i in range(2048)])  # Assuming avgpool output size is 2048

    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                   os.path.isfile(os.path.join(image_dir, fname)) and (fname.lower().endswith('.jpg') or fname.lower().endswith('.bmp'))]
    visualize_gradcam(image_paths, model, target_layer, output_dir, csv_file)
