import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch.nn as nn
import os
import torch
from transformers import SegformerForSemanticSegmentation

import torch
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(150, 128, kernel_size=4, stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1),)

        self.header = torch.nn.Sequential(
            torch.nn.Linear(64, 256),  # Adjusted from Segformer
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        # Pass input through Segformer backbone
        x = self.backbone(x)
        x = x.logits  # Extract the logits

        # Permute dimensions to (batch_size, height, width, channels)

        # Upsample to (512, 512)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)
        # Pass through header
        x = self.header(x)
        return x

class SegmentationInference:
    def __init__(self, device=None, output_dir="static/Results"):
        """
        Initialize the inference class.
        :param model: The trained segmentation model.
        :param device: The device (cpu or cuda) on which the model is loaded.
        :param output_dir: Directory where predicted images will be saved.
        """
        self.model = SegmentationModel()
        state_dict_path = os.path.abspath('AImodels/Weight/Retinal Segmantation/segmodel_best.pt')
        state_dict=torch.load(state_dict_path)
        self.model.load_state_dict(state_dict)
        device=device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode
 
        # Ensure the output directory exists
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.image_processor=AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    def predict(self, image_path):
        """
        Predict the segmentation mask for a given image path.
        :param image_path: Path to the input image.
        :return: Predicted mask (tensor).
        """
        # Load image
        image = Image.open(image_path).convert('RGB')  # Convert to RGB if it's a grayscale image
        # Apply transformation to the image
        input_image = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        input_image = input_image.to(self.device)
        # Make prediction
        with torch.no_grad():
            output = self.model(input_image)
            prediction = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
            prediction = (prediction > 0.5).cpu().squeeze(0).numpy()  # Threshold and move to CPU for plotting
        #save prediction
        result_path=self.save_prediction(prediction, 'Segmantation')
        result_path=result_path.replace("\\","/")
        return prediction, image,result_path

    def save_prediction(self, prediction, image_name):
        """
        Save the predicted mask as an image.
        :param prediction: The predicted binary mask.
        :param image_name: The name to save the predicted mask as.
        """
        prediction = prediction.squeeze()
        predicted_image = (prediction * 255).astype(np.uint8)
        predicted_image = Image.fromarray(predicted_image)  # Convert to PIL image
        save_path=os.path.join(self.output_dir, f"{image_name}_prediction.png")
        # Save the image
        predicted_image.save(save_path)
        print(f"Prediction saved to: {save_path}")
        return save_path

    def plot_results(self, prediction, input_image, image_name):
        """
        Plot the input image, ground truth, and prediction, and save the prediction image.
        :param prediction: The predicted mask.
        :param input_image: The input image (PIL Image).
        :param image_name: The name of the input image.
        """
        # Convert input image to numpy array for plotting
        input_image = np.array(input_image)

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot input image
        axs[0].imshow(input_image)
        axs[0].set_title("Input Image")
        axs[0].axis("off")


        # Plot prediction
        axs[1].imshow(prediction, cmap="gray")
        axs[1].set_title("Prediction")
        axs[1].axis("off")

        # Save the predicted mask image
        self.save_prediction(prediction, image_name)

        plt.show()