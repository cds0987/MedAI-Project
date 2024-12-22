from transformers import AutoImageProcessor, PvtForImageClassification
import torch
import torch.nn as nn
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class PVTModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(PVTModel, self).__init__()
        self.backbone = PvtForImageClassification.from_pretrained("Zetatech/pvt-tiny-224")
        self.backbone.classifier = torch.nn.Identity()  # Remove existing classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # Assuming feature size is 256
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self.bbox = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # Bounding box output: (x, y, w, h)
        )
        self.device = 'cpu'
        state_dict_path = os.path.abspath('AImodels/Weight/BrainTumor detection/PvtBrainTumor_state_dict.pt')
        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.image_processor = AutoImageProcessor.from_pretrained("Zetatech/pvt-tiny-224")

    def process_image(self, image_path, device='cpu'):
        """Preprocess the input image."""
        self.device = device
        image = Image.open(image_path).convert("RGB")
        encoding = self.image_processor(images=image, return_tensors="pt")
        return encoding['pixel_values'].to(self.device), image

    def forward(self, image_path,device='cpu'):
        """Forward pass with postprocessing and visualization."""

        # Preprocess image
        x, original_image = self.process_image(image_path,device)
        self.to(self.device)
        self.eval()
        with torch.no_grad():
        # Get predictions
           logits = self.backbone(x).logits
           logits = logits.view(-1, 2, 256)

           classifier = self.classifier(logits)
           bbox = self.bbox(logits)
           pred_labels = classifier.argmax(dim=-1).cpu().numpy()
           bbox=bbox.cpu().numpy()
        return pred_labels, bbox,x


