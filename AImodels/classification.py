import torch
from torchvision import transforms
from PIL import Image
import os
class EyeDiseaseInference:
    def __init__(self, class_names=None, device=None):
        """
        Initialize the inference class with the model, class names, and device.

        Args:
            model_path (str): Path to the trained model file.
            class_names (list): List of class names for the labels.
            device (torch.device): Device for inference (CPU or GPU). If None, automatically selects.
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(os.path.abspath('AImodels/Weight/Cataract/Cataract_Model_efficientnet_b0_best.pt'), map_location=self.device)
        self.model.eval()  # Set the model to evaluation mode
        self.class_names = ['Normal', 'Cataract', 'Glaucoma','Retina']

        # Define a standard preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size as per model requirements
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])

    def preprocess_image(self, image_path):
        """
        Preprocess an image for model inference.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image
        return self.transform(image).unsqueeze(0)  # Add batch dimension

    def predict(self, image_path):
        """
        Predict the class of an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Prediction results including class name, label, and confidence score.
        """
        image_tensor = self.preprocess_image(image_path).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_label].item()

        result = {
            "image_path": image_path,
            "predicted_label": self.class_names[predicted_label] if self.class_names else predicted_label,
            "confidence_score": confidence
        }
        return result



import torch.nn.functional as F

class BrainTumorDiseaseInference:
    def __init__(self, device=None):
        """
        Initialize the inference class with the model, device, and pre-trained weights.

        Args:
            model_name (str): Name of the model (e.g., 'resnet50', 'mobilenetv2').
            model_path (str): Path to the trained model weights.
            device (torch.device): Device for inference (CPU or GPU). If None, automatically selects.
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(os.path.abspath('AImodels/Weight/BrainTumor classification/Modelefficientnet_b0_BrainTumorClassification.pt')) # Load the model architecture
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(self.device)  # Move the model to the specified device
        # Define a standard preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size as per model requirements
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])

    def preprocess_image(self, image_path):
        """
        Preprocess an image for model inference.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image
        return self.transform(image).unsqueeze(0)  # Add batch dimension

    def predict(self, image_path):
        """
        Predict the class of an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Prediction results including confidence score and label.
        """
        # Preprocess the image
        image_tensor = self.preprocess_image(image_path).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)  # Raw model output (logits)
            confidence_score = torch.sigmoid(outputs).item()  # Sigmoid to map to [0, 1]
            predicted_label = 'Tumor' if confidence_score > 0.5 else 'Normal'  # Binary decision

            # Adjust confidence score based on predicted class
            if predicted_label == 'Tumor':  # Class 1
                final_confidence_score = confidence_score
            else:  # Class 0
                final_confidence_score = 1 - confidence_score

        # Create result dictionary
        result = {
            "image_path": image_path,
            "confidence_score": final_confidence_score,
            "predicted_label": predicted_label
        }
        return result





class OralDiseaseInference:
    def __init__(self, device=None):
        """
        Initialize the inference class with the model, class names, and device.

        Args:
            model_path (str): Path to the trained model file.
            class_names (list): List of class names for the labels.
            device (torch.device): Device for inference (CPU or GPU). If None, automatically selects.
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(os.path.abspath('AImodels/Weight/BrainTumor classification/Modelefficientnet_b0_BrainTumorClassification.pt'))
        self.model.eval()  # Set the model to evaluation mode
        self.class_names = ['Calculus', 'Ginggivits', 'hypodonotia']

        # Define a standard preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size as per model requirements
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])

    def preprocess_image(self, image_path):
        """
        Preprocess an image for model inference.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image
        return self.transform(image).unsqueeze(0)  # Add batch dimension

    def predict(self, image_path):
        """
        Predict the class of an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Prediction results including class name, label, and confidence score.
        """
        image_tensor = self.preprocess_image(image_path).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_label].item()

        result = {
            "image_path": image_path,
            "predicted_label": self.class_names[predicted_label] if self.class_names else predicted_label,
            "confidence_score": confidence
        }
        return result




