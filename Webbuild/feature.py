from flask import Blueprint, render_template, request, redirect, url_for, session, flash
import sys
import os
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AImodels.segmantation import SegmentationInference
from AImodels.classification import OralDiseaseInference, BrainTumorDiseaseInference, EyeDiseaseInference
from AImodels.detection import PVTModel

oralInference=OralDiseaseInference()
brainTumoreInference=BrainTumorDiseaseInference()
eyeDiseaseInference=EyeDiseaseInference()
pvtModel=PVTModel()
segInferecne=SegmentationInference()

featurebp = Blueprint('feature', __name__)

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

def plot_image_with_bboxes(encode_image, bboxes, classifiers, save_image=False, output_dir='static/Results', output_filename='prediction_result.png'):
    """
    Plots an image with bounding boxes and labels and optionally saves the result.

    Parameters:
    - encode_image: PyTorch tensor of the encoded image (C, H, W).
    - bboxes: List or tensor of bounding boxes (N, 4), where each box is (x_min, y_min, width, height).
    - classifiers: Tensor of classifier outputs for the corresponding bboxes.
    - save_image: Boolean flag to save the image or not.
    - output_dir: Directory to save the output image.
    - output_filename: Name of the output file.
    """
    # Convert the image tensor to NumPy array for visualization
    image_tensor = encode_image.squeeze()
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
    #image_np to [0.255]
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    # Prepare plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_np)
    ax.axis('off')

    # Get predicted labels
    pred_labels = classifiers

    c = 0
    for i, bbox in enumerate(bboxes[0]):  # Assuming bboxes is nested (e.g., [batch, N, 4])
        x_min, y_min, width, height = bbox
        label = pred_labels[0][c]
        if label == 0:
            continue  # Skip if no tumor detected
        label = 'Tumor' if label == 2 else 'No Tumor'
        c = c + 1

        # Draw the bounding box
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # Add the label text
        ax.text(
            x_min, y_min - 5, f"Pred: {label}",
            color='red', fontsize=8, backgroundcolor='white'
        )

    # Title for the plot
    plt.title("Red: Predictions Tumor's Zone")

    # Save the plot if save_image flag is True
    if save_image:
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)  # Save with tight layout
        print(f"Image saved to {output_path}")
    else:
        plt.show()  # Show the plot if not saving
     
    # Close the plot to free memory
    plt.close()
    return output_path












ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Max file size 16MB
UPLOAD_FOLDER = 'static/uploads'
Result_FOLDER = 'static/Results'
#check Result folder exists
os.makedirs(Result_FOLDER, exist_ok=True)
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if a file is an allowed type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@featurebp.route('/describe_image', methods=['POST'])
def describe_image():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            file.save(filepath)
            if session.get('modality') == 'oral':
                results=oralInference.predict(filepath)
                filepath=filepath.replace("\\", "/")
                file_name = os.path.basename(filepath)
                file_name='uploads/'+file_name
                return render_template('ResearchFeatures/Result.html', results=results, img_path=file_name)
            

            if session.get('modality') == 'cataract':
                results=eyeDiseaseInference.predict(filepath)
                filepath=filepath.replace("\\", "/")
                file_name = os.path.basename(filepath)
                file_name='uploads/'+file_name
                return render_template('ResearchFeatures/Result.html', results=results, img_path=file_name)
            
            
            
            if session.get('modality') == 'brain_class':
                results=brainTumoreInference.predict(filepath)
                filepath=filepath.replace("\\", "/")
                file_name = os.path.basename(filepath)
                file_name='uploads/'+file_name
                return render_template('ResearchFeatures/Result.html', results=results, img_path=file_name)
            
            
            
            
            
            if session.get('modality') == 'retina':
                mask, image, result_path=segInferecne.predict(filepath)
                results={}
                results['image_path'] = result_path
                results['mask_path'] = mask
                results['image'] = image
                print('result_path',result_path)
                file_name = os.path.basename(result_path)
                file_name='Results/'+file_name
                #render to mainpage
                return redirect(url_for('static', filename=file_name))
            if session.get('modality') == 'brain_detect':
                pred_labels, bbox,x=pvtModel.forward(filepath)
                output_path = plot_image_with_bboxes(x, bbox, pred_labels, save_image=True, output_dir=Result_FOLDER, output_filename='detection.png')
                output_path_corrected = output_path.replace("\\", "/")
                print(output_path_corrected)
                file_name = os.path.basename(output_path_corrected)
                file_name='Results/'+file_name
                print(file_name)
                return redirect(url_for('static', filename=file_name))
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(url_for('feature.upload_image'))