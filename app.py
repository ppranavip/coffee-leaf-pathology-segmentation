import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Layer, Conv2D, Add
import tensorflow as tf

# Register and define custom layer
@register_keras_serializable()
class LightweightAttentionBlock(Layer):
    def __init__(self, num_filters, **kwargs):
        super(LightweightAttentionBlock, self).__init__(**kwargs)
        self.num_filters = num_filters

    def call(self, inputs, **kwargs):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'num_filters': self.num_filters})
        return config


# Initialize Flask app
app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model('model.keras', custom_objects={"LightweightAttentionBlock": LightweightAttentionBlock})

# Constants for image processing
IMG_H, IMG_W = 512, 1024
COLORMAP = [
    [0, 0, 0],       # Background
    [0, 176, 0],     # Leaf
    [255, 0, 0],     # Symptom
]


def preprocess_image(image_path, img_height, img_width):
    """Preprocess the input image for prediction."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (img_width, img_height))
    image = image / 255.0
    return np.expand_dims(image, axis=0)


def visualize_results(original_image_path, predicted_mask, colormap, save_path):
    """Visualize and save the original image and predicted mask."""
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Convert predicted mask to RGB
    mask_rgb = np.zeros((*predicted_mask.shape[:2], 3), dtype=np.uint8)
    for class_idx, color in enumerate(colormap):
        mask_rgb[predicted_mask == class_idx] = color

    # Calculate stress percentage
    red_pixels = np.sum(predicted_mask == 2)  # Red -> Symptom
    green_pixels = np.sum(predicted_mask == 1)  # Green -> Leaf (healthy)
    stress_percentage = (red_pixels / green_pixels) * 100 if green_pixels > 0 else 0

    # Save the result
    result_image = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, result_image)

    return stress_percentage


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'})
        if file:
            filename = file.filename.replace(' ', '_')
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Return filename for further processing
            return jsonify({'status': 'Processing', 'filename': filename})

    return render_template('index.html')


@app.route('/process_image/<filename>', methods=['GET'])
def process_image(filename):
    """Process the uploaded image and generate the segmentation result."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'File not found'})

    # Preprocess image
    processed_image = preprocess_image(file_path, IMG_H, IMG_W)

    # Predict the mask
    predicted_mask = model.predict(processed_image, verbose=0)[0]
    predicted_mask = np.argmax(predicted_mask, axis=-1)

    # Resize to original dimensions
    original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    original_height, original_width = original_image.shape[:2]
    predicted_mask = cv2.resize(predicted_mask.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Save results and calculate stress percentage
    result_filename = f'result_{filename}'
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    stress_percentage = visualize_results(file_path, predicted_mask, COLORMAP, result_path)

    uploaded_image_url = url_for('static', filename=f'uploads/{filename}')
    result_image_url = url_for('static', filename=f'results/{result_filename}')

    return jsonify({
        'status': 'Complete',
        'uploaded_image_url': uploaded_image_url,
        'result_image_url': result_image_url,
        'stress_percentage': round(stress_percentage, 2)
    })


if __name__ == '__main__':
    app.run(debug=True)
