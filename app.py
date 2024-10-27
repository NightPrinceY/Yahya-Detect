import gradio as gr
import torch
from ultralytics import YOLO
from PIL import Image

# Load the trained YOLOv8 model
model = YOLO("yahya-detection.pt")

# Define the prediction function
def predict(image):
    results = model(image)  # Run YOLOv8 model on the uploaded image
    results_img = results[0].plot()  # Get image with bounding boxes
    return Image.fromarray(results_img)

# Create Gradio interface
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Image(type="pil"),
    title="Yahya Detect",
    description="Detect Yahya now!"
)

# Launch Gradio app
interface.launch()