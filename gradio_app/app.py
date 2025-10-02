from pathlib import Path

import gradio as gr
from fastai.vision.all import *

# Load the trained model
learn = load_learner("model.pkl")

# Extract categories (class labels) from the model
categories = learn.dls.vocab


# Function to classify an image
def classify_image(img):
    """
    Classify an uploaded image using the trained model.

    Args:
        img: PIL Image or numpy array

    Returns:
        dict: Dictionary mapping class names to their probabilities
    """
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


# Define examples - you can add paths to example images here
examples = [
    "example1.jpg",  # Replace with actual example images
    "example2.jpg",
]

# Create Gradio interface
title = "🐾 Animal Classifier"
description = """
This model classifies images into 5 animal categories:
- 🐱 Cat
- 🐶 Dog  
- 🐰 Rabbit
- 🦫 Capybara
- 🦉 Owl

Upload an image or try one of the examples!
"""

article = """
### Model Information
- **Architecture**: ResNet18 (fine-tuned)
- **Dataset**: Custom animal dataset with 5 classes
- **Framework**: FastAI + PyTorch

Created as part of Data Science Practicum 2025.
"""

# Create and launch the interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=examples if Path(examples[0]).exists() else None,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
