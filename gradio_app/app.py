import os
import torch
from pathlib import Path
import gradio as gr

# Force CPU usage to avoid GPU-related issues in deployment
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    from fastai.vision.all import *
    
    # Set CPU as default device
    torch.cuda.is_available = lambda: False
    torch.backends.cudnn.enabled = False
    defaults.device = torch.device('cpu')
    
    # Load the trained model
    learn = load_learner("model.pkl", cpu=True)
    
    # Extract categories (class labels) from the model
    categories = ['capybara', 'cat', 'dog', 'owl', 'rabbit']  # Hardcoded as fallback
    if hasattr(learn, 'dls') and hasattr(learn.dls, 'vocab'):
        categories = learn.dls.vocab
    
    print(f"✅ Model loaded successfully. Categories: {categories}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Create a dummy function for testing
    def dummy_predict(img):
        return {'cat': 0.5, 'dog': 0.3, 'rabbit': 0.1, 'owl': 0.05, 'capybara': 0.05}
    
    categories = ['capybara', 'cat', 'dog', 'owl', 'rabbit']


def classify_image(img):
    """
    Classify an uploaded image using the trained model.
    """
    try:
        if 'learn' in globals():
            pred, idx, probs = learn.predict(img)
            return dict(zip(categories, map(float, probs)))
        else:
            # Fallback for testing
            return dummy_predict(img)
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return equal probabilities in case of error
        prob = 1.0 / len(categories)
        return {cat: prob for cat in categories}


# Define examples
example_files = ["example1.jpg", "example3.jpg"]
examples = [f for f in example_files if Path(f).exists()]

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
    examples=examples if examples else None,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
