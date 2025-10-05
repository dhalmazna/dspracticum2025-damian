# 🐾 Animal Classifier - HuggingFace Spaces Deployment

This is a FastAI-based animal classifier that can identify 5 different animals: cats, dogs, rabbits, capybaras, and owls.

## Files for Deployment

- `app.py` - Main Gradio application
- `requirements.txt` - Python dependencies
- `model.pkl` - Trained FastAI model
- `example1.jpg`, `example3.jpg` - Example images
- `README.md` - This file

## Deployment Instructions

### Option 1: HuggingFace Spaces (Recommended)

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free tier)
   - **Visibility**: Public
4. Upload all files from this folder
5. The space will automatically build and deploy

### Option 2: Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## Model Information

- **Architecture**: ResNet18 (fine-tuned with FastAI)
- **Dataset**: Custom animal dataset with 636 training images
- **Classes**: 5 (capybara, cat, dog, owl, rabbit)
- **Training Framework**: FastAI + PyTorch
- **Deployment**: CPU-optimized for HuggingFace Spaces

## Troubleshooting

If you encounter pickle loading errors:

1. Ensure the model was exported with CPU compatibility
2. Check that torch is available and working
3. Verify all dependencies are installed correctly

## Created By

Data Science Practicum 2025 - Damian
