from fastai.vision.all import *
import gradio as gr

# Load model
learn = load_learner('model.pkl')
categories = learn.dls.vocab

# Classification function
def classify_image(img):
    # Convert to PILImage if needed
    if not isinstance(img, PILImage):
        img = PILImage.create(img)
    _, _, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Gradio interface
image = gr.Image(type="pil", width=224, height=224)  # <- important: type="pil"
label = gr.Label()
examples = ['example1.jpg', 'example3.jpg']

intf = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    title="🐾 Animal Classifier",
    description="Upload an image to classify it as one of: " + ", ".join(categories),
    examples=examples)
intf.launch()