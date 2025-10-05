#!/usr/bin/env python3
import os
import torch
from pathlib import Path

# Set up the same environment as the app
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False
torch.backends.cudnn.enabled = False

try:
    from fastai.vision.all import *
    defaults.device = torch.device('cpu')
    
    print("✅ FastAI imported successfully")
    
    # Test model loading
    print("\n🔄 Testing model loading...")
    
    try:
        learn = load_learner("model.pkl", cpu=True)
        print("✅ Model loaded successfully with load_learner")
        print(f"📊 Classes: {learn.dls.vocab}")
        print(f"🏗️ Architecture: {learn.arch}")
        print(f"💻 Device: {next(learn.model.parameters()).device}")
        
        # Test a quick prediction
        print("\n🧪 Testing prediction...")
        test_img_path = Path("example1.jpg")
        if test_img_path.exists():
            pred, idx, probs = learn.predict(test_img_path)
            print(f"✅ Prediction test successful: {pred}")
        else:
            print("⚠️ No example image found, skipping prediction test")
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ General error: {e}")
    import traceback
    traceback.print_exc()
