import os

# Disable TensorFlow and force PyTorch backend in Hugging Face Transformers
# This prevents Keras 3 conflicts when sentence-transformers or transformers are loaded
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
