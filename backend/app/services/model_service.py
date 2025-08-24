from sentence_transformers import SentenceTransformer
import threading
import os
from app.config import MODEL_PATH

_model_instance=None
_model_lock=threading.Lock()

def get_model():
    global _model_instance
    with _model_lock:
        if _model_instance is None:
            model_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"leetcode_unixcoder_final"))
            _model_instance = SentenceTransformer(model_path)
        return _model_instance