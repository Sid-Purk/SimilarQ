import os
import zipfile
import gdown
from app.config import MODEL_ID

MODEL_ZIP_PATH= "leetcode_unixcoder_final.zip"
MODEL_DIR=os.path.join(os.path.dirname(__file__),"leetcode_unixcoder_final")

def download_n_extract():
    if not os.path.exists(MODEL_DIR):
        print("Downloading model")
        url=f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url,MODEL_ZIP_PATH,quiet=False)
        with zipfile.ZipFile(MODEL_ZIP_PATH,"r") as f:
            f.extractall(os.path.dirname(__file__))
        os.remove(MODEL_ZIP_PATH)
        print("Model ready")

if __name__=="__main__":
    download_n_extract()