import os
import urllib.request

filename = "train-00003-of-00017.parquet"
url = "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil/resolve/main/data/" + filename
dest = os.path.join("DATASET FOR TAMIL", filename)

if os.path.exists(dest):
    print(f"Removing corrupt file {dest}...")
    os.remove(dest)

print(f"Redownloading {filename}...")
try:
    urllib.request.urlretrieve(url, dest)
    print("Download successful.")
except Exception as e:
    print(f"Download failed: {e}")
