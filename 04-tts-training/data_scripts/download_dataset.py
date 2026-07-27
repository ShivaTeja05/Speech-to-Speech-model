import os
import urllib.request

base_url = "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil/resolve/main/data/"
output_dir = "DATASET FOR TAMIL"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Starting download to {output_dir}...")

for i in range(17):
    filename = f"train-{i:05d}-of-00017.parquet"
    url = base_url + filename
    dest = os.path.join(output_dir, filename)
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
