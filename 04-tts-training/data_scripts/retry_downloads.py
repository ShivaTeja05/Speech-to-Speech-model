import os
import urllib.request
import time

base_url = "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil/resolve/main/data/"
output_dir = "DATASET FOR TAMIL"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Checking for missing files in {output_dir}...")

for i in range(17):
    filename = f"train-{i:05d}-of-00017.parquet"
    url = base_url + filename
    dest = os.path.join(output_dir, filename)
    
    if not os.path.exists(dest):
        print(f"File missing: {filename}. Retrying download...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    else:
        # Check if file is empty (basic validation)
        if os.path.getsize(dest) < 1024:
             print(f"File {filename} seems corrupt (too small). Retrying...")
             try:
                urllib.request.urlretrieve(url, dest)
                print(f"Successfully downloaded {filename}")
             except Exception as e:
                print(f"Failed to download {filename}: {e}")

print("Verification complete.")
