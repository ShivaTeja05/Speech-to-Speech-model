import os
import urllib.request

# Configuration
base_dir = "piper tamil model /tts_dojo/PRETRAINED_CHECKPOINTS/default"
urls = {
    "M_voice/medium": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt?download=true",
    # "M_voice/high": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/high/epoch%3D2218-step%3D838782.ckpt?download=true" # Skipping high for speed
}

def get_filename_from_url(url):
    filename = url.split("?")[0].split("/")[-1]
    filename = filename.replace("%3D", "=")
    return filename

for subpath, url in urls.items():
    target_dir = os.path.join(base_dir, subpath)
    os.makedirs(target_dir, exist_ok=True)
    
    filename = get_filename_from_url(url)
    dest_path = os.path.join(target_dir, filename)
    
    print(f"Downloading to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded {filename}")
        
        # Also create the language var file
        lang_var_file = os.path.join(base_dir, ".ESPEAK_LANGUAGE")
        with open(lang_var_file, "w") as f:
            f.write("generic")
            
    except Exception as e:
        print(f"Failed to download: {e}")
