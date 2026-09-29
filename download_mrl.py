import os
import sys
import ssl
import urllib.request
import zipfile
import time

def download_and_extract():
    url = "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
    zip_path = "mrlEyes_2018_01.zip"
    extract_to = "dataset_mrl"
    
    os.makedirs(extract_to, exist_ok=True)
    
    # Bypass self-signed university SSL certificate check
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    # 1. Download with progress reporting
    if not os.path.exists(zip_path):
        print(f"Connecting to {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as resp, open(zip_path, 'wb') as out_file:
            total_size = int(resp.headers.get('content-length', 0))
            downloaded = 0
            block_size = 1024 * 512 # 512 KB chunks
            t0 = time.time()
            last_print = 0
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                
                # Print progress every 1 second
                if time.time() - last_print > 1.0 or downloaded == total_size:
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024) if total_size > 0 else 326.0
                    pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                    elapsed = time.time() - t0
                    speed = downloaded_mb / elapsed if elapsed > 0 else 0
                    print(f"Downloading: {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s")
                    last_print = time.time()
        print("Download completed successfully!")
    else:
        print(f"Archive '{zip_path}' already exists.")

    # 2. Extract Archive
    print(f"Extracting '{zip_path}' into '{extract_to}/'...")
    t_start = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed in {time.time() - t_start:.2f} seconds!")

    # 3. Verify total images
    import glob
    all_pngs = glob.glob(f"{extract_to}/**/*.png", recursive=True)
    print(f"Total eye image files found in '{extract_to}': {len(all_pngs)}")

if __name__ == "__main__":
    download_and_extract()
