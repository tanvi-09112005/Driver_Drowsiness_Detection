"""
Script to create sample MRL-format eye images for testing the notebook offline.
"""
import os
import numpy as np
from PIL import Image

def generate_samples(base_dir="dataset_mrl", n_samples=300):
    open_dir = os.path.join(base_dir, "open")
    closed_dir = os.path.join(base_dir, "closed")
    os.makedirs(open_dir, exist_ok=True)
    os.makedirs(closed_dir, exist_ok=True)
    
    IMG_SIZE = 64
    for i in range(n_samples // 2):
        # Closed eye
        c_img = np.full((IMG_SIZE, IMG_SIZE, 3), 130 + np.random.randint(-10, 10), dtype=np.uint8)
        y = IMG_SIZE // 2 + np.random.randint(-2, 3)
        c_img[y-1:y+2, 12:IMG_SIZE-12] = 25
        c_fname = f"s00{i%37:02d}_{i:05d}_0_0_0_0_1_01.png"
        Image.fromarray(c_img).save(os.path.join(closed_dir, c_fname))

        # Open eye
        o_img = np.full((IMG_SIZE, IMG_SIZE, 3), 130 + np.random.randint(-10, 10), dtype=np.uint8)
        Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        dist = np.sqrt(((X - IMG_SIZE/2)/1.4)**2 + ((Y - IMG_SIZE/2)/0.8)**2)
        o_img[dist <= 18] = 215
        iris_dist = np.sqrt((X - IMG_SIZE/2)**2 + (Y - IMG_SIZE/2)**2)
        o_img[iris_dist <= 9] = 55
        o_img[iris_dist <= 4] = 10
        o_fname = f"s00{i%37:02d}_{i:05d}_0_0_1_0_1_01.png"
        Image.fromarray(o_img).save(os.path.join(open_dir, o_fname))
        
    print(f"Generated {n_samples} sample images in '{base_dir}'.")

if __name__ == "__main__":
    generate_samples()
