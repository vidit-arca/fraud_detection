import numpy as np
from PIL import Image
import os

def create_image(path):
    # Create a random RGB image
    data = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    img.save(path)

os.makedirs('data/train_genuine', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

print("Creating dummy training images...")
for i in range(5):
    create_image(f'data/train_genuine/genuine_{i}.jpg')

print("Creating dummy test image...")
create_image('data/test/test_sample.jpg')
