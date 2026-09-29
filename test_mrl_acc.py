import tensorflow as tf
import glob, os, cv2
from PIL import Image
import numpy as np

model_path = r'C:\Users\Tanvi Khadatkar\.gemini\antigravity\scratch\Driver_Drowsiness_Detection\models\driver_drowsiness_custom_full85k.keras'
model = tf.keras.models.load_model(model_path)

base = r'C:\Users\Tanvi Khadatkar\.gemini\antigravity\scratch\Driver_Drowsiness_Detection\dataset_mrl\mrlEyes_2018_01'
all_files = glob.glob(os.path.join(base, '**', '*.png'), recursive=True)

# MRL format: s0001_00001_0_0_0_0_0_01.png -> parts[4] is eye state
closed = [f for f in all_files if os.path.basename(f).split('_')[4] == '0'][:10]
opened = [f for f in all_files if os.path.basename(f).split('_')[4] == '1'][:10]

print('=== 10 REAL MRL CLOSED EYES ===')
correct_c = 0
for p in closed:
    img = Image.open(p).convert('RGB').resize((64, 64))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    pred = model.predict(arr, verbose=0)[0]
    is_correct = pred[0] > pred[1]
    if is_correct: correct_c += 1
    tag = "OK" if is_correct else "WRONG"
    print(f'  {os.path.basename(p)} -> P(Closed)={pred[0]:.3f}, P(Open)={pred[1]:.3f} | {tag}')

print(f'\nClosed accuracy: {correct_c}/10')

print('\n=== 10 REAL MRL OPEN EYES ===')
correct_o = 0
for p in opened:
    img = Image.open(p).convert('RGB').resize((64, 64))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    pred = model.predict(arr, verbose=0)[0]
    is_correct = pred[1] > pred[0]
    if is_correct: correct_o += 1
    tag = "OK" if is_correct else "WRONG"
    print(f'  {os.path.basename(p)} -> P(Closed)={pred[0]:.3f}, P(Open)={pred[1]:.3f} | {tag}')

print(f'\nOpen accuracy: {correct_o}/10')
