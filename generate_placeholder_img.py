from PIL import Image, ImageDraw, ImageFont
import os

# Create a clean placeholder demo screenshot (800x450)
img = Image.new("RGB", (800, 450), color=(25, 28, 36))
draw = ImageDraw.Draw(img)

# Border
draw.rectangle([(8, 8), (791, 441)], outline=(60, 68, 88), width=2)

# Top Bar
draw.rectangle([(10, 10), (790, 50)], fill=(35, 39, 52))
draw.text((25, 22), "AIDS2 MiniProject - Driver Drowsiness Detection [Live Webcam Window]", fill=(200, 210, 230))

# Simulated Face Box
draw.rectangle([(250, 90), (550, 390)], outline=(0, 220, 255), width=3)
draw.text((255, 68), "Driver Face [Tracked]", fill=(0, 220, 255))

# Simulated Eyes
draw.rectangle([(290, 180), (370, 240)], outline=(0, 255, 100), width=2)
draw.text((295, 160), "L: Open (99%)", fill=(0, 255, 100))

draw.rectangle([(430, 180), (510, 240)], outline=(0, 255, 100), width=2)
draw.text((435, 160), "R: Open (98%)", fill=(0, 255, 100))

# Bottom HUD / Notification
draw.rectangle([(10, 400), (790, 440)], fill=(18, 20, 26))
draw.text((25, 412), "STATUS: ATTENTIVE", fill=(0, 255, 100))
draw.text((560, 412), "FPS: 62.4  |  Threshold: 1.2s", fill=(180, 190, 210))

# Center watermark
draw.rectangle([(180, 250), (620, 310)], fill=(20, 24, 32, 200), outline=(0, 180, 255))
draw.text((200, 265), "[ Replace this with your actual webcam screenshot ]", fill=(0, 200, 255))
draw.text((250, 285), "Save your webcam capture as 'demo_screenshot.png'", fill=(160, 170, 190))

img.save(r"C:\Users\Tanvi Khadatkar\.gemini\antigravity\scratch\Driver_Drowsiness_Detection\demo_screenshot.png")
print("Saved demo_screenshot.png placeholder!")
