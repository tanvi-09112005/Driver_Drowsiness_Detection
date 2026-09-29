"""
=============================================================================
AIDS2 DS Lab MiniProject: Driver Drowsiness Detection System
Real-Time Video Stream & Webcam Inference Pipeline (v3.0 - Precision Track)
=============================================================================
Key Improvements in v3.0:
1. Dynamic Eye Anchor: Uses Haar cascade for tight eye localization, then
   locks onto the exact ocular coordinates so closing your eyes never drops
   or distorts the crop.
2. Tight Ocular Aspect Ratio: Crops only the eye slit + eyelid (no cheek/forehead).
3. BGR -> RGB color conversion matching the training tensor.
4. On-screen PIP (Picture-in-Picture) preview of both eye crops in the corner
   so you can visually verify the exact image fed to the neural network.
5. Continuous non-stuttering audio alarm with persistent loop on Windows.
6. Wall-clock duration tracker with on-screen visual progress bar.
=============================================================================
"""

import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np
import tensorflow as tf

# =============================================================================
# 1. AUDIO ALARM SUBSYSTEM
# =============================================================================
ALARM_PLAYING = False
SOUND_BACKEND = None

if os.name == 'nt':
    import winsound
    SOUND_BACKEND = "winsound"
else:
    try:
        import pygame
        pygame.mixer.init()
        SOUND_BACKEND = "pygame"
    except Exception:
        SOUND_BACKEND = None

def start_alarm(sound_file="alarm.wav"):
    """Starts the alarm sound continuously in a non-blocking loop."""
    global ALARM_PLAYING
    if ALARM_PLAYING:
        return
    ALARM_PLAYING = True
    
    def _play_worker():
        try:
            if SOUND_BACKEND == "winsound":
                if os.path.exists(sound_file):
                    winsound.PlaySound(sound_file, winsound.SND_ASYNC | winsound.SND_LOOP | winsound.SND_FILENAME)
                else:
                    while ALARM_PLAYING:
                        winsound.Beep(2500, 300)
                        time.sleep(0.05)
            elif SOUND_BACKEND == "pygame" and os.path.exists(sound_file):
                pygame.mixer.Sound(sound_file).play(loops=-1)
        except Exception as e:
            print(f"[Sound Worker Warning]: {e}")

    threading.Thread(target=_play_worker, daemon=True).start()

def stop_alarm():
    """Stops the alarm sound immediately."""
    global ALARM_PLAYING
    if not ALARM_PLAYING:
        return
    ALARM_PLAYING = False
    try:
        if SOUND_BACKEND == "winsound":
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif SOUND_BACKEND == "pygame":
            pygame.mixer.stop()
    except Exception:
        pass

# =============================================================================
# 2. ARGUMENT PARSER
# =============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection - Precision Live Inference")
    parser.add_argument("--model", type=str, default="models/driver_drowsiness_custom_full85k.keras",
                        help="Path to trained .keras or .h5 model file")
    parser.add_argument("--webcam", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--duration", type=float, default=1.2,
                        help="Seconds of continuous eye closure to trigger alarm (default: 1.2s)")
    parser.add_argument("--alarm", type=str, default="alarm.wav",
                        help="Path to alarm .wav sound file")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Probability threshold to classify eye as closed (default: 0.50)")
    return parser.parse_args()

# =============================================================================
# 3. HELPER: EXTRACT TIGHT EYE CROP
# =============================================================================
def extract_tight_crop(img, cx, cy, size):
    """Crops a square region centered at (cx, cy) with bounds safety."""
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None, (x1, y1, x2, y2)
    return crop, (x1, y1, x2, y2)

# =============================================================================
# 4. MAIN DETECTION PIPELINE
# =============================================================================
def main():
    args = parse_arguments()
    print("=" * 70)
    print("🚗 AIDS2 MiniProject: Real-Time Driver Drowsiness Detection System v3.0")
    print("=" * 70)

    # 1. Load Haar Cascades
    cascade_dir = "haar_cascade_files"
    
    # Face cascade (tilt-tolerant default)
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        face_cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_alt.xml")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    print(f"Loaded Face Cascade: {os.path.basename(face_cascade_path)}")

    # Eye cascade
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    if not os.path.exists(eye_cascade_path):
        eye_cascade_path = os.path.join(cascade_dir, "haarcascade_lefteye_2splits.xml")
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    print(f"Loaded Eye Cascade:  {os.path.basename(eye_cascade_path)}")

    # 2. Load Trained Deep Learning Model
    print(f"Loading Neural Model: {args.model}")
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    model = tf.keras.models.load_model(args.model)
    print(f"Model loaded successfully! Input shape: {model.input_shape}")

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {args.webcam}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[INFO] Live camera feed active.")
    print(f"[INFO] Alarm triggers if eyes remain closed for >= {args.duration:.1f} seconds.")
    print("[INFO] Press 'q' or 'ESC' to exit.\n")

    # Tracking & State Variables
    last_face = None
    face_lost_counter = 0
    MAX_FACE_LOST = 20 # Keep tracking head during tilts for up to 20 frames

    # Eye Anchor Centers relative to face ROI (fractional: x/fw, y/fh)
    # Default initial biometric priors:
    anchor_left = (0.30, 0.38)   # Left eye center
    anchor_right = (0.70, 0.38)  # Right eye center
    anchor_calibrated = False

    closed_start_time = None
    closed_duration = 0.0
    fps = 0.0
    prev_time = time.time()
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Exponential moving average for smooth probabilities (debouncing)
    smooth_left_prob = 0.0
    smooth_right_prob = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------------------------------------------------
        # STEP A: Tilt-Tolerant Face Detection & Memory Persistence
        # -----------------------------------------------------------------
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(100, 100))

        active_face = None
        if len(faces) > 0:
            active_face = max(faces, key=lambda b: b[2] * b[3])
            last_face = active_face
            face_lost_counter = 0
        elif last_face is not None and face_lost_counter < MAX_FACE_LOST:
            active_face = last_face
            face_lost_counter += 1

        left_crop_disp = None
        right_crop_disp = None
        left_eye_closed = False
        right_eye_closed = False
        eyes_detected = False

        if active_face is not None:
            fx, fy, fw, fh = active_face
            
            # Draw Face Box
            box_col = (0, 220, 255) if face_lost_counter == 0 else (0, 160, 200)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_col, 2)
            cv2.putText(frame, "Driver Face", (fx, fy - 8), font, 0.6, box_col, 2)

            face_gray = gray[fy:fy + fh, fx:fx + fw]
            face_color = frame[fy:fy + fh, fx:fx + fw]

            # -------------------------------------------------------------
            # STEP B: High-Precision Dynamic Eye Anchor Localization
            # -------------------------------------------------------------
            # Search eyes in upper ocular band (20% to 55% of face height)
            ocular_band_y1 = int(0.20 * fh)
            ocular_band_y2 = int(0.55 * fh)
            ocular_band_gray = face_gray[ocular_band_y1:ocular_band_y2, :]

            detected_eyes = eye_cascade.detectMultiScale(
                ocular_band_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20), maxSize=(int(fw*0.35), int(fh*0.35))
            )

            # Refine anchor positions if open eyes are cleanly detected
            left_eyes_found = []
            right_eyes_found = []
            for (ex, ey_band, ew, eh) in detected_eyes:
                ey = ey_band + ocular_band_y1
                center_x = (ex + ew / 2.0) / fw
                center_y = (ey + eh / 2.0) / fh
                
                # Partition by mid-face line (0.50)
                if center_x < 0.48:
                    left_eyes_found.append((center_x, center_y))
                elif center_x > 0.52:
                    right_eyes_found.append((center_x, center_y))

            if len(left_eyes_found) > 0:
                best_lx, best_ly = left_eyes_found[0]
                # Smooth anchor update
                anchor_left = (0.8 * anchor_left[0] + 0.2 * best_lx, 0.8 * anchor_left[1] + 0.2 * best_ly)
                anchor_calibrated = True

            if len(right_eyes_found) > 0:
                best_rx, best_ry = right_eyes_found[0]
                anchor_right = (0.8 * anchor_right[0] + 0.2 * best_rx, 0.8 * anchor_right[1] + 0.2 * best_ry)
                anchor_calibrated = True

            # -------------------------------------------------------------
            # STEP C: Tight Ocular Cropping & Neural Classification
            # -------------------------------------------------------------
            # Crop size proportional to face width (tight ocular frame, typically ~22% of face width)
            eye_box_size = int(fw * 0.24)

            # 1. Left Eye Crop
            lcx = int(anchor_left[0] * fw)
            lcy = int(anchor_left[1] * fh)
            left_roi, (lx1, ly1, lx2, ly2) = extract_tight_crop(face_color, lcx, lcy, eye_box_size)

            if left_roi is not None and left_roi.size > 0:
                # Convert BGR -> RGB for neural input
                left_rgb = cv2.cvtColor(left_roi, cv2.COLOR_BGR2RGB)
                left_resized = cv2.resize(left_rgb, (64, 64))
                left_crop_disp = cv2.resize(left_roi, (80, 80)) # for PIP preview
                
                left_tensor = np.expand_dims(left_resized.astype(np.float32) / 255.0, axis=0)
                raw_pred_l = model.predict(left_tensor, verbose=0)[0]
                
                # Debounced probability
                smooth_left_prob = 0.65 * smooth_left_prob + 0.35 * float(raw_pred_l[0])
                left_eye_closed = smooth_left_prob >= args.threshold

                col_l = (0, 0, 255) if left_eye_closed else (0, 255, 0)
                cv2.rectangle(face_color, (lx1, ly1), (lx2, ly2), col_l, 2)
                cv2.putText(face_color, f"L: {smooth_left_prob*100:.0f}%", (lx1, ly1 - 5), font, 0.45, col_l, 1)

            # 2. Right Eye Crop
            rcx = int(anchor_right[0] * fw)
            rcy = int(anchor_right[1] * fh)
            right_roi, (rx1, ry1, rx2, ry2) = extract_tight_crop(face_color, rcx, rcy, eye_box_size)

            if right_roi is not None and right_roi.size > 0:
                right_rgb = cv2.cvtColor(right_roi, cv2.COLOR_BGR2RGB)
                right_resized = cv2.resize(right_rgb, (64, 64))
                right_crop_disp = cv2.resize(right_roi, (80, 80))
                
                right_tensor = np.expand_dims(right_resized.astype(np.float32) / 255.0, axis=0)
                raw_pred_r = model.predict(right_tensor, verbose=0)[0]
                
                smooth_right_prob = 0.65 * smooth_right_prob + 0.35 * float(raw_pred_r[0])
                right_eye_closed = smooth_right_prob >= args.threshold

                col_r = (0, 0, 255) if right_eye_closed else (0, 255, 0)
                cv2.rectangle(face_color, (rx1, ry1), (rx2, ry2), col_r, 2)
                cv2.putText(face_color, f"R: {smooth_right_prob*100:.0f}%", (rx1, ry1 - 5), font, 0.45, col_r, 1)

            eyes_detected = True

        # -----------------------------------------------------------------
        # STEP D: Wall-Clock Drowsiness Tracking
        # -----------------------------------------------------------------
        # Both eyes closed, OR average probability of closure >= threshold
        avg_closed_prob = (smooth_left_prob + smooth_right_prob) / 2.0
        eyes_shut = eyes_detected and (
            (left_eye_closed and right_eye_closed) or 
            (avg_closed_prob >= args.threshold)
        )

        if eyes_shut:
            if closed_start_time is None:
                closed_start_time = time.time()
            closed_duration = time.time() - closed_start_time
        else:
            closed_start_time = None
            closed_duration = max(0.0, closed_duration - 0.25)

        is_drowsy = closed_duration >= args.duration

        # -----------------------------------------------------------------
        # STEP E: Emergency Alarm & HUD
        # -----------------------------------------------------------------
        if is_drowsy:
            start_alarm(args.alarm)
            border_thick = int(8 + 4 * np.sin(time.time() * 14))
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), max(4, border_thick))
            
            # Prominent Red Alert Banner
            cv2.rectangle(frame, (30, h//2 - 45), (w - 30, h//2 + 45), (0, 0, 220), -1)
            cv2.putText(frame, "DROWSINESS ALERT! WAKE UP!", (55, h//2 + 12), font, 1.05, (255, 255, 255), 3)
        else:
            stop_alarm()

        # Top Status Banner
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        
        status_text = "DROWSY / SLEEPING!" if is_drowsy else ("ATTENTIVE" if eyes_detected else "SEARCHING DRIVER...")
        status_color = (0, 0, 255) if is_drowsy else ((0, 255, 0) if eyes_detected else (0, 165, 255))
        cv2.putText(frame, f"STATUS: {status_text}", (15, 36), font, 0.72, status_color, 2)

        # Timer & Closure Progress Bar
        bar_x = w - 340
        cv2.putText(frame, f"Closed: {closed_duration:.1f}s / {args.duration:.1f}s", (bar_x, 24), font, 0.5, (220, 220, 220), 1)
        
        progress = min(1.0, closed_duration / args.duration)
        bar_len = int(progress * 180)
        cv2.rectangle(frame, (bar_x, 32), (bar_x + 180, 46), (60, 60, 60), -1)
        bar_col = (0, 0, 255) if progress >= 0.8 else ((0, 165, 255) if progress >= 0.4 else (0, 255, 0))
        cv2.rectangle(frame, (bar_x, 32), (bar_x + bar_len, 46), bar_col, -1)
        
        # FPS display
        cv2.putText(frame, f"{fps:.1f} FPS", (w - 70, 24), font, 0.45, (170, 170, 170), 1)

        # -----------------------------------------------------------------
        # STEP F: Picture-in-Picture (PIP) Live Eye Inspection Preview
        # -----------------------------------------------------------------
        # Shows the exact crops sent to the CNN in the bottom corner
        if left_crop_disp is not None and right_crop_disp is not None:
            pip_y = h - 95
            pip_w, pip_h = 80, 80
            
            # Left Eye PIP
            pip_lx = w - 180
            frame[pip_y:pip_y+pip_h, pip_lx:pip_lx+pip_w] = left_crop_disp
            cv2.rectangle(frame, (pip_lx, pip_y), (pip_lx+pip_w, pip_y+pip_h), (0, 0, 255) if left_eye_closed else (0, 255, 0), 2)
            cv2.putText(frame, "L Crop", (pip_lx, pip_y - 4), font, 0.35, (220, 220, 220), 1)

            # Right Eye PIP
            pip_rx = w - 90
            frame[pip_y:pip_y+pip_h, pip_rx:pip_rx+pip_w] = right_crop_disp
            cv2.rectangle(frame, (pip_rx, pip_y), (pip_rx+pip_w, pip_y+pip_h), (0, 0, 255) if right_eye_closed else (0, 255, 0), 2)
            cv2.putText(frame, "R Crop", (pip_rx, pip_y - 4), font, 0.35, (220, 220, 220), 1)

        cv2.imshow("AIDS2 MiniProject - Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed cleanly.")

if __name__ == "__main__":
    main()
