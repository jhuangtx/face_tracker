import cv2
import dlib
import tkinter as tk
from PIL import Image, ImageTk
import sounddevice as sd
import numpy as np
import pyvirtualcam

smiley_closed_img = cv2.imread("smiley.png", cv2.IMREAD_UNCHANGED)
smiley_open_img = cv2.imread("smiley_open.png", cv2.IMREAD_UNCHANGED)

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

smiley_w, smiley_h = 255, 255

last_x, last_y = 200, 200

audio_data = np.zeros(1136)

def audio_callback(indata, frames, time, status):
    global talking, audio_data
    volume_norm = np.linalg.norm(indata) * 10
    audio_data[-len(indata):] = indata[:, 0]
    if volume_norm > 3:
        talking = True
    else:
        talking = False

stream = sd.InputStream(device=2, callback=audio_callback)
stream.start()

def on_closing():
    global running
    running = False
    window.destroy()
    cap.release()
    cam.stop()

window = tk.Tk()
window.title("Video with Smiley Overlay")
window.protocol("WM_DELETE_WINDOW", on_closing)

lmain = tk.Label(window)
lmain.pack()

running = True
talking = False

alpha = 0.5

frame_height, frame_width = 480, 640  # Assuming 640x480 for now, modify as needed

with pyvirtualcam.Camera(frame_width, frame_height, 30, fmt=pyvirtualcam.PixelFormat.RGBA) as cam:  # Assuming 20 FPS, modify as needed
    def show_frame():
        global last_x, last_y, talking, audio_data

        if not running:
            return

        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        black_frame = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

        if len(faces) > 0:
            face = faces[0]
            x, y = face.left(), face.top()
            x = int(alpha * x + (1 - alpha) * last_x)
            y = int(alpha * y + (1 - alpha) * last_y)
            last_x, last_y = x, y

        else:
            x = int(frame_width // 2 - smiley_w // 2)
            y = int(frame_height // 2 - smiley_h // 2)
            x = int(alpha * x + (1 - alpha) * last_x)
            y = int(alpha * y + (1 - alpha) * last_y)
            last_x, last_y = x, y

        smiley_img = smiley_open_img if talking else smiley_closed_img
        x = max(0, min(x, frame_width - smiley_w))
        y = max(0, min(y, frame_height - smiley_h))

        resized_smiley = cv2.resize(smiley_img, (smiley_w, smiley_h))
        mask = resized_smiley[:, :, 3]
        roi = black_frame[y:y+smiley_h, x:x+smiley_w]
        for c in range(0, 4):
            roi[:, :, c] = roi[:, :, c] * (1 - mask / 255.0) + resized_smiley[:, :, c] * (mask / 255.0)

        black_frame[y:y+smiley_h, x:x+smiley_w] = roi

        # Add code to draw audio waveform
        waveform_y = 450
        waveform_length = 200
        midpoint = frame_width // 2
        waveform_x_start = midpoint - (waveform_length // 2)
        waveform_x_end = midpoint + (waveform_length // 2)
        cv2.line(black_frame, (waveform_x_start, waveform_y), (waveform_x_end, waveform_y), (255, 255, 255, 255), 1)
        
        for i in range(len(audio_data) - 1):
            x1 = int(np.interp(i, [0, len(audio_data)], [waveform_x_start, waveform_x_end]))
            x2 = int(np.interp(i + 1, [0, len(audio_data)], [waveform_x_start, waveform_x_end]))
            y1 = waveform_y + int(audio_data[i] * 100)
            y2 = waveform_y + int(audio_data[i + 1] * 100)
            cv2.line(black_frame, (x1, y1), (x2, y2), (0, 255, 0, 255), 1)

        img = Image.fromarray(black_frame, 'RGBA')
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)

        # Send frame to virtual webcam
        # rgb_frame = black_frame[:, :, :3]
        # cam.send(rgb_frame)
        cam.send(black_frame)
        cam.sleep_until_next_frame()

    show_frame()
    window.mainloop()
