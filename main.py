import tkinter as tk
import threading
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2

video_thread: threading.Thread
video_condition: threading.Condition = threading.Condition()
video_path: str = ""

video_capture: cv2.VideoCapture

video_paused = True

pos_line = 550

offset = 6

delay = 60

width_min = 80
height_min = 80


def catch(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


def load_video(path: str):
    global video_capture
    if len(path) == 0:
        video_capture = None
        return

    video_capture = cv2.VideoCapture(path)


def select_file(label: tk.Label) -> str:
    path = filedialog.askopenfilename()
    load_video(path)
    play_frame(label)
    return path


def create_view(window: tk.Tk) -> tk.Label:
    window.minsize(800, 600)

    right_frame = tk.Frame(master=window, bg="gray", width=500, height=500)
    left_frame = tk.Frame(master=window, bg="darkgray", width=200, height=500)

    label = tk.Label(master=right_frame, text="Load image")
    label.pack(fill=tk.BOTH, expand=True)

    right_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
    left_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    button = tk.Button(master=left_frame, text=f"Load file", command=lambda: select_file(label))
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
    button = tk.Button(master=left_frame, text=f"Start", command=start)
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
    button = tk.Button(master=left_frame, text=f"Pause", command=pause)
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    return label


def process_frame(frame: np.ndarray) -> np.ndarray:
    return frame


subtract = cv2.bgsegm.createBackgroundSubtractorMOG()


def play_frame(label: tk.Label) -> bool:
    if not video_capture.isOpened():
        return False

    detect = []
    cars = 0
    ret, frame = video_capture.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtract.apply(blur)
    dilation = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    contour, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, pos_line), (1200, pos_line), (255, 127, 0), 3)
    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= width_min) and (h >= height_min)
        if not validate_contour:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        centre = catch(x, y, w, h)
        detect.append(centre)
        cv2.circle(frame, centre, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if (pos_line + offset) > y > (pos_line - offset):
                cars += 1
                cv2.line(frame, (25, pos_line), (1200, pos_line), (0, 127, 255), 3)
                detect.remove((x, y))
                print("Car is detected : " + str(cars))

    cv2.putText(frame, "DETECTED CARS: " + str(cars), (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
    cv2.imshow("Detecting", dilatada)
    if ret:
        process_frame(frame.copy())
        image = Image.fromarray(frame)
        image = image.resize((600, 400))
        image = ImageTk.PhotoImage(image)

        label.config(image=image)
        label.image = image

        return True
    else:
        return False


def play_video(label: tk.Label):
    video_condition.acquire()
    video_condition.wait()

    while True:
        if video_paused:
            video_condition.wait()
        if not play_frame(label):
            break


def start():
    global video_paused

    if not video_paused:
        return

    video_paused = False

    video_condition.acquire()
    video_condition.notify()
    video_condition.release()


def pause():
    global video_paused
    video_paused = True


if __name__ == "__main__":
    window = tk.Tk()
    label = create_view(window)

    video_thread = threading.Thread(target=play_video, args=(label,))
    video_thread.daemon = True
    video_thread.start()

    window.mainloop()