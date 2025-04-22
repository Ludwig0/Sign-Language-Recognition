import cv2
import mediapipe as mp
import threading
import time
from tkinter import Tk, Label
from PIL import Image, ImageTk
from model import DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
from architecture import MAX_NUM_HANDS, MODEL_COMPLEXITY

class HandGestureUI:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.root = Tk()
        self.root.title("Hand Gesture Recognition")
        self.label = Label(self.root)
        self.label.pack()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.running = True

    def start(self):
        thread = threading.Thread(target=self.video_loop)
        thread.daemon = True
        thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.mainloop()

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im_pil)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    HandGestureUI().start()