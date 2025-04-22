import time
import cv2
import mediapipe as mp
import sys
import numpy as np

from model import DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
from architecture import MAX_NUM_HANDS, MODEL_COMPLEXITY


def test_image(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image:", image_path)
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    start = time.time()
    results = hands.process(image_rgb)
    end = time.time()
    hands.close()

    latency = (end - start) * 1000  
    count = 0
    if results.multi_hand_landmarks:
        count = sum(len(hand.landmark) for hand in results.multi_hand_landmarks)

    print(f"Detected {count} keypoints in {latency:.2f} ms.")
    if count == 0:
        print("Test failed: no keypoints detected.")
        sys.exit(1)
    if latency > 300:
        print("Test failed: detection latency exceeds 300ms.")
        sys.exit(1)
    print("Test passed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)
    test_image(sys.argv[1])
