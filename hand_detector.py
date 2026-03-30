"""
hand_detector.py
----------------
Wraps MediaPipe Hands to expose simple landmark access.
"""

import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices
WRIST       = 0
THUMB_CMC   = 1;  THUMB_MCP   = 2;  THUMB_IP    = 3;  THUMB_TIP   = 4
INDEX_MCP   = 5;  INDEX_PIP   = 6;  INDEX_DIP   = 7;  INDEX_TIP   = 8
MIDDLE_MCP  = 9;  MIDDLE_PIP  = 10; MIDDLE_DIP  = 11; MIDDLE_TIP  = 12
RING_MCP    = 13; RING_PIP    = 14; RING_DIP    = 15; RING_TIP    = 16
PINKY_MCP   = 17; PINKY_PIP   = 18; PINKY_DIP   = 19; PINKY_TIP   = 20

FINGER_TIPS  = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS  = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGER_PIPS  = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


class HandDetector:
    def __init__(self, max_hands: int = 2,
                 detection_confidence: float = 0.75,
                 tracking_confidence: float = 0.75):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray):
        """
        Process a BGR frame and return (landmarks_list, annotated_frame).

        landmarks_list: list of dicts per detected hand.
            Each dict:  {
                'lm':     list of (x, y, z) normalised [0-1],
                'lm_px':  list of (x_px, y_px) in pixel coords,
                'handed': 'Left' | 'Right'
            }
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        hands_data = []
        if results.multi_hand_landmarks:
            for hand_lm, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                lm_norm  = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]
                lm_px    = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm.landmark]
                handed   = hand_info.classification[0].label  # 'Left'/'Right'
                hands_data.append({
                    "lm":       lm_norm,
                    "lm_px":   lm_px,
                    "handed":  handed,
                    "raw":     hand_lm,          # for drawing
                })

        return hands_data, results

    def close(self):
        self.hands.close()
