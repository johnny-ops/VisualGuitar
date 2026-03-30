"""
chord_engine.py
---------------
Maps MediaPipe hand landmarks to guitar chord names using
finger curl / extension pattern recognition.
"""

import numpy as np
from typing import Optional


# ── Finger indices from hand_detector.py ──────────────────────────────────────
WRIST       = 0
THUMB_CMC=1; THUMB_MCP=2; THUMB_IP=3;  THUMB_TIP=4
INDEX_MCP=5; INDEX_PIP=6; INDEX_DIP=7; INDEX_TIP=8
MIDDLE_MCP=9; MIDDLE_PIP=10; MIDDLE_DIP=11; MIDDLE_TIP=12
RING_MCP=13; RING_PIP=14; RING_DIP=15; RING_TIP=16
PINKY_MCP=17; PINKY_PIP=18; PINKY_DIP=19; PINKY_TIP=20

# (tip, pip, mcp) triples for each finger
FINGERS = [
    (INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    (RING_TIP,   RING_PIP,   RING_MCP),
    (PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
]


def _curl(lm, tip, pip, mcp) -> float:
    """
    Curl value: 0.0 = fully extended, 1.0 = fully curled.
    Uses the vertical (Y) position delta; larger Y = lower in frame.
    """
    tip_y   = lm[tip][1]
    pip_y   = lm[pip][1]
    mcp_y   = lm[mcp][1]
    # In normalised coords Y increases downward
    # Extended: tip_y < mcp_y  →  negative diff
    # Curled:   tip_y > pip_y  →  positive diff
    raw = (tip_y - pip_y) / (abs(mcp_y - pip_y) + 1e-5)
    return float(np.clip((raw + 1) / 2, 0, 1))   # map to [0, 1]


def _thumb_extended(lm) -> bool:
    """Rough check: thumb tip is to the side of thumb MCP."""
    tip_x = lm[THUMB_TIP][0]
    mcp_x = lm[THUMB_MCP][0]
    return abs(tip_x - mcp_x) > 0.06


def get_finger_states(lm) -> dict:
    """Return dict of finger curl values (0=open, 1=closed) + thumb bool."""
    idx, mid, rng, pnk = [_curl(lm, *f) for f in FINGERS]
    thumb = _thumb_extended(lm)
    return {
        "index":  idx,
        "middle": mid,
        "ring":   rng,
        "pinky":  pnk,
        "thumb":  thumb,
    }


# ── Chord detection rules ─────────────────────────────────────────────────────
#  Each rule: (chord_name, check_function)
#  check_function receives the finger-state dict and returns bool.

CHORD_RULES = [
    # Em — all 4 fingers fairly extended (open position)
    ("Em",  lambda s: s["index"] < 0.4 and s["middle"] < 0.4
                      and s["ring"]  < 0.45 and s["pinky"] < 0.45),

    # G  — index closed, middle open, ring & pinky closed
    ("G",   lambda s: s["index"] > 0.55 and s["middle"] < 0.4
                      and s["ring"] > 0.55 and s["pinky"] > 0.55),

    # C  — index semi, middle bent, ring curled, pinky open
    ("C",   lambda s: 0.3 < s["index"] < 0.65
                      and 0.3 < s["middle"] < 0.65
                      and s["ring"]  > 0.55 and s["pinky"] < 0.4),

    # Am — index slightly curled, middle/ring bent, pinky open-ish
    ("Am",  lambda s: 0.25 < s["index"] < 0.6
                      and s["middle"] > 0.45
                      and s["ring"]   > 0.45 and s["pinky"] < 0.5),

    # D  — index bent, middle bent, pinky & ring open
    ("D",   lambda s: s["index"] > 0.5 and s["middle"] > 0.5
                      and s["ring"]  < 0.45 and s["pinky"] < 0.35),

    # F  — all four fingers moderately curled (barre shape)
    ("F",   lambda s: s["index"] > 0.45 and s["middle"] > 0.45
                      and s["ring"] > 0.45  and s["pinky"] > 0.45),

    # E  — index, middle closed; ring, pinky open
    ("E",   lambda s: s["index"] > 0.55 and s["middle"] > 0.55
                      and s["ring"]  < 0.4  and s["pinky"] < 0.4),

    # A  — index open, middle/ring/pinky all curled
    ("A",   lambda s: s["index"] < 0.4
                      and s["middle"] > 0.55
                      and s["ring"]   > 0.55 and s["pinky"] > 0.55),

    # B7 — index bent, middle/ring semi, pinky extended
    ("B7",  lambda s: s["index"] > 0.5
                      and 0.3 < s["middle"] < 0.65
                      and 0.3 < s["ring"]   < 0.65
                      and s["pinky"] < 0.4),

    # Dm — index slightly, middle closed, ring open, pinky closed
    ("Dm",  lambda s: 0.2 < s["index"] < 0.55
                      and s["middle"] > 0.5
                      and s["ring"]   < 0.45
                      and s["pinky"]  > 0.5),

    # E7 — index/pinky open, middle/ring closed
    ("E7",  lambda s: s["index"] < 0.4
                      and s["middle"] > 0.55
                      and s["ring"]   > 0.55
                      and s["pinky"]  < 0.4),

    # A7 — index/ring open; middle/pinky closed
    ("A7",  lambda s: s["index"] < 0.4
                      and s["middle"] > 0.55
                      and s["ring"]   < 0.4
                      and s["pinky"]  > 0.55),
]


# ── Chord metadata ─────────────────────────────────────────────────────────────
CHORD_INFO = {
    "Am": {"full_name": "A minor",      "type": "minor", "color": (180, 100, 220)},
    "C":  {"full_name": "C major",      "type": "major", "color": (80,  200, 120)},
    "D":  {"full_name": "D major",      "type": "major", "color": (80,  170, 255)},
    "Em": {"full_name": "E minor",      "type": "minor", "color": (200, 120, 100)},
    "G":  {"full_name": "G major",      "type": "major", "color": (255, 200, 60)},
    "F":  {"full_name": "F major",      "type": "major", "color": (60,  220, 220)},
    "E":  {"full_name": "E major",      "type": "major", "color": (255, 130, 60)},
    "A":  {"full_name": "A major",      "type": "major", "color": (255, 80,  120)},
    "B7": {"full_name": "B dominant 7", "type": "dom7",  "color": (150, 255, 180)},
    "Dm": {"full_name": "D minor",      "type": "minor", "color": (180, 80,  255)},
    "E7": {"full_name": "E dominant 7", "type": "dom7",  "color": (255, 160, 60)},
    "A7": {"full_name": "A dominant 7", "type": "dom7",  "color": (60,  200, 255)},
}

# Chord finger diagrams: (string, fret) — 6 strings, fret 0=open, None=muted
CHORD_DIAGRAMS = {
    "Am": [None, 0, 2, 2, 1, 0],
    "C":  [None, 3, 2, 0, 1, 0],
    "D":  [None, None, 0, 2, 3, 2],
    "Em": [0, 2, 2, 0, 0, 0],
    "G":  [3, 2, 0, 0, 0, 3],
    "F":  [1, 1, 2, 3, 3, 1],
    "E":  [0, 2, 2, 1, 0, 0],
    "A":  [None, 0, 2, 2, 2, 0],
    "B7": [None, 2, 1, 2, 0, 2],
    "Dm": [None, None, 0, 2, 3, 1],
    "E7": [0, 2, 0, 1, 0, 0],
    "A7": [None, 0, 2, 0, 2, 0],
}


class ChordEngine:
    def __init__(self, hold_frames: int = 8):
        """
        hold_frames: number of consecutive frames a chord must match
                     before being considered 'detected' (reduces jitter).
        """
        self.hold_frames = hold_frames
        self._candidate   = None
        self._candidate_count = 0
        self._current_chord   = None
        self._current_conf    = 0.0

    def detect(self, hand_data: dict) -> dict:
        """
        hand_data: single hand dict from HandDetector (with 'lm' key).
        Returns: {chord, full_name, type, color, diagram, confidence, finger_states}
        """
        lm = hand_data["lm"]
        states = get_finger_states(lm)

        matched_chord = None
        for chord_name, rule_fn in CHORD_RULES:
            if rule_fn(states):
                matched_chord = chord_name
                break

        # Temporal smoothing
        if matched_chord == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate       = matched_chord
            self._candidate_count = 1

        if self._candidate_count >= self.hold_frames:
            self._current_chord = self._candidate
            self._current_conf  = min(1.0, self._candidate_count / (self.hold_frames * 2))

        chord = self._current_chord
        if chord and chord in CHORD_INFO:
            info = CHORD_INFO[chord]
            return {
                "chord":         chord,
                "full_name":     info["full_name"],
                "type":          info["type"],
                "color":         info["color"],
                "diagram":       CHORD_DIAGRAMS.get(chord, []),
                "confidence":    self._current_conf,
                "finger_states": states,
            }
        return {
            "chord":         None,
            "full_name":     "—",
            "type":          "—",
            "color":         (150, 150, 150),
            "diagram":       [],
            "confidence":    0.0,
            "finger_states": states,
        }

    def reset(self):
        self._candidate = None
        self._candidate_count = 0
        self._current_chord = None
