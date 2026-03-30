"""
guitar_ui.py
------------
Renders the interactive virtual guitar overlay on the webcam frame using OpenCV.

Features:
  • 6 glowing guitar strings with strum animation
  • Fret lines with position markers
  • Hand landmark skeleton
  • Large chord name + type badge (top-left)
  • Chord finger diagram (bottom-right)
  • Confidence bar
  • FPS counter
  • Mute indicator
"""

import cv2
import numpy as np
import math
import time

# ── Warm acoustic gold colour palette ─────────────────────────────────────────
BG_OVERLAY      = (20,  14,  8)         # dark warm background tint
FRET_COLOR      = (60,  50,  30)        # dark fret lines
STRING_COLOR    = (200, 165, 100)       # neutral string colour (warm gold)
STRING_GLOW     = (255, 220, 130)       # bright string glow on strum
MARKER_COLOR    = (160, 130, 70)        # fret dot markers
HUD_BG          = (15,  10,  5, 180)   # semi-transparent HUD panel
TEXT_PRIMARY    = (255, 240, 200)       # main text
TEXT_SECONDARY  = (180, 155, 100)       # secondary text
SKELETON_COLOR  = (80,  200, 255)       # hand skeleton colour
LANDMARK_COLOR  = (255, 200, 80)        # landmark dot colour
UI_ACCENT       = (255, 191, 71)        # accent / highlight

FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD     = cv2.FONT_HERSHEY_DUPLEX
NUM_STRINGS   = 6
NUM_FRETS     = 5


class GuitarUI:
    def __init__(self, strum_anim_frames: int = 15):
        self._strum_frames   = strum_anim_frames
        self._strum_counter  = 0          # counts down per frame
        self._last_velocity  = 0.0        # last strum velocity
        self._fps_ring       = []
        self._last_ts        = time.time()

    # ── Public entry point ────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, chord_hand: dict | None, strum_hand: dict | None,
             chord_info: dict, trigger_strum: bool, muted: bool = False, 
             strum_velocity: float = 0.0) -> np.ndarray:
        """
        Composite all guitar UI elements onto *frame* (in place) and return it.
        """
        h, w = frame.shape[:2]

        # Measure FPS
        now = time.time()
        self._fps_ring.append(now - self._last_ts)
        self._last_ts = now
        if len(self._fps_ring) > 30:
            self._fps_ring.pop(0)
        fps = 1.0 / (sum(self._fps_ring) / len(self._fps_ring)) if self._fps_ring else 0

        # Trigger strum animation when a physical strum is detected
        if trigger_strum:
            self._strum_counter = self._strum_frames
            self._last_velocity = strum_velocity

        # Draw layers
        frame = self._draw_fretboard(frame, w, h)
        frame = self._draw_strings(frame, w, h)
        frame = self._draw_hand_skeleton(frame, chord_hand, is_strum=False)
        frame = self._draw_hand_skeleton(frame, strum_hand, is_strum=True)
        frame = self._draw_chord_hud(frame, w, h, chord_info)
        frame = self._draw_chord_diagram(frame, w, h, chord_info)
        frame = self._draw_confidence_bar(frame, w, h, chord_info)
        frame = self._draw_strum_indicator(frame, w, h, strum_hand, strum_velocity)
        frame = self._draw_top_right_badges(frame, w, fps, muted)

        if self._strum_counter > 0:
            self._strum_counter -= 1

        return frame

    # ── Fretboard ─────────────────────────────────────────────────────────────
    def _fret_positions(self, w: int) -> list:
        """Return x-coordinates of fret lines (evenly spaced in lower 60% of width)."""
        x_start = int(w * 0.15)
        x_end   = int(w * 0.85)
        return [int(x_start + i * (x_end - x_start) / NUM_FRETS)
                for i in range(NUM_FRETS + 1)]

    def _string_positions(self, h: int) -> list:
        """Return y-coordinates for each string."""
        y_start = int(h * 0.28)
        y_end   = int(h * 0.78)
        return [int(y_start + i * (y_end - y_start) / (NUM_STRINGS - 1))
                for i in range(NUM_STRINGS)]

    def _draw_fretboard(self, frame: np.ndarray, w: int, h: int) -> np.ndarray:
        """Draw semi-transparent fretboard background and fret lines."""
        frets    = self._fret_positions(w)
        strings  = self._string_positions(h)

        # Dark fretboard panel
        overlay = frame.copy()
        pad = 18
        cv2.rectangle(overlay,
                      (frets[0] - pad, strings[0] - pad),
                      (frets[-1] + pad, strings[-1] + pad),
                      (28, 18, 8), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Fret lines
        for x in frets:
            thickness = 2 if x != frets[0] else 4
            cv2.line(frame, (x, strings[0] - 10), (x, strings[-1] + 10),
                     FRET_COLOR, thickness, cv2.LINE_AA)

        # Fret position dots (standard: 3, 5 → frets 1,3 in our 5-fret view)
        dot_frets = [1, 3]
        mid_y = (strings[0] + strings[-1]) // 2
        for fi in dot_frets:
            if fi < len(frets) - 1:
                dot_x = (frets[fi] + frets[fi + 1]) // 2
                cv2.circle(frame, (dot_x, mid_y), 6, MARKER_COLOR, -1, cv2.LINE_AA)

        return frame

    def _draw_strings(self, frame: np.ndarray, w: int, h: int) -> np.ndarray:
        """Draw 6 guitar strings with thickness gradient and strum glow."""
        frets   = self._fret_positions(w)
        strings = self._string_positions(h)
        x0, x1  = frets[0] - 10, frets[-1] + 10

        strum_progress = self._strum_counter / max(self._strum_frames, 1)

        for i, y in enumerate(strings):
            # Thicker strings at top (lower pitched)
            thickness = max(1, 4 - i // 2)

            if strum_progress > 0:
                # Animated glow: wave-like Y offset per string
                phase   = (i / NUM_STRINGS) * math.pi
                jitter  = math.sin(strum_progress * math.pi * 3 + phase) * 6 * strum_progress
                y_draw  = int(y + jitter)
                color   = _lerp_color(STRING_COLOR, STRING_GLOW, strum_progress)
                # Draw glowing halo
                cv2.line(frame, (x0, y_draw), (x1, y_draw),
                         _lerp_color((0, 0, 0), color, 0.4), thickness + 4, cv2.LINE_AA)
            else:
                y_draw = y
                color  = STRING_COLOR

            cv2.line(frame, (x0, y_draw), (x1, y_draw), color, thickness, cv2.LINE_AA)

        return frame

    # ── Hand Skeleton ─────────────────────────────────────────────────────────
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),                      # thumb
        (0,5),(5,6),(6,7),(7,8),                      # index
        (0,9),(9,10),(10,11),(11,12),                 # middle
        (0,13),(13,14),(14,15),(15,16),               # ring
        (0,17),(17,18),(18,19),(19,20),               # pinky
        (5,9),(9,13),(13,17),                         # palm
    ]

    def _draw_hand_skeleton(self, frame: np.ndarray,
                             hand: dict | None, is_strum: bool) -> np.ndarray:
        if not hand:
            return frame
            
        lm_px = hand["lm_px"]
        skeleton_colour = (200, 100, 255) if is_strum else SKELETON_COLOR
        dot_colour = (255, 150, 255) if is_strum else LANDMARK_COLOR

        # Connections
        for a, b in self.HAND_CONNECTIONS:
            cv2.line(frame, lm_px[a], lm_px[b],
                     skeleton_colour, 2, cv2.LINE_AA)
        
        # Landmark dots
        for idx, (x, y) in enumerate(lm_px):
            is_tip = idx in (4, 8, 12, 16, 20)
            r = 6 if is_tip else 4
            cv2.circle(frame, (x, y), r, dot_colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), r + 1, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw a pick on the strum hand's index finger tip (landmark 8)
        if is_strum:
            ix, iy = lm_px[8]
            cv2.circle(frame, (ix, iy), 12, (50, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 12, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "PICK", (ix - 12, iy - 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
        return frame

    # ── Chord HUD (top-left panel) ────────────────────────────────────────────
    def _draw_chord_hud(self, frame: np.ndarray, w: int, h: int,
                         chord_info: dict) -> np.ndarray:
        chord_name = chord_info.get("chord") or "—"
        full_name  = chord_info.get("full_name", "—")
        chord_type = chord_info.get("type", "")
        color      = chord_info.get("color", (150, 150, 150))
        confidence = chord_info.get("confidence", 0.0)

        # Panel
        panel_w, panel_h = 240, 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                      (12, 8, 4), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Border glow using chord colour
        border_color = _scale_color(color, 0.7) if confidence > 0.3 else (60, 50, 30)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h),
                      border_color, 2, cv2.LINE_AA)

        # Show instruction if no chord detected
        if chord_name == "—" or confidence < 0.1:
            cv2.putText(frame, "CHORD HAND", (20, 45),
                       FONT_BOLD, 0.8, (80, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Hold LEFT hand in", (20, 70),
                       FONT, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(frame, "chord shape position", (20, 90),
                       FONT, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
        else:
            # Big chord name
            chord_display = chord_name if chord_name != "—" else "?"
            scale = 2.8 if len(chord_display) <= 2 else 2.0
            cv2.putText(frame, chord_display, (20, 75),
                        FONT_BOLD, scale, _scale_color(color, 1.0), 3, cv2.LINE_AA)

            # Subtitle line
            subtitle = f"{full_name}"
            if chord_type and chord_type not in ("—",):
                subtitle += f"  [{chord_type}]"
            cv2.putText(frame, subtitle, (20, 98),
                        FONT, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)

        # Acoustic badge beneath panel
        badge_y = 10 + panel_h + 6
        cv2.rectangle(frame, (10, badge_y), (130, badge_y + 22),
                      (40, 25, 10), -1)
        cv2.rectangle(frame, (10, badge_y), (130, badge_y + 22),
                      UI_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Acoustic Guitar", (14, badge_y + 15),
                    FONT, 0.40, UI_ACCENT, 1, cv2.LINE_AA)

        return frame

    # ── Chord finger diagram (bottom-right) ───────────────────────────────────
    def _draw_chord_diagram(self, frame: np.ndarray, w: int, h: int,
                             chord_info: dict) -> np.ndarray:
        diagram  = chord_info.get("diagram", [])
        color    = chord_info.get("color", (150, 150, 150))
        chord_name = chord_info.get("chord")
        if not diagram or not chord_name:
            return frame

        # Diagram parameters
        box_w, box_h = 170, 160
        x0 = w - box_w - 10
        y0 = h - box_h - 10
        cell_w = (box_w - 30) // 5   # 5 frets shown
        cell_h = (box_h - 30) // 5   # 6 strings - 1 gaps

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                      (12, 8, 4), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h),
                      _scale_color(color, 0.7), 2, cv2.LINE_AA)

        # Title
        cv2.putText(frame, chord_name, (x0 + 6, y0 + 18),
                    FONT_BOLD, 0.65, _scale_color(color, 1.0), 2, cv2.LINE_AA)

        # Grid origin
        gx = x0 + 28
        gy = y0 + 28

        # Draw fret lines (vertical)
        for f in range(6):
            lx = gx + f * cell_w
            cv2.line(frame, (lx, gy), (lx, gy + 5 * cell_h),
                     FRET_COLOR, 1, cv2.LINE_AA)

        # Draw string lines (horizontal)
        for s in range(6):
            ly = gy + s * cell_h
            cv2.line(frame, (gx, ly), (gx + 5 * cell_w, ly),
                     STRING_COLOR, 1, cv2.LINE_AA)

        # Draw finger dots + open/muted markers
        for si, fret in enumerate(diagram):
            sx = gx - 12           # to the left of grid
            sy = gy + si * cell_h
            if fret is None:
                # Muted: draw X
                cv2.putText(frame, "X", (sx - 2, sy + 5),
                            FONT, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
            elif fret == 0:
                # Open: draw O
                cv2.circle(frame, (sx + 6, sy), 5, TEXT_SECONDARY, 1, cv2.LINE_AA)
            else:
                # Fretted: filled dot
                dot_x = gx + (fret - 1) * cell_w + cell_w // 2
                dot_y = sy
                cv2.circle(frame, (dot_x, dot_y), 7,
                           _scale_color(color, 0.9), -1, cv2.LINE_AA)
                cv2.circle(frame, (dot_x, dot_y), 7, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    # ── Confidence bar (bottom-left) ──────────────────────────────────────────
    def _draw_confidence_bar(self, frame: np.ndarray, w: int, h: int,
                              chord_info: dict) -> np.ndarray:
        conf  = chord_info.get("confidence", 0.0)
        color = chord_info.get("color", (150, 150, 150))

        bar_x, bar_y = 10, h - 30
        bar_w, bar_h = 160, 10

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (30, 20, 10), -1)
        fill_w = int(bar_w * conf)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h),
                          _scale_color(color, 0.9), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      UI_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Conf {int(conf * 100)}%", (bar_x, bar_y - 4),
                    FONT, 0.38, TEXT_SECONDARY, 1, cv2.LINE_AA)
        return frame

    # ── Strum indicator (right side) ──────────────────────────────────────────
    def _draw_strum_indicator(self, frame: np.ndarray, w: int, h: int,
                              strum_hand: dict | None, velocity: float) -> np.ndarray:
        """Draw strum velocity indicator and instructions."""
        if not strum_hand:
            # Show instruction when no strum hand detected
            msg_x = w - 280
            msg_y = h // 2
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (msg_x - 10, msg_y - 40), 
                         (msg_x + 260, msg_y + 40), (12, 8, 4), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.rectangle(frame, (msg_x - 10, msg_y - 40), 
                         (msg_x + 260, msg_y + 40), (200, 100, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame, "STRUM HAND", (msg_x, msg_y - 10),
                       FONT_BOLD, 0.7, (200, 100, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Move RIGHT hand up/down", (msg_x, msg_y + 15),
                       FONT, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(frame, "over strings to strum!", (msg_x, msg_y + 35),
                       FONT, 0.45, TEXT_SECONDARY, 1, cv2.LINE_AA)
            return frame
        
        # Show velocity meter when strumming
        if velocity > 0.1:
            meter_x = w - 50
            meter_y = h // 2 - 80
            meter_h = 160
            meter_w = 20
            
            # Background
            cv2.rectangle(frame, (meter_x, meter_y), 
                         (meter_x + meter_w, meter_y + meter_h),
                         (30, 20, 10), -1)
            
            # Fill based on velocity
            fill_h = int(meter_h * velocity)
            fill_y = meter_y + meter_h - fill_h
            
            # Color gradient based on velocity
            if velocity < 0.5:
                color = (100, 200, 100)  # Green - soft
            elif velocity < 0.8:
                color = (100, 200, 255)  # Yellow - medium
            else:
                color = (100, 100, 255)  # Red - hard
            
            cv2.rectangle(frame, (meter_x, fill_y),
                         (meter_x + meter_w, meter_y + meter_h),
                         color, -1)
            
            # Border
            cv2.rectangle(frame, (meter_x, meter_y),
                         (meter_x + meter_w, meter_y + meter_h),
                         UI_ACCENT, 2, cv2.LINE_AA)
            
            # Label
            cv2.putText(frame, "STRUM", (meter_x - 45, meter_y - 5),
                       FONT, 0.4, TEXT_SECONDARY, 1, cv2.LINE_AA)
            
        return frame

    # ── Top-right: FPS + mute ─────────────────────────────────────────────────
    def _draw_top_right_badges(self, frame: np.ndarray, w: int,
                                fps: float, muted: bool) -> np.ndarray:
        # FPS
        fps_text = f"FPS {fps:.0f}"
        cv2.putText(frame, fps_text, (w - 90, 28),
                    FONT, 0.50, TEXT_SECONDARY, 1, cv2.LINE_AA)

        # Mute indicator
        if muted:
            cv2.putText(frame, "[ MUTED ]", (w - 100, 52),
                        FONT, 0.50, (60, 60, 200), 1, cv2.LINE_AA)

        # Hint bar at very bottom
        hints = "Q: quit   |   M: mute"
        cv2.putText(frame, hints,
                    (int(w / 2) - 100, frame.shape[0] - 8),
                    FONT, 0.38, (80, 60, 40), 1, cv2.LINE_AA)

        return frame


# ── Utility colour helpers ─────────────────────────────────────────────────────
def _lerp_color(c1, c2, t: float) -> tuple:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _scale_color(color: tuple, scale: float) -> tuple:
    return tuple(min(255, int(c * scale)) for c in color)
