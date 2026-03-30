import cv2
import sys
import time
from hand_detector import HandDetector
from chord_engine   import ChordEngine
from audio_engine   import AudioEngine
from strum_engine   import StrumEngine
from guitar_ui      import GuitarUI

# ── Configuration ──────────────────────────────────────────────────────────────
WINDOW_TITLE    = "🎸  Virtual Acoustic Guitar"
CAM_INDEX       = 0          # try 1 or 2 if 0 doesn't open your webcam
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
FLIP_HORIZONTAL = True       # mirror mode (more natural for player)


def main():
    # ── Setup ──────────────────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════╗")
    print("║  🎸  Virtual Acoustic Guitar  🎸             ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Controls:                                   ║")
    print("║    Q / ESC  →  Quit                          ║")
    print("║    M        →  Toggle mute                   ║")
    print("║    R        →  Reset chord detection         ║")
    print("╚══════════════════════════════════════════════╝\n")

    # Initialise modules
    detector = HandDetector(max_hands=2, detection_confidence=0.75,
                            tracking_confidence=0.75)
    chord_engine = ChordEngine(hold_frames=6)   # slightly faster detection
    strum_engine = StrumEngine(velocity_threshold=0.06, cooldown_ms=180)
    audio        = AudioEngine()
    ui           = GuitarUI(strum_anim_frames=18)

    # Open webcam
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {CAM_INDEX}.")
        print("  → Try changing CAM_INDEX in main.py")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, FRAME_WIDTH, FRAME_HEIGHT)

    chord_info = {
        "chord": None, "full_name": "—", "type": "—",
        "color": (150, 150, 150), "diagram": [], "confidence": 0.0,
        "finger_states": {},
    }
    
    current_chord = None  # Track the current chord being held

    print("✅ Webcam opened.  Hold your hands in front of the camera to play!")
    print("   🤚 LEFT HAND (on screen) = Chord shapes")
    print("   ✋ RIGHT HAND (on screen) = Strum up/down to play")
    print("   💡 TIP: Hold a chord shape, then strum with your other hand!\n")

    # ── Main loop ───────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame — retrying...")
            time.sleep(0.01)
            continue

        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        # Hand detection
        hands_data, _ = detector.process(frame)

        chord_hand = None
        strum_hand = None

        # Improved hand assignment logic
        if len(hands_data) == 1:
            # Single hand - check if it's making a chord shape or strumming
            hand = hands_data[0]
            # If hand is on the left side of screen, assume chord hand
            wrist_x = hand["lm"][0][0]
            if wrist_x < 0.5:  # Left side of screen
                chord_hand = hand
            else:  # Right side - could be either, default to chord
                chord_hand = hand
                
        elif len(hands_data) >= 2:
            # Two hands detected - assign based on screen position
            # Sort by X position (left to right)
            hands_data.sort(key=lambda h: h["lm"][0][0])
            chord_hand = hands_data[0]  # Left hand = chord
            strum_hand = hands_data[1]  # Right hand = strum

        # 1. Chord detection (Left hand)
        if chord_hand:
            chord_info = chord_engine.detect(chord_hand)
            # Update current chord if confidence is good
            if chord_info["chord"] and chord_info["confidence"] > 0.5:
                current_chord = chord_info["chord"]
        else:
            chord_info = {
                "chord": current_chord,  # Keep showing last detected chord
                "full_name": "Hold chord shape with LEFT hand",
                "type": "—", "color": (80, 60, 40), "diagram": [],
                "confidence": 0.0, "finger_states": {},
            }

        # 2. Strum detection (Right hand)
        trigger_strum = False
        strum_velocity = 0.0
        
        if strum_hand:
            # Track the index finger tip's Y position for strumming
            index_y = strum_hand["lm"][8][1]
            trigger_strum, strum_velocity = strum_engine.process(index_y)

            # Play audio on strum ONLY if we have a chord
            if trigger_strum and current_chord:
                # Pass velocity for dynamic volume
                audio.play(current_chord, velocity=strum_velocity)
        else:
            # Reset strum tracker if hand disappears
            strum_engine.reset()

        # Update chord info display with current chord
        if current_chord and not chord_info["chord"]:
            chord_info["chord"] = current_chord
            chord_info["full_name"] = f"{current_chord} (held)"

        # Draw UI
        frame = ui.draw(frame, chord_hand, strum_hand, chord_info, trigger_strum, 
                       muted=audio.muted, strum_velocity=strum_velocity)

        cv2.imshow(WINDOW_TITLE, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):          # Q or ESC
            break
        elif key in (ord('m'), ord('M')):            # M — mute
            muted = audio.toggle_mute()
            print(f"  🔇 Muted" if muted else "  🔊 Unmuted")
        elif key in (ord('r'), ord('R')):            # R — reset
            chord_engine.reset()
            audio.stop()
            print("  ↩ Chord engine reset")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    print("\nClosing...")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    audio.close()
    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
