"""
audio_engine.py
---------------
Manages playback of acoustic guitar chord WAV samples using pygame.mixer.
"""

import os
import pygame
import time

SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")
FADE_OUT_MS = 800   # ms for last chord to fade before new one


class AudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
        pygame.mixer.init()

        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._current_chord: str | None = None
        self._channels = [pygame.mixer.Channel(i) for i in range(8)]
        self._channel_idx = 0
        self._muted = False

        self._preload_sounds()

    # ── Sound loading ─────────────────────────────────────────────────────────
    def _preload_sounds(self):
        """Load all WAV files from the sounds/ directory."""
        if not os.path.isdir(SOUNDS_DIR):
            print(f"[AudioEngine] ⚠ Sounds directory not found: {SOUNDS_DIR}")
            print("  → Run:  python generate_sounds.py  first.")
            return

        for fname in os.listdir(SOUNDS_DIR):
            if fname.endswith(".wav"):
                chord_name = fname[:-4]   # strip .wav
                path = os.path.join(SOUNDS_DIR, fname)
                try:
                    snd = pygame.mixer.Sound(path)
                    snd.set_volume(0.88)
                    self._sounds[chord_name] = snd
                except Exception as e:
                    print(f"[AudioEngine] Could not load {fname}: {e}")

        print(f"[AudioEngine] Loaded {len(self._sounds)} chord samples.")

    # ── Public API ────────────────────────────────────────────────────────────
    def play(self, chord_name: str | None, velocity: float = 1.0):
        """
        Play a chord immediately when strummed. Uses round-robin channels.
        
        Args:
            chord_name: Name of the chord to play
            velocity: Strum velocity (0.0-1.0) for dynamic volume
        """
        if self._muted or chord_name is None:
            return

        sound = self._sounds.get(chord_name)
        if sound is None:
            return

        # Apply velocity to volume (0.3 to 1.0 range)
        volume = 0.88 * max(0.3, min(1.0, velocity))
        sound.set_volume(volume)

        # Play on the next available channel (allows chords to ring out over each other slightly)
        channel = self._channels[self._channel_idx]
        if channel.get_busy():
            channel.fadeout(150)
        
        channel.play(sound)
        self._current_chord = chord_name
        self._channel_idx = (self._channel_idx + 1) % len(self._channels)
        
        # Debug output
        print(f"  🎸 Strummed {chord_name} (velocity: {velocity:.2f}, volume: {volume:.2f})")

    def stop(self):
        """Fade out all sounds."""
        for channel in self._channels:
            if channel.get_busy():
                channel.fadeout(FADE_OUT_MS)
        self._current_chord = None

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        if self._muted:
            self.stop()
        return self._muted

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def current_chord(self) -> str | None:
        return self._current_chord

    def close(self):
        pygame.mixer.quit()
