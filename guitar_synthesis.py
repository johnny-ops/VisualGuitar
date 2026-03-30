"""
guitar_synthesis.py
-------------------
Reusable acoustic guitar synthesis utilities.
Import this module to synthesize custom guitar chords programmatically.

Example usage:
    from guitar_synthesis import synthesize_chord, synthesize_note
    
    # Synthesize a custom chord
    chord_notes = [(1, 0), (2, 2), (3, 2), (4, 1), (5, 0)]  # Am chord
    audio = synthesize_chord(chord_notes, duration=3.0)
    
    # Synthesize a single note
    audio = synthesize_note(string_idx=2, fret=3, duration=2.0)
"""

import numpy as np
from generate_sounds import (
    synthesize_acoustic_guitar_chord,
    note_frequency,
    karplus_strong_enhanced,
    apply_body_resonance,
    apply_room_reverb,
    save_wav,
    SAMPLE_RATE,
    STRING_FREQS
)


def synthesize_chord(chord_notes: list, duration: float = 3.0, 
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesize a guitar chord with realistic acoustic tone.
    
    Args:
        chord_notes: List of (string_idx, fret) tuples where:
                    - string_idx: 0-5 (0=low E, 5=high E)
                    - fret: 0-24 or None for muted string
        duration: Duration in seconds
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        Audio signal as numpy array (normalized to [-1, 1])
    
    Example:
        # C major chord
        c_major = [(1, 3), (2, 2), (3, 0), (4, 1), (5, 0)]
        audio = synthesize_chord(c_major, duration=4.0)
    """
    return synthesize_acoustic_guitar_chord(chord_notes, sample_rate, duration)


def synthesize_note(string_idx: int, fret: int, duration: float = 2.0,
                   sample_rate: int = SAMPLE_RATE, 
                   add_effects: bool = True) -> np.ndarray:
    """
    Synthesize a single guitar note.
    
    Args:
        string_idx: String index 0-5 (0=low E, 5=high E)
        fret: Fret number 0-24
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        add_effects: Whether to add body resonance and reverb
    
    Returns:
        Audio signal as numpy array
    
    Example:
        # Play the 5th fret on the A string
        audio = synthesize_note(string_idx=1, fret=5, duration=3.0)
    """
    freq = note_frequency(string_idx, fret)
    wave = karplus_strong_enhanced(freq, duration, sample_rate, string_idx)
    
    if add_effects:
        wave = apply_body_resonance(wave, sample_rate)
        wave = apply_room_reverb(wave, sample_rate)
    
    # Normalize
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= peak
    
    return wave


def save_chord_to_file(chord_notes: list, filename: str, 
                       duration: float = 3.0, sample_rate: int = SAMPLE_RATE):
    """
    Synthesize and save a chord directly to a WAV file.
    
    Args:
        chord_notes: List of (string_idx, fret) tuples
        filename: Output filename (should end with .wav)
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    
    Example:
        # Create and save a D minor chord
        dm = [(2, 0), (3, 2), (4, 3), (5, 1)]
        save_chord_to_file(dm, "my_dm_chord.wav", duration=4.0)
    """
    audio = synthesize_chord(chord_notes, duration, sample_rate)
    save_wav(filename, audio, sample_rate)
    print(f"✓ Saved chord to {filename}")


def get_string_frequency(string_idx: int, fret: int = 0) -> float:
    """
    Get the frequency of a specific string and fret.
    
    Args:
        string_idx: String index 0-5 (0=low E, 5=high E)
        fret: Fret number (default 0 for open string)
    
    Returns:
        Frequency in Hz
    
    Example:
        freq = get_string_frequency(string_idx=2, fret=3)  # D string, 3rd fret
        print(f"Frequency: {freq:.2f} Hz")
    """
    return note_frequency(string_idx, fret)


# Common chord shapes for easy reference
COMMON_CHORDS = {
    "Am": [(1, 0), (2, 2), (3, 2), (4, 1), (5, 0)],
    "C":  [(1, 3), (2, 2), (3, 0), (4, 1), (5, 0)],
    "D":  [(2, 0), (3, 2), (4, 3), (5, 2)],
    "Em": [(0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)],
    "G":  [(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 3)],
    "F":  [(0, 1), (1, 1), (2, 2), (3, 3), (4, 3), (5, 1)],
    "E":  [(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)],
    "A":  [(1, 0), (2, 2), (3, 2), (4, 2), (5, 0)],
}


def synthesize_chord_by_name(chord_name: str, duration: float = 3.0) -> np.ndarray:
    """
    Synthesize a common chord by name.
    
    Args:
        chord_name: Name of chord (e.g., "Am", "C", "G")
        duration: Duration in seconds
    
    Returns:
        Audio signal as numpy array
    
    Example:
        audio = synthesize_chord_by_name("G", duration=4.0)
    """
    if chord_name not in COMMON_CHORDS:
        raise ValueError(f"Unknown chord: {chord_name}. Available: {list(COMMON_CHORDS.keys())}")
    
    return synthesize_chord(COMMON_CHORDS[chord_name], duration)


if __name__ == "__main__":
    # Demo: synthesize a few chords
    print("🎸 Guitar Synthesis Demo\n")
    
    print("Synthesizing G major chord...")
    g_audio = synthesize_chord_by_name("G", duration=2.0)
    save_wav("demo_g.wav", g_audio, SAMPLE_RATE)
    print("✓ Saved to demo_g.wav\n")
    
    print("Synthesizing single note (A string, 5th fret)...")
    note_audio = synthesize_note(string_idx=1, fret=5, duration=2.0)
    save_wav("demo_note.wav", note_audio, SAMPLE_RATE)
    print("✓ Saved to demo_note.wav\n")
    
    print("Done! You can now import this module to synthesize custom chords.")
