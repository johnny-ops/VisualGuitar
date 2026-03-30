"""
generate_sounds.py
------------------
Generates highly realistic acoustic guitar chord WAV files using advanced
Karplus-Strong synthesis with body resonance, harmonics, and attack modeling.

Run once before starting the main app:
    python generate_sounds.py
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy import signal

SAMPLE_RATE = 44100
DURATION_SEC = 4.5  # seconds per chord ring
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sounds")

# Guitar standard tuning frequencies (E2 A2 D3 G3 B3 E4)
STRING_FREQS = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]

# Chord definitions: list of (string_index, fret) or None for muted string
# Fret 0 = open, None = muted
CHORDS = {
    "Am": [(0, None), (1, 0), (2, 2), (3, 2), (4, 1), (5, 0)],
    "C":  [(0, None), (1, 3), (2, 2), (3, 0), (4, 1), (5, 0)],
    "D":  [(0, None), (1, None),(2, 0), (3, 2), (4, 3), (5, 2)],
    "Em": [(0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)],
    "G":  [(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 3)],
    "F":  [(0, 1), (1, 1), (2, 2), (3, 3), (4, 3), (5, 1)],
    "E":  [(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)],
    "A":  [(0, None),(1, 0), (2, 2), (3, 2), (4, 2), (5, 0)],
    "B7": [(0, None),(1, 2), (2, 1), (3, 2), (4, 0), (5, 2)],
    "Dm": [(0, None),(1, None),(2, 0),(3, 2),(4, 3),(5, 1)],
    "E7": [(0, 0), (1, 2), (2, 0), (3, 1), (4, 0), (5, 0)],
    "A7": [(0, None),(1, 0), (2, 2), (3, 0), (4, 2), (5, 0)],
}

FRET_SEMITONE = 2 ** (1 / 12)  # one semitone per fret


def generate_pick_attack(sample_rate: int, duration_ms: float = 8.0) -> np.ndarray:
    """Generate realistic pick attack transient for plucked string."""
    from scipy import signal as sp_signal
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)
    
    # Sharp attack with quick decay
    envelope = np.exp(-t * 400) * (1 + 3 * np.exp(-t * 1200))
    # High-frequency content for pick noise
    noise = np.random.uniform(-1, 1, n_samples)
    attack = noise * envelope
    
    # High-pass filter to emphasize pick brightness
    sos = sp_signal.butter(4, 2000, 'hp', fs=sample_rate, output='sos')
    attack = sp_signal.sosfilt(sos, attack)
    
    return attack * 0.15


def karplus_strong_enhanced(frequency: float, duration: float, sample_rate: int,
                            string_idx: int = 3) -> np.ndarray:
    """
    Enhanced Karplus-Strong with harmonics, body resonance, and realistic decay.
    
    Args:
        frequency: Fundamental frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        string_idx: String index (0-5) for thickness-dependent characteristics
    """
    n_samples = int(sample_rate * duration)
    buf_len = max(2, int(round(sample_rate / frequency)))
    
    # String-dependent parameters (thicker strings = more bass, longer sustain)
    thickness_factor = (5 - string_idx) / 5.0  # 0=thin, 1=thick
    damping = 0.4988 + thickness_factor * 0.0008  # Thicker = longer sustain
    brightness = 0.80 + thickness_factor * 0.20
    
    # Initialize buffer with shaped noise (warmer for bass strings)
    buf = np.random.uniform(-1.0, 1.0, buf_len) * brightness
    
    # Multi-stage filtering for natural tone
    # Low-pass for warmth
    buf = np.convolve(buf, [0.2, 0.6, 0.2], mode='same')
    # Add slight high-frequency content for realism
    buf += np.random.uniform(-0.08, 0.08, buf_len) * (1 - thickness_factor * 0.5)
    
    output = np.zeros(n_samples, dtype=np.float64)
    
    # Add pick attack transient
    attack = generate_pick_attack(sample_rate, duration_ms=10.0)
    attack_len = min(len(attack), n_samples)
    output[:attack_len] += attack[:attack_len] * 1.2
    
    # Karplus-Strong loop with frequency-dependent damping
    for i in range(n_samples):
        output[i] += buf[0] * 0.90
        
        # Two-point averaging with damping
        avg = (buf[0] + buf[1]) * 0.5
        
        # Frequency-dependent low-pass (more damping for high frequencies)
        if i % 4 == 0:  # Occasional extra damping for realism
            avg *= damping * 0.9985
        else:
            avg *= damping
        
        buf = np.roll(buf, -1)
        buf[-1] = avg
        
        # Add subtle string vibration noise
        if i % 80 == 0:
            buf[-1] += np.random.uniform(-0.0008, 0.0008) * thickness_factor
    
    # Add harmonics for richer tone
    output = add_harmonics(output, frequency, sample_rate, string_idx)
    
    return output


def add_harmonics(audio_signal: np.ndarray, fundamental: float, 
                  sample_rate: int, string_idx: int) -> np.ndarray:
    """Add natural harmonics (overtones) for realistic guitar timbre."""
    n_samples = len(audio_signal)
    output = audio_signal.copy()
    
    # Harmonic amplitudes (decreasing with harmonic number)
    # Guitar strings emphasize certain harmonics
    harmonics = [
        (2, 0.18),   # Octave - strong
        (3, 0.12),   # Perfect fifth
        (4, 0.09),   # Two octaves
        (5, 0.06),   # Major third
        (6, 0.04),   # Perfect fifth
        (7, 0.02),   # Minor seventh
    ]
    
    thickness_factor = (5 - string_idx) / 5.0
    
    for harmonic_num, amplitude in harmonics:
        harmonic_freq = fundamental * harmonic_num
        
        # Skip if harmonic exceeds Nyquist
        if harmonic_freq >= sample_rate / 2:
            continue
        
        # Thicker strings have stronger low harmonics
        if harmonic_num <= 3:
            amplitude *= (1 + thickness_factor * 0.4)
        else:
            amplitude *= (1 - thickness_factor * 0.2)
        
        # Generate harmonic with slight detuning for realism
        detune = np.random.uniform(0.997, 1.003)
        buf_len = max(2, int(round(sample_rate / (harmonic_freq * detune))))
        buf = np.random.uniform(-1, 1, buf_len) * 0.25
        
        harmonic_signal = np.zeros(n_samples, dtype=np.float64)
        damping = 0.495  # Harmonics decay faster
        
        for i in range(n_samples):
            harmonic_signal[i] = buf[0]
            avg = (buf[0] + buf[1]) * 0.5 * damping
            buf = np.roll(buf, -1)
            buf[-1] = avg
        
        output += harmonic_signal * amplitude
    
    return output


def note_frequency(string_idx: int, fret: int) -> float:
    """Return frequency for a given string + fret."""
    return STRING_FREQS[string_idx] * (FRET_SEMITONE ** fret)


def apply_string_coupling(signals: list, sample_rate: int) -> list:
    """Simulate sympathetic resonance between strings."""
    coupled = []
    for i, sig in enumerate(signals):
        coupled_sig = sig.copy()
        # Each string picks up subtle vibrations from neighbors
        for j, other_sig in enumerate(signals):
            if i != j:
                distance = abs(i - j)
                coupling_strength = 0.02 / distance  # Closer strings couple more
                coupled_sig += other_sig * coupling_strength
        coupled.append(coupled_sig)
    return coupled


def make_chord_wave(chord_notes: list, sample_rate: int, duration: float) -> np.ndarray:
    """
    Mix individual string waves for a full chord with authentic strum delay,
    string coupling, and realistic dynamics.
    """
    n_samples = int(sample_rate * duration)
    mix = np.zeros(n_samples, dtype=np.float64)

    # Natural strum timing with slight variation
    strum_delay_ms = 22  # Base delay between strings
    strum_samples = int(sample_rate * strum_delay_ms / 1000)
    
    # Generate individual string signals
    string_signals = []
    active_strings = []
    
    for i, (string_idx, fret) in enumerate(chord_notes):
        if fret is None:
            string_signals.append(None)
            continue
            
        freq = note_frequency(string_idx, fret)
        wave = karplus_strong_enhanced(freq, duration, sample_rate, string_idx)
        string_signals.append(wave)
        active_strings.append((i, string_idx, wave))
    
    # Apply string coupling for sympathetic resonance
    if len(active_strings) > 1:
        coupled_waves = apply_string_coupling([w for _, _, w in active_strings], sample_rate)
        for idx, (i, string_idx, _) in enumerate(active_strings):
            string_signals[i] = coupled_waves[idx]
    
    # Mix strings with strum timing and dynamics
    for i, (string_idx, fret) in enumerate(chord_notes):
        if fret is None or string_signals[i] is None:
            continue
            
        wave = string_signals[i]
        
        # Strum timing with natural variation
        timing_variation = np.random.uniform(-0.3, 0.3)
        delay = int((i * strum_samples) * (1 + timing_variation * 0.1))
        
        # String-dependent volume (bass strings slightly louder)
        thickness_factor = (5 - string_idx) / 5.0
        volume = 0.65 + thickness_factor * 0.15
        
        # Add slight velocity variation per string
        volume *= np.random.uniform(0.92, 1.08)
        
        end = min(delay + len(wave), n_samples)
        mix[delay:end] += wave[: end - delay] * volume

    # Normalize with soft clipping for warmth
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix /= (peak * 1.08)
        # Soft clipping for analog warmth
        mix = np.tanh(mix * 1.2) * 0.85
    
    return mix


def apply_body_resonance(audio_signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Simulate acoustic guitar body resonance with multiple resonant modes.
    Guitar bodies have characteristic resonant frequencies around 100Hz, 200Hz, 400Hz.
    """
    from scipy import signal as sp_signal
    output = audio_signal.copy()
    
    # Main body resonances (Helmholtz resonance + top plate modes)
    resonances = [
        (100, 0.08, 15),   # Helmholtz (air cavity)
        (200, 0.06, 12),   # First top plate mode
        (400, 0.04, 8),    # Second top plate mode
    ]
    
    for freq, amplitude, q_factor in resonances:
        # Create resonant filter
        sos = sp_signal.butter(2, [freq - freq/q_factor, freq + freq/q_factor], 
                           'bandpass', fs=sample_rate, output='sos')
        resonant = sp_signal.sosfilt(sos, output)
        output += resonant * amplitude
    
    return output


def apply_room_reverb(audio_signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Multi-tap delay reverb simulating room acoustics with early reflections
    and diffuse reverb tail.
    """
    from scipy import signal as sp_signal
    output = audio_signal.copy()
    
    # Early reflections (walls, ceiling, floor)
    early_reflections = [
        (23, 0.15),   # First wall
        (37, 0.12),   # Side wall
        (48, 0.10),   # Ceiling
        (61, 0.08),   # Back wall
        (79, 0.06),   # Floor
    ]
    
    for delay_ms, amplitude in early_reflections:
        delay_samples = int(sample_rate * delay_ms / 1000)
        if delay_samples < len(output):
            for i in range(delay_samples, len(output)):
                output[i] += output[i - delay_samples] * amplitude
    
    # Diffuse reverb tail (multiple cascading delays)
    reverb_delays = [
        (97, 0.25),
        (127, 0.20),
        (157, 0.15),
        (211, 0.10),
    ]
    
    for delay_ms, decay in reverb_delays:
        delay_samples = int(sample_rate * delay_ms / 1000)
        if delay_samples < len(output):
            for i in range(delay_samples, len(output)):
                output[i] += output[i - delay_samples] * decay
    
    # Damping filter (high frequencies decay faster in reverb)
    sos = sp_signal.butter(2, 4000, 'lowpass', fs=sample_rate, output='sos')
    output = sp_signal.sosfilt(sos, output)
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= (peak * 1.05)
    
    return output


def save_wav(filename: str, audio_signal: np.ndarray, sample_rate: int):
    data = (audio_signal * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, data)


def synthesize_acoustic_guitar_chord(chord_notes: list, sample_rate: int = SAMPLE_RATE,
                                     duration: float = DURATION_SEC) -> np.ndarray:
    """
    Complete acoustic guitar synthesis pipeline - modular and reusable.
    
    Args:
        chord_notes: List of (string_idx, fret) tuples
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        Synthesized audio signal as numpy array
    """
    from scipy import signal as sp_signal
    
    # 1. Generate raw chord with string synthesis
    wave = make_chord_wave(chord_notes, sample_rate, duration)
    
    # 2. Apply guitar body resonance
    wave = apply_body_resonance(wave, sample_rate)
    
    # 3. Apply room acoustics
    wave = apply_room_reverb(wave, sample_rate)
    
    # 4. Final EQ and polish
    # Slight bass boost for warmth
    sos_bass = sp_signal.butter(1, 150, 'lowpass', fs=sample_rate, output='sos')
    bass = sp_signal.sosfilt(sos_bass, wave) * 0.15
    wave += bass
    
    # Presence boost around 3kHz for clarity
    sos_presence = sp_signal.butter(2, [2500, 4000], 'bandpass', fs=sample_rate, output='sos')
    presence = sp_signal.sosfilt(sos_presence, wave) * 0.08
    wave += presence
    
    # Final normalization
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= (peak * 1.02)
    
    return wave


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🎸 Generating realistic acoustic guitar chord samples → {OUTPUT_DIR}\n")
    print("Using enhanced Karplus-Strong with harmonics, body resonance, and room reverb...\n")

    for chord_name, chord_notes in CHORDS.items():
        print(f"  ♪ Synthesizing {chord_name:3s} ...", end="", flush=True)
        wave = synthesize_acoustic_guitar_chord(chord_notes, SAMPLE_RATE, DURATION_SEC)
        out_path = os.path.join(OUTPUT_DIR, f"{chord_name}.wav")
        save_wav(out_path, wave, SAMPLE_RATE)
        print(f" ✓ saved → {chord_name}.wav")

    # Generate open strum (no chord detected fallback)
    print(f"  ♪ Synthesizing open ...", end="", flush=True)
    open_notes = [(i, 0) for i in range(6)]
    wave = synthesize_acoustic_guitar_chord(open_notes, SAMPLE_RATE, DURATION_SEC * 0.5)
    save_wav(os.path.join(OUTPUT_DIR, "open.wav"), wave, SAMPLE_RATE)
    print(f" ✓ saved → open.wav")

    print(f"\n✅ Complete! {len(CHORDS) + 1} high-quality chord files written to '{OUTPUT_DIR}'")
    print("🎵 The acoustic guitar synthesis includes:")
    print("   • Enhanced Karplus-Strong algorithm with pick attack")
    print("   • Natural harmonics and overtones")
    print("   • String coupling (sympathetic resonance)")
    print("   • Guitar body resonance modeling")
    print("   • Multi-tap room reverb")
    print("   • Frequency-dependent damping")
    print("   • Realistic strum timing variations")


if __name__ == "__main__":
    main()
