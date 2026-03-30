# Virtual Acoustic Guitar

A real-time computer vision application that transforms your webcam into a virtual acoustic guitar. Using hand detection and gesture recognition, the system recognizes guitar chord shapes and strum motions to produce realistic acoustic guitar sounds.

## Overview

This application uses MediaPipe for hand tracking, custom algorithms for chord recognition and strum detection, and advanced audio synthesis to create an interactive virtual guitar experience.

### Key Features

- Real-time hand tracking with MediaPipe (21 landmarks per hand)
- Two-hand system: left hand for chord shapes, right hand for strumming
- 12 supported guitar chords with accurate detection
- Velocity-sensitive audio playback
- Enhanced Karplus-Strong synthesis for realistic acoustic guitar sound
- Visual feedback with fretboard display and hand skeleton overlay

### How It Works

**Left Hand**: Forms chord shapes (simulates fretting on a guitar neck)  
**Right Hand**: Strums up or down to trigger sound (simulates picking/strumming)

The system only produces sound when you actively strum with your right hand while holding a recognized chord shape with your left hand, mimicking the mechanics of playing a real guitar.

## Installation

### Prerequisites

- Python 3.10 or higher
- Webcam
- Windows, macOS, or Linux

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/virtual-acoustic-guitar.git
cd virtual-acoustic-guitar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate acoustic guitar sounds:
```bash
python generate_sounds.py
```

This creates a `sounds/` directory with WAV files for all supported chords using an enhanced Karplus-Strong synthesis algorithm.

4. Run the application:
```bash
python main.py
```

## Usage

### Basic Controls

| Key | Action |
|-----|--------|
| Q or ESC | Quit application |
| M | Toggle mute |
| R | Reset chord detection |

### Playing the Guitar

1. Position yourself in front of your webcam with good lighting
2. Hold your left hand in a chord shape (see Supported Chords below)
3. Move your right hand up and down to strum
4. The chord plays when you strum while holding a valid chord shape

### Supported Chords

| Chord | Type | Hand Shape Description |
|-------|------|------------------------|
| Am | Minor | Index semi-curled, middle and ring bent, pinky open |
| C | Major | Index and middle semi-curled, ring curled, pinky open |
| D | Major | Index and middle bent, ring and pinky open |
| Em | Minor | All four fingers fully extended |
| G | Major | Index, ring, and pinky curled; middle open |
| F | Major | All four fingers moderately curled (barre position) |
| E | Major | Index, middle, and ring curled; pinky open |
| A | Major | Index open, middle, ring, and pinky curled |
| B7 | Dominant 7th | Index and middle bent, ring and pinky open |
| Dm | Minor | Index and middle bent, ring curled, pinky open |
| E7 | Dominant 7th | Index and ring curled, middle and pinky open |
| A7 | Dominant 7th | Index open, middle curled, ring and pinky open |

## Technical Architecture

### System Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Hand Detection | MediaPipe Hands | Tracks 21 landmarks per hand in real-time |
| Chord Recognition | Custom geometry engine | Maps finger positions to chord shapes |
| Strum Detection | Motion tracking | Detects vertical hand movements and velocity |
| Audio Synthesis | Enhanced Karplus-Strong | Generates realistic acoustic guitar tones |
| Audio Playback | pygame.mixer | Manages sound playback with velocity control |
| Visual Interface | OpenCV | Renders video feed with overlays |

### Audio Synthesis Features

The enhanced Karplus-Strong algorithm includes:

- Realistic pick attack transients
- Natural harmonic overtones
- Guitar body resonance modeling
- Multi-tap room reverb
- String coupling (sympathetic resonance)
- Frequency-dependent damping
- Realistic strum timing variations

## Project Structure

```
virtual-acoustic-guitar/
├── main.py                 # Application entry point
├── hand_detector.py        # MediaPipe hand tracking wrapper
├── chord_engine.py         # Chord recognition logic
├── strum_engine.py         # Strum detection and velocity calculation
├── audio_engine.py         # Audio playback management
├── guitar_ui.py            # OpenCV rendering and UI
├── generate_sounds.py      # Acoustic synthesis engine
├── guitar_synthesis.py     # Reusable synthesis utilities
├── test_synthesis_quality.py # Audio quality testing
├── sounds/                 # Generated chord WAV files
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md              # This file
```

## Advanced Usage

### Custom Chord Synthesis

You can programmatically synthesize custom chords using the synthesis utilities:

```python
from guitar_synthesis import synthesize_chord, save_chord_to_file

# Define chord shape: [(string_index, fret), ...]
# string_index: 0=low E, 1=A, 2=D, 3=G, 4=B, 5=high E
custom_chord = [(1, 0), (2, 2), (3, 2), (4, 1), (5, 0)]  # Am shape

# Synthesize audio
audio = synthesize_chord(custom_chord, duration=4.0)

# Or save directly to file
save_chord_to_file(custom_chord, "custom_chord.wav")
```

### Synthesis Parameters

The synthesis engine supports customization:

```python
from guitar_synthesis import synthesize_chord

audio = synthesize_chord(
    chord_notes=[(1, 0), (2, 2), (3, 2)],
    duration=3.0,      # Duration in seconds
    sample_rate=44100  # Sample rate in Hz
)
```

## Performance Considerations

- Requires adequate lighting for reliable hand detection
- Webcam should be positioned to capture both hands clearly
- CPU usage depends on video resolution and frame rate
- Audio latency is minimized through efficient buffer management

## Troubleshooting

**Hand detection not working:**
- Ensure good lighting conditions
- Position hands clearly in frame
- Avoid cluttered backgrounds
- Check webcam permissions

**Chord not recognized:**
- Hold hand steady for 0.5 seconds
- Ensure fingers are clearly visible
- Try adjusting hand position slightly
- Press R to reset detection

**Audio issues:**
- Verify sounds/ directory contains WAV files
- Run `python generate_sounds.py` to regenerate sounds
- Check system audio settings
- Press M to unmute if needed

## Dependencies

- opencv-python: Video capture and rendering
- mediapipe: Hand tracking and landmark detection
- pygame: Audio playback
- numpy: Numerical computations
- scipy: Signal processing for audio synthesis

See `requirements.txt` for complete list with versions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe by Google for hand tracking technology
- Karplus-Strong algorithm for physical modeling synthesis
- pygame community for audio playback capabilities

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Areas for Contribution

- Additional chord support
- Improved chord recognition accuracy
- Enhanced audio synthesis quality
- UI/UX improvements
- Performance optimizations
- Documentation improvements

## Future Enhancements

- Support for more complex chords (9th, 11th, 13th)
- Fingerpicking pattern recognition
- MIDI output support
- Recording and playback functionality
- Customizable tunings
- Real-time audio effects (distortion, chorus, delay)
