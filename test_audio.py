"""
test_audio.py

Test audio.py using an MP3 file.

Usage:
    python test_audio.py dog_bark.mp3
"""

import sys
import os
import numpy as np
import librosa

from audio import classify_vocalization


TARGET_SAMPLE_RATE = 16000


def main():

    print("=" * 65)
    print("YAMNET DOG VOCALIZATION TEST")
    print("=" * 65)

    # --------------------------------------------------------
    # Check argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print("\nUsage:")
        print("python test_audio.py dog_bark.mp3")
        return

    audio_path = sys.argv[1]

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(audio_path):

        print(
            f"\nERROR: File not found:\n{audio_path}"
        )
        return

    print(f"\nAudio file: {audio_path}")

    # --------------------------------------------------------
    # Load MP3
    # --------------------------------------------------------

    print("\nLoading MP3...")

    try:

        waveform, sample_rate = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True
        )

    except Exception as e:

        print(
            f"\nERROR loading MP3:\n{e}"
        )
        return

    # --------------------------------------------------------
    # Ensure float32
    # --------------------------------------------------------

    waveform = np.asarray(
        waveform,
        dtype=np.float32
    )

    waveform = np.clip(
        waveform,
        -1.0,
        1.0
    )

    # --------------------------------------------------------
    # Audio information
    # --------------------------------------------------------

    duration = (
        len(waveform)
        / TARGET_SAMPLE_RATE
    )

    rms = np.sqrt(
        np.mean(
            waveform ** 2
        )
    )

    print("\nAudio information")
    print("-" * 65)

    print(
        f"Sample rate : {sample_rate} Hz"
    )

    print(
        f"Channels    : Mono"
    )

    print(
        f"Samples     : {len(waveform)}"
    )

    print(
        f"Duration    : {duration:.2f} seconds"
    )

    print(
        f"Min         : {np.min(waveform):.4f}"
    )

    print(
        f"Max         : {np.max(waveform):.4f}"
    )

    print(
        f"RMS         : {rms:.4f}"
    )

    # --------------------------------------------------------
    # Run YAMNet
    # --------------------------------------------------------

    print("\nRunning YAMNet...")
    print("Please wait...")

    result = classify_vocalization(
        waveform
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("RESULT")
    print("=" * 65)

    print(
        f"Detected sound : {result['label']}"
    )

    print(
        f"Confidence     : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Raw YAMNet     : "
        f"{result['raw_top_class']}"
    )

    print(
        f"Model available: "
        f"{result['available']}"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if result["label"] == "Barking":

        print("\n🐕 Barking detected.")

    elif result["label"] == "Growling":

        print("\n🐕 Growling detected.")

    elif result["label"] == "Howling":

        print("\n🐕 Howling detected.")

    elif result["label"] == "Whimpering":

        print("\n🐕 Whimpering detected.")

    elif result["label"] == "Quiet":

        print("\n🔇 No strong dog vocalization detected.")

    if not result["available"]:

        print(
            "\nWARNING: YAMNet was not available."
        )


if __name__ == "__main__":
    main()