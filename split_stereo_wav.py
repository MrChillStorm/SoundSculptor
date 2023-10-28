#!/usr/bin/env python3

import sys
from pydub import AudioSegment

def split_stereo_wav(input_file):
    # Load the stereo WAV file
    audio = AudioSegment.from_wav(input_file)

    # Split the stereo audio into left and right channels
    left_channel = audio.split_to_mono()[0]
    right_channel = audio.split_to_mono()[1]

    # Get the base filename without extension
    base_filename = input_file.rsplit('.', 1)[0]

    # Define output filenames
    left_output_file = f"{base_filename}-left.wav"
    right_output_file = f"{base_filename}-right.wav"

    # Save left and right channels as separate WAV files
    left_channel.export(left_output_file, format="wav")
    right_channel.export(right_output_file, format="wav")

    print(f"Split into {left_output_file} and {right_output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_stereo_wav.py <input_file.wav>")
        sys.exit(1)

    input_file = sys.argv[1]
    split_stereo_wav(input_file)
