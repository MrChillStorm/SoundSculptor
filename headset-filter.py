#!/usr/bin/env python3

import argparse
import numpy as np
from scipy import interpolate
from scipy.fftpack import fft, ifft
from scipy.io import wavfile

# Create the parser
parser = argparse.ArgumentParser(description='This Python script applies EQ gains to an input WAV file and writes the processed audio to an output file. The script works with both mono and stereo files. It uses Fast Fourier Transform (FFT) to transform the audio data to frequency domain and then applies the EQ gains. Inverse FFT is used to transform the data back to time domain. The EQ gains are specified in dB and are interpolated to match the FFT frequencies.')
parser.add_argument('input_file', type=str, nargs='?', help='The input wav file')
parser.add_argument('output_file', type=str, nargs='?', default='output.wav', help='The output wav file')

# Parse the arguments
args = parser.parse_args()

if args.input_file is None:
    parser.print_help()
    exit(1)

# Your EQ points (The Bose Aviation Headset X)
freqs = np.array([0, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 24000])
gains = np.array([0, -2.2, -7.0, 0.0, -5.2, -7.8, -4.0, -9.0, -8.4, -7.8, -10.4, -9.8, -10.6, -10.2, -7.4, -8.8, -13.4, -13.2, -9.8, -11, -16.4, -23.0, -19.2, -18.6, -21.4, -19.0, -19.0])

# Convert gains from dB to linear
gains_linear = 10 ** (gains / 20)

# Load wav file
sample_rate, data = wavfile.read(args.input_file)

# Handle mono/stereo files
if len(data.shape) == 1:
    data = np.expand_dims(data, axis=-1)

# Calculate FFT and apply EQ gains
output_data = []
output_data_left = []
output_data_right = []

for channel in range(data.shape[1]):
    # Calculate FFT
    fft_data = fft(data[:, channel])

    # Calculate frequencies for FFT data
    fft_freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)

    # Interpolate gains to match FFT frequencies
    interpolator = interpolate.interp1d(freqs, gains_linear, bounds_error=False, fill_value=(gains_linear[0], gains_linear[-1]))
    fft_gains = interpolator(np.abs(fft_freqs))

    # Apply gains and perform inverse FFT
    eq_data = ifft(fft_data * fft_gains)

    # Append to output data
    output_data.append(np.real(eq_data))
    if channel == 0:  # left channel
        output_data_left.append(np.real(eq_data))
    elif channel == 1:  # right channel
        output_data_right.append(np.real(eq_data))

# Convert back to stereo/mono and 16-bit PCM
output_data = np.stack(output_data, axis=-1).astype(np.int16)

# Write to output file
wavfile.write(args.output_file, sample_rate, output_data)

# If stereo, also save left and right channels separately
if data.shape[1] > 1:
    output_data_left = np.stack(output_data_left, axis=-1).astype(np.int16)
    output_data_right = np.stack(output_data_right, axis=-1).astype(np.int16)
    wavfile.write(args.output_file.replace(".wav", "-left.wav"), sample_rate, output_data_left)
    wavfile.write(args.output_file.replace(".wav", "-right.wav"), sample_rate, output_data_right)
