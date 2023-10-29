#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pyloudnorm as pyln
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d
import scipy.io.wavfile as wav
import scipy.signal
import sys

# Your EQ points (The Bose Aviation Headset X)
freqs = np.array([0, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 24000])
gains = np.array([0, -2.2, -7.0, 0.0, -5.2, -7.8, -4.0, -9.0, -8.4, -7.8, -10.4, -9.8, -10.6, -10.2, -7.4, -8.8, -13.4, -13.2, -9.8, -11, -16.4, -23.0, -19.2, -18.6, -21.4, -19.0, -19.0])

# Convert gains from dB to linear
gains_linear = 10 ** (gains / 20)

def apply_headset_filter(fs, data):
    # Compute FFT
    fft_data = fft(data)
    # Create the EQ filter using linear interpolation
    f = interp1d(freqs, gains_linear, kind='linear', fill_value="extrapolate")
    xnew = np.linspace(0, fs//2, len(fft_data)//2)
    ynew = f(xnew)
    ynew = np.concatenate((ynew, ynew[::-1]))  # mirror for negative frequencies
    # Apply EQ filter to the FFT data
    filtered_fft = fft_data * ynew
    # Compute inverse FFT
    return np.real(ifft(filtered_fft))

def split_stereo_to_mono_files(filename, data, sample_rate):
    # Split the data into left and right channels
    left_channel = data[:, 0]
    right_channel = data[:, 1]

    # Output filenames
    base_filename, ext = os.path.splitext(filename)
    left_filename = f"{base_filename}-left{ext}"
    right_filename = f"{base_filename}-right{ext}"

    # Write left and right channels to separate files
    wav.write(left_filename, sample_rate, left_channel)
    wav.write(right_filename, sample_rate, right_channel)

def process_audio(input_file, output_file, desired_length_seconds, apply_filter):
    # Constants
    NORMALIZATION_FACTOR = 32768.0

    # Load input file
    fs, data = wav.read(input_file)

    # Convert to floating point
    data = data.astype(np.float64)

    # Normalize to -1.0 to 1.0 range
    data /= NORMALIZATION_FACTOR

    # Make sure the data is 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Initialize an array to hold the processed audio data
    processed_data = np.zeros((fs * desired_length_seconds, data.shape[1]))

    # Process each channel separately
    for channel in range(data.shape[1]):

        # Select the current channel
        data_channel = data[:, channel]

        # Generate white noise
        noise = np.random.normal(size=fs * desired_length_seconds)

        # PSD Matching
        # Calculate the PSD of the input signal and the noise
        freqs1, psd1 = scipy.signal.welch(data_channel, fs)
        freqs2, psd2 = scipy.signal.welch(noise, fs)

        # Create a filter to match the PSDs
        ratio = np.sqrt(psd1 / psd2)
        ratio = np.interp(np.linspace(0, fs // 2, len(np.fft.rfft(noise))), freqs1, ratio)

        # Apply the filter to the noise
        result = np.fft.irfft(np.fft.rfft(noise) * ratio, len(noise))

        # Apply Fourier Transform to get the spectrum
        spectrum1 = np.fft.fft(data_channel)
        spectrum2 = np.fft.fft(result)

        # Calculate the phase of the second file's spectrum
        phase2 = np.angle(spectrum2)

        # Check if lengths of the spectrum1 and spectrum2 match
        if len(spectrum1) != len(spectrum2):
            # Create an interpolation function based on the spectrum1 frequencies and amplitudes
            f = interp1d(np.fft.fftfreq(len(spectrum1)), np.abs(spectrum1), fill_value="extrapolate")

            # Use the interpolation function to resize the spectrum1 amplitudes to match spectrum2 frequencies
            new_spectrum1_amplitudes = f(np.fft.fftfreq(len(spectrum2)))
        else:
            new_spectrum1_amplitudes = np.abs(spectrum1)

        # Apply the resized spectrum1 amplitudes to the spectrum2 phases
        new_spectrum2 = new_spectrum1_amplitudes * np.exp(1j * phase2)

        # Apply Inverse Fourier Transform to get the new data
        result = np.fft.ifft(new_spectrum2).real

        # Measure the loudness of input and result
        meter = pyln.Meter(fs)  # create a Meter instance
        loudness_input = meter.integrated_loudness(data_channel)
        loudness_output = meter.integrated_loudness(result)

        # Match the loudness of the result to the input
        result_loudness_matched = pyln.normalize.loudness(result, loudness_output, loudness_input)

        # Store the processed data
        processed_data[:len(result_loudness_matched), channel] = result_loudness_matched

        # Apply the headset filter if specified
        if apply_filter:
            for channel in range(data.shape[1]):
                processed_data[:, channel] = apply_headset_filter(fs, processed_data[:, channel])

    # Make sure the processed data is 1D for mono audio
    if processed_data.shape[1] == 1:
        processed_data = processed_data.flatten()

    # Convert back to 16-bit data
    processed_data = (processed_data * NORMALIZATION_FACTOR).astype(np.int16)

    # Write result to file
    wav.write(output_file, fs, processed_data)

    # If the audio is stereo, save the left and right channels to separate files
    if processed_data.ndim > 1 and processed_data.shape[1] == 2:
        split_stereo_to_mono_files(output_file, processed_data, fs)

def main():
    parser = argparse.ArgumentParser(description='Generate noise similar to the input for simulated aircraft by matching the Power Spectral Density (PSD), spectrum, and loudness of the input with generated white noise.')
    parser.add_argument('input_file', help='Input WAV file path.')
    parser.add_argument('-o', '--output_file', default='output.wav', help='Output WAV file path. Default is output.wav.')
    parser.add_argument('-l', '--length', type=int, default=15, help='Desired length of the processed audio in seconds. Default is 15.')
    parser.add_argument('-f', '--filter', action='store_true', help='Apply headset filter to the output.')
    args = parser.parse_args()

    process_audio(args.input_file, args.output_file, args.length, args.filter)

if __name__ == "__main__":
    if len(sys.argv)==1:
        print("\nThis script is designed to generate noise similar to the input for simulated"
              "\naircraft. The noise is generated by creating white noise and modifying it to"
              "\nmatch the Power Spectral Density (PSD), spectrum, and loudness of the input."
              "\nThe length of the generated noise can be customized."
              "\n\nUsage: python3 script.py <input_file> [-o output_file] [-l length] [-f]"
              "\n\nArguments:"
              "\n  <input_file>  Path to the input WAV file."
              "\n  -o            Path to the output WAV file. Default is 'output.wav'."
              "\n  -l            Desired length of the generated audio in seconds. Default is 15."
              "\n  -f            Apply headset filter to the output."
              "\n\nExample: python3 script.py engine_noise.wav -o output_noise.wav -l 30 -f\n")
    else:
        main()
