#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pyloudnorm as pyln
from scipy import signal
from scipy.interpolate import interp1d
import scipy.io.wavfile as wav
import sys

# Bose Aviation Headset X EQ
freqs = np.array([0, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 24000])
gains = np.array([0, -2.2, -7.0, 0.0, -5.2, -7.8, -4.0, -9.0, -8.4, -7.8, -10.4, -9.8, -10.6, -10.2, -7.4, -8.8, -13.4, -13.2, -9.8, -11, -16.4, -23.0, -19.2, -18.6, -21.4, -19.0, -19.0])
gains_linear = 10 ** (gains / 20)

def apply_headset_filter(fs, data):
    """Apply headset EQ filter in frequency domain"""
    fft_data = np.fft.fft(data)
    f = interp1d(freqs, gains_linear, kind='linear', fill_value="extrapolate")
    xnew = np.linspace(0, fs//2, len(fft_data)//2)
    ynew = f(xnew)
    ynew = np.concatenate((ynew, ynew[::-1]))
    filtered_fft = fft_data * ynew
    return np.real(np.fft.ifft(filtered_fft))

def extract_magnitude_spectrogram(data, fs, n_fft, hop_length):
    """
    Extract magnitude spectrogram using STFT.
    """
    _, _, Zxx = signal.stft(data, fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    return np.abs(Zxx)

def match_spectral_envelope(source_data, target_length, fs, n_fft, hop_length):
    """
    Advanced synthesis using STFT-based spectral envelope matching
    with temporal modulation preservation.
    """
    # Extract magnitude spectrogram from source
    mag_source = extract_magnitude_spectrogram(source_data, fs, n_fft, hop_length)

    # Generate white noise of target length
    noise = np.random.randn(target_length)

    # Get noise STFT
    _, _, Zxx_noise = signal.stft(noise, fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    mag_noise = np.abs(Zxx_noise)
    phase_noise = np.angle(Zxx_noise)

    # Compute average spectral envelope from source
    avg_spectrum_source = np.mean(mag_source, axis=1)
    avg_spectrum_noise = np.mean(mag_noise, axis=1)

    # Create spectral shaping filter
    spectrum_ratio = avg_spectrum_source / (avg_spectrum_noise + 1e-10)
    spectrum_ratio = np.maximum(spectrum_ratio, 0)  # Ensure non-negative

    # Apply spectral shaping to each time frame
    mag_shaped = mag_noise * spectrum_ratio[:, np.newaxis]

    # Preserve temporal modulation patterns
    # Match modulation depth (temporal variation in each frequency band)
    mod_depth_source = np.std(mag_source, axis=1)
    mod_depth_noise = np.std(mag_shaped, axis=1)
    mod_ratio = mod_depth_source / (mod_depth_noise + 1e-10)

    # Apply modulation matching
    mag_mean = np.mean(mag_shaped, axis=1, keepdims=True)
    mag_shaped = mag_mean + (mag_shaped - mag_mean) * mod_ratio[:, np.newaxis]

    # Reconstruct with shaped magnitude and noise phases
    Zxx_result = mag_shaped * np.exp(1j * phase_noise)

    # Inverse STFT with explicit length to avoid truncation artifacts
    _, result = signal.istft(Zxx_result, fs, nperseg=n_fft, noverlap=n_fft-hop_length)

    # Ensure exact length
    if len(result) > target_length:
        result = result[:target_length]
    elif len(result) < target_length:
        result = np.pad(result, (0, target_length - len(result)), mode='constant')

    return result

def match_higher_order_statistics(data, noise, fs):
    """
    Match higher-order statistics: kurtosis and skewness
    to capture transient characteristics.
    """
    # Compute statistics in frequency bands
    # Use a filter bank approach
    n_bands = 32
    # Ensure we stay well below Nyquist frequency
    max_freq = fs / 2.0 - 100  # Leave some margin
    band_edges = np.logspace(np.log10(20), np.log10(max_freq), n_bands + 1)

    result = noise.copy()

    for i in range(n_bands):
        # Design bandpass filter
        try:
            sos = signal.butter(4, [band_edges[i], band_edges[i+1]], 'bandpass', fs=fs, output='sos')

            # Filter both signals
            data_band = signal.sosfilt(sos, data)
            noise_band = signal.sosfilt(sos, result)
        except ValueError:
            # Skip this band if filter design fails
            continue

        # Match kurtosis (captures impulsiveness)
        from scipy.stats import kurtosis
        kurt_data = kurtosis(data_band)
        kurt_noise = kurtosis(noise_band)

        # Simple kurtosis matching through non-linear transformation
        if kurt_noise != 0 and not np.isnan(kurt_data):
            alpha = np.clip(np.sqrt(abs(kurt_data / kurt_noise)), 0.5, 2.0)
            noise_band_adjusted = np.sign(noise_band) * (np.abs(noise_band) ** alpha)

            # Replace the band in result
            result_filtered = signal.sosfilt(sos, result)
            result = result - result_filtered + noise_band_adjusted

    return result

def process_audio_enhanced(input_file, output_file, desired_length_seconds,
                          apply_filter, stft_quality='high'):
    """
    Enhanced audio processing with STFT-based synthesis
    """
    NORMALIZATION_FACTOR = 32768.0

    # STFT parameters based on quality
    if stft_quality == 'high':
        n_fft = 4096      # ~85ms windows
        hop_length = 512   # 87.5% overlap for smooth reconstruction
    elif stft_quality == 'medium':
        n_fft = 2048      # ~43ms windows
        hop_length = 512   # 75% overlap
    else:  # low
        n_fft = 1024      # ~21ms windows
        hop_length = 256   # 75% overlap

    # Load input file
    fs, data = wav.read(input_file)
    data = data.astype(np.float64) / NORMALIZATION_FACTOR

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    target_samples = fs * desired_length_seconds
    processed_data = np.zeros((target_samples, data.shape[1]))

    for channel in range(data.shape[1]):
        data_channel = data[:, channel]

        print(f"Processing channel {channel + 1}/{data.shape[1]}...")

        # Step 1: STFT-based spectral envelope matching with modulation preservation
        print("  - Matching spectral envelope and temporal modulation...")
        result = match_spectral_envelope(data_channel, target_samples, fs, n_fft, hop_length)

        # Step 2: Match higher-order statistics for transient characteristics
        print("  - Matching higher-order statistics...")
        result = match_higher_order_statistics(data_channel, result, fs)

        # Step 3: Fine-tune with traditional PSD matching
        print("  - Fine-tuning with PSD matching...")
        _, psd_data = signal.welch(data_channel, fs, nperseg=min(len(data_channel)//4, 8192))
        freqs_psd, psd_result = signal.welch(result, fs, nperseg=min(len(result)//4, 8192))

        # Smooth the ratio to avoid artifacts
        ratio = np.sqrt(psd_data / (psd_result + 1e-10))
        # Apply smoothing
        from scipy.ndimage import gaussian_filter1d
        ratio_smooth = gaussian_filter1d(ratio, sigma=2)

        # Interpolate to FFT bins
        ratio_interp = np.interp(
            np.linspace(0, fs // 2, len(np.fft.rfft(result))),
            freqs_psd,
            ratio_smooth
        )

        # Apply in frequency domain
        result_fft = np.fft.rfft(result)
        result_fft *= ratio_interp
        result = np.fft.irfft(result_fft, len(result))

        # Step 4: Match loudness (BEFORE headset filter)
        print("  - Matching loudness...")
        meter = pyln.Meter(fs)
        try:
            loudness_input = meter.integrated_loudness(data_channel)
            loudness_output = meter.integrated_loudness(result)
            result = pyln.normalize.loudness(result, loudness_output, loudness_input)
        except ValueError:
            print("  - Warning: Could not measure loudness, skipping normalization")

        processed_data[:, channel] = result

    # Apply headset filter
    if apply_filter:
        print("Applying headset filter...")
        for channel in range(data.shape[1]):
            processed_data[:, channel] = apply_headset_filter(fs, processed_data[:, channel])

    # Prepare output
    if processed_data.shape[1] == 1:
        processed_data = processed_data.flatten()

    processed_data = np.clip(processed_data, -1.0, 1.0)
    processed_data = (processed_data * NORMALIZATION_FACTOR).astype(np.int16)

    # Write output
    wav.write(output_file, fs, processed_data)
    print(f"Output written to {output_file}")

    # Split stereo if needed
    if processed_data.ndim > 1 and processed_data.shape[1] == 2:
        split_stereo_to_mono_files(output_file, processed_data, fs)

def split_stereo_to_mono_files(filename, data, sample_rate):
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    base_filename, ext = os.path.splitext(filename)
    left_filename = f"{base_filename}-left{ext}"
    right_filename = f"{base_filename}-right{ext}"
    wav.write(left_filename, sample_rate, left_channel)
    wav.write(right_filename, sample_rate, right_channel)

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced aircraft noise generator using STFT-based spectral envelope matching, '
                   'temporal modulation preservation, and higher-order statistics matching.'
    )
    parser.add_argument('input_file', help='Input WAV file path.')
    parser.add_argument('-o', '--output_file', default='output.wav',
                       help='Output WAV file path. Default is output.wav.')
    parser.add_argument('-l', '--length', type=int, default=15,
                       help='Desired length of the processed audio in seconds. Default is 15.')
    parser.add_argument('-f', '--filter', action='store_true',
                       help='Apply headset filter to the output.')
    parser.add_argument('-q', '--quality', choices=['low', 'medium', 'high'], default='high',
                       help='STFT processing quality. Higher quality = better results but slower. Default is high.')
    parser.add_argument('-s', '--seed', type=int, default=None,
                       help='Random seed for reproducible noise generation.')
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    process_audio_enhanced(args.input_file, args.output_file, args.length, args.filter, args.quality)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nEnhanced Aircraft Noise Generator")
        print("=" * 50)
        print("\nThis script generates realistic aircraft noise using advanced techniques:")
        print("  • STFT-based spectral envelope matching")
        print("  • Temporal modulation preservation")
        print("  • Higher-order statistics matching (kurtosis)")
        print("  • Multi-stage PSD matching with smoothing")
        print("\nUsage: python3 script.py <input_file> [-o output_file] [-l length] [-f] [-q quality] [-s seed]")
        print("\nArguments:")
        print("  <input_file>  Path to the input WAV file.")
        print("  -o            Path to the output WAV file. Default is 'output.wav'.")
        print("  -l            Desired length in seconds. Default is 15.")
        print("  -f            Apply headset filter to the output.")
        print("  -q            Quality: low/medium/high. Default is high.")
        print("  -s            Random seed for reproducible results.")
        print("\nExample: python3 script.py engine_noise.wav -o output_noise.wav -l 30 -f -q high -s 42\n")
    else:
        main()