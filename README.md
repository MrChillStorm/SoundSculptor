# SoundSculptor
Collection of Python scripts for advanced audio processing, including EQ gain application and custom noise generation for simulated aircraft.

# Headset Filter and Noise Generator Scripts

This repository contains two Python scripts to assist in audio processing:

1. `headset-filter.py`: This script applies EQ gains to an input WAV file and writes the processed audio to an output file. It can work with both mono and stereo files. It uses Fast Fourier Transform (FFT) to transform the audio data to the frequency domain, applies the EQ gains, and uses Inverse FFT to transform the data back to the time domain.

2. `generate_noise.py`: This script is designed to generate noise similar to the input for simulated aircraft by matching the Power Spectral Density (PSD), spectrum, and loudness of the input with generated white noise. The length of the generated noise can be customized.

## Usage

### Headset Filter Script

```bash
python3 headset-filter.py <input_file> <output_file>
```
* `input_file`: The input wav file.
* `output_file`: The output wav file. Default is `output.wav`.

### Noise Generator Script

```bash
python3 generate_noise.py <input_file> [-o output_file] [-l length] [-f]
```
* `input_file`: Path to the input WAV file.
* `-o, --output_file`: Path to the output WAV file. Default is `output.wav`.
* `-l, --length`: Desired length of the processed audio in seconds. Default is 15.
* `-f, --filter`: Apply headset filter to the output.

## Requirements

Both scripts require Python 3 and the following Python packages:

* `numpy`
* `scipy`
* `argparse`
* `pyloudnorm`

You can install these packages with pip:

```bash
pip install numpy scipy argparse pyloudnorm
```

## Notes

The EQ gains used in the scripts are based on the Bose Aviation Headset X. You can modify these values as per your requirement.
