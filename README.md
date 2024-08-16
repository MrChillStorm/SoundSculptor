# SoundSculptor
Collection of Python scripts for advanced audio processing, including EQ gain application and custom noise generation for simulated aircraft.

# Headset Filter, Noise Generator and Audio Splitter Scripts

This repository contains three Python scripts to assist in audio processing:

1. `headset-filter.py`: This script applies EQ gains to an input WAV file and writes the processed audio to an output file. It can work with both mono and stereo files. It uses Fast Fourier Transform (FFT) to transform the audio data to the frequency domain, applies the EQ gains, and uses Inverse FFT to transform the data back to the time domain.

2. `generate_noise.py`: This script is designed to generate noise similar to the input for simulated aircraft by matching the Power Spectral Density (PSD), spectrum, and loudness of the input with generated white noise. The length of the generated noise can be customized.

3. `split_stereo_wav.py`: This script takes a stereo WAV file as input and splits it into two separate WAV files, one for the left channel and one for the right channel.

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

### Audio Splitter Script

```bash
python3 split_stereo_wav.py <input_file.wav>
```
* `input_file.wav`: The stereo input wav file.

## Requirements

All scripts require Python 3 and the following Python packages:

* `numpy`
* `scipy`
* `pyloudnorm`
* `pydub`

You can install these packages with pip:

```bash
pip3 install numpy scipy pyloudnorm pydub
```

## Notes

The EQ gains used in the scripts are based on the Bose Aviation Headset X. You can modify these values as per your requirement. The `split_stereo_wav.py` script will create two files named `<input_file>-left.wav` and `<input_file>-right.wav` in the same directory as the input file.
