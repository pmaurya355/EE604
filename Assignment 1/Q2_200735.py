import numpy as np
import librosa
import matplotlib.pyplot as plt

def solution(audio_path):
    audio, sample_rate = librosa.load(audio_path, sr = None)

    # Define parameters
    n_fft = 2048
    hop_length = 512
    fmax = 22000

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y = audio, sr = sample_rate, n_fft = n_fft, hop_length = hop_length, fmax = fmax)

    # Apply the Fourier transform to the Mel spectrogram
    spectrum = np.fft.fft(mel_spectrogram)

    # Calculate the maximum absolute value in the spectrum
    maximum = abs(spectrum).max()

    # plt.figure(figsize = (10, 4))
    # librosa.display.specshow(spectrum, sr = sample_rate, x_axis = 'time', y_axis = 'mel')
    # plt.colorbar(format = '%+2.0f dB')
    # plt.title('Mel Spectrogram (dB)')
    # plt.show()

    # Classify the audio based on the maximum value in the spectrum
    if maximum < 5000:
        class_name = 'cardboard'
    else:
        class_name = 'metal'

    return class_name


