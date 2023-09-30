import numpy as np
import matplotlib.pyplot as plt
import librosa


#Load the audio file:
audio_filename = "Dial Up Modem Handshake Sound.wav"
audio_data, sample_rate = librosa.load(audio_filename)

#Compute the short-time Fourier transform (STFT) of the audio signal:
stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)

#Compute the magnitude of the STFT:
magnitude = np.abs(stft)

#Plot the spectrogram:
plt.figure(figsize=(10, 6))
plt.pcolormesh(magnitude, cmap="viridis")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram of " + audio_filename)
plt.colorbar()
plt.show()
