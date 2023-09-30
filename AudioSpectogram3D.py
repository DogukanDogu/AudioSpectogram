import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import librosa

audio_filename = "Dial Up Modem Handshake Sound.wav"
audio_data, sample_rate = librosa.load(audio_filename)

stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)

magnitude = np.abs(stft)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

# Create the mesh grid
x, y = np.meshgrid(np.arange(magnitude.shape[1]), np.arange(magnitude.shape[0]))

# Plot the surface
ax.plot_surface(x, y, magnitude, cmap="viridis")

# Set the axis labels
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_zlabel("Magnitude")

# Set the title
ax.set_title("3D Spectrogram of " + audio_filename)

# Show the plot
plt.show()
