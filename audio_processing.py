import  librosa
import matplotlib.pyplot as plt
import numpy as np

file_path='audios/WhatsApp Audio 2025-03-13 at 20.42.07_e4969cc9.opus'
y,sr=librosa.load(file_path,sr=22050)
print("sample rate is =",sr)
print("Audio array shape=",y.shape)

plt.figure(figsize=(12,4))
librosa.display.waveshow(y,sr=sr)
plt.title("wave form of audio")
plt.xlabel("Time(s)")
plt.ylabel("amplitude")
plt.show()
##########################
D=librosa.stft(y)
S_db=librosa.amplitude_to_db(abs(D),ref=np.max)
plt.figure(figsize=(12,4))
librosa.display.specshow(S_db,sr=sr,x_axis='time',y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectogram (dB)")
plt.show()

##################################
mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs,sr=sr,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title("MFCCS")
plt.show()

#################################
chroma=librosa.feature.chroma_stft(y=y,sr=sr)
plt.figure(figsize=(10,4))
librosa.display.specshow(chroma,sr=sr,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title("CHROMA")
plt.show()

#########################
spectral_centroid=librosa.feature.spectral_centroid(y=y,sr=sr)
plt.figure(figsize=(10,4))
librosa.display.specshow(spectral_centroid,color='r',label='Feature')
librosa.display.waveshow(y,sr=sr,alpha=0.5)
plt.legend(loc='upper right')
plt.title("Spectral Centroid")
plt.show()