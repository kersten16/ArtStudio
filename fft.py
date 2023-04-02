import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd
import numpy as np

fs = 44100
##def plotFreq():
##    
##    s_rate, signal = wavfile.read("./popSample.wav") 
##    print (s_rate,signal)
##    FFT = abs(scipy.fft.fft(signal))
##    freqs = fftpk.fftfreq(len(FFT), (1.0/s_rate))
##
##    plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])                                                          
##    plt.xlabel('Frequency (Hz)')
##    plt.ylabel('Amplitude')
##    plt.show()

def print_sound(indata, outdata, frames, time, status):
    volume_norm = np.linalg.norm(indata)*10
##    can set this to close on button release
    print ("|" * int(volume_norm))

def getFreq():
    ##can set this to be called on button press
    seconds = 3
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  
    wavfile.write('output.wav', fs, myrecording)   

    fs_rate, signal = wavfile.read("output.wav")
    l_audio = len(signal.shape)
    N = signal.shape[0]
    secs = N / float(fs_rate)
    Ts = 1.0/fs_rate 
    t = np.arange(0, secs, Ts) 
    FFT = abs(scipy.fft.fft(signal))
    FFT_side = FFT[range(N//2)] 
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)] 
    fft_freqs_side = np.array(freqs_side)

    volume=np.array(abs(FFT_side))
    audible=np.where(volume>5)

    HighestAudibleFrequency=max(freqs_side[audible])
    print(HighestAudibleFrequency)
    
    with sd.Stream(samplerate = fs ,callback=print_sound):
        sd.sleep(10000)

getFreq()

