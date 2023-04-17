import pyaudio
import numpy as np
import threading
import time
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd
import soundfile as sf
import threading
import subprocess
import os
import wave


fs = 44100
songToPlay=""



def set_volume(value):
    volume_percentage = int(volume_level * 100)
    cmd = "osascript -e 'set volume output volume {}'".format(volume_percentage)
    subprocess.run(cmd, shell=True, check=True)

def _rec():
    CHUNK = 1024
    RATE = 44100

    def audio_callback(in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        volume_norm = np.linalg.norm(audio_data) * 10
        print('|' * int(volume_norm))
        set_volume(max(0.1, min(float(volume_norm / 60), 1)))
        return (in_data, pyaudio.paContinue)

    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=RATE,
                        input=True,
                        output=False,
                        frames_per_buffer=CHUNK,
                        stream_callback=audio_callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    audio.terminate()

def _playsound(sound):
    CHUNK = 1024
    RATE = 44100

    def play_audio_callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)

    audio = pyaudio.PyAudio()
    wf = wave.open(sound, 'rb')

    stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        input=False,
                        output=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=play_audio_callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    audio.terminate()

def getFreq():
    global songToPlay
    songToPlay = "classicalSample.wav"  # Default value

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
    if HighestAudibleFrequency < 20000:
        songToPlay="classicalSample.wav"
    elif (HighestAudibleFrequency >= 20000 and HighestAudibleFrequency<22000):
        songToPlay="reggaeSample.wav"

    elif (HighestAudibleFrequency >= 22000 and HighestAudibleFrequency<24000):
        songToPlay="discoSample.wav"

    else:
        songToPlay="popSample.wav"

    print(f"Selected song: {songToPlay}")

    # Make sure the specified file paths exist
    if not os.path.isfile(songToPlay):
        print(f"Error: File '{songToPlay}' not found.")
        return
    _playsound(songToPlay)
    _rec()

getFreq()
