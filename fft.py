import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd
import soundfile as sf
import threading
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume, ISimpleAudioVolume
# import obspython as obs

# if it doesn't work, put that every time we update the volume
# scenesource = obs.obs_frontend_get_current_scene()
# scene = obs.obs_scene_from_source(scenesource)
# #obs.script_log(obs.LOG_DEBUG,"Scene "+str(scene))
# sceneitem = obs.obs_scene_find_source(scene,sourcename)
# #obs.script_log(obs.LOG_DEBUG,"Scene item "+str(sceneitem))
# source = obs.obs_sceneitem_get_source(sceneitem)


fs = 44100
songToPlay=""
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate (IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume=None
#volume = cast(interface, POINTER(IAudioEndpointVolume))

def _play(sound):
    event =threading.Event()
    global volume
    def callback(outdata, frames, time, status):
        data = wf.buffer_read(frames, dtype='float32')
        if len(outdata) > len(data):
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
            raise sd.CallbackStop
        else:
            outdata[:] = data

    with sf.SoundFile(sound) as wf:
        stream = sd.RawOutputStream(samplerate=wf.samplerate,
                                    channels=wf.channels,
                                    callback=callback,
                                    blocksize=1024,
                                    finished_callback=event.set)
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            if session.Process and session.Process.name() == "python.exe":
                volume=session._ctl.QueryInterface(ISimpleAudioVolume)
                print(session.Process.name())
                break
        with stream:
            event.wait()

def _rec():
    def print_sound(indata, outdata, frames, time, status):
        volume_norm = np.linalg.norm(indata)*10
    ##    can set this to close on button release
        print ('|'*int(volume_norm))
        #volume.SetMasterVolumeLevelScalar(max(.10,min(float(volume_norm/60),1)),None)
        volume.SetMasterVolume(max(.10,min(float(volume_norm/60),1)),None)
        # obs.obs_source_set_volume(source,max(.10,min(float(volume_norm/60),1)))
    with sd.Stream(samplerate = fs ,callback=print_sound):
        sd.sleep(-1)

def _playsound(sound):
    new_thread = threading.Thread(target=_play, args=(sound,))
    new_thread.start()
    
def _recsound():
    new_thread = threading.Thread(target=_rec)
    new_thread.start()
    


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
    if HighestAudibleFrequency < 6000:
        songToPlay="classicalSample.wav"
    elif (HighestAudibleFrequency >= 6000 and HighestAudibleFrequency<8000):
        songToPlay="reggaeSample.wav"

    elif (HighestAudibleFrequency >= 8000 and HighestAudibleFrequency<10000):
        songToPlay="discoSample.wav"

    else:
        songToPlay="popSample.wav"
        
    _playsound(songToPlay)
    _recsound()

getFreq()

