import matplotlib.pyplot as plt
import matplotlib.transforms as mpt
import librosa
import librosa.display
from pydub import AudioSegment
import numpy as np
from BeatNet.BeatNet import BeatNet
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor, majmin_targets_to_chord_labels
from madmom.audio.chroma import DeepChromaProcessor
from madmom.evaluation.chords import encode as encode_chords, merge_chords, reduce_to_triads
import os
import soundfile as sf
from scipy.spatial.distance import cdist


#display chromagram

def plot_specgram(s, name, bounds=None):
    figure = plt.figure(figsize=(10,8), dpi=80)
    ax = figure.add_subplot()
    trans = mpt.blended_transform_factory(
            ax.transData, ax.transAxes)
    img = librosa.display.specshow(s, y_axis='log', x_axis='time', ax=ax)
    title = 'spec_gram' + name
    ax.set(title=title)

    if bounds is not None:
        ax.vlines(bounds, 0, 1, color='linen', linestyle='--',
          linewidth=2, alpha=0.9, label='Segment boundaries',
          transform=trans)
        ax.legend() 

    figure.colorbar(img, ax=ax)

def plot_specshow(chroma, name, bounds=None):
    figure = plt.figure(figsize=(10,8), dpi=80)
    ax = figure.add_subplot()
    trans = mpt.blended_transform_factory(
            ax.transData, ax.transAxes)
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    title = 'chroma_cqt' + name
    ax.set(title=title)

    if bounds is not None:
        ax.vlines(bounds, 0, 1, color='linen', linestyle='--',
          linewidth=2, alpha=0.9, label='Segment boundaries',
          transform=trans)
        ax.legend() 

    figure.colorbar(img, ax=ax)

def plot_centroid(cent, cent2, name, name2, S, bounds=None):
    times = librosa.times_like(cent)
    fig, ax = plt.subplots()
    trans = mpt.blended_transform_factory(
            ax.transData, ax.transAxes)
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.plot(times, cent.T, label='Spectral centroid ' + name, color='r')
    ax.plot(times, cent2.T, label='Spectral centroid ' + name2, color='b')
    ax.set(title='spectral centroids')

    if bounds is not None:
        ax.vlines(bounds, 0, 1, color='green', linestyle='--',
          linewidth=2, alpha=0.9, label='Segment boundaries',
          transform=trans)
        ax.legend(loc='lower right')

def plot_tonal_centroids(tonnetz, name, bounds = None):
    figure = plt.figure(figsize=(10,8), dpi=80)
    ax = figure.add_subplot()
    trans = mpt.blended_transform_factory(
            ax.transData, ax.transAxes)
    img = librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time', ax=ax)
    title = 'tonal centroids ' + name
    ax.set(title=title)

    if bounds is not None:
        ax.vlines(bounds, 0, 1, color='green', linestyle='--',
          linewidth=2, alpha=0.9, label='Segment boundaries',
          transform=trans)
        ax.legend() 

    figure.colorbar(img, ax=ax)

#load files
#songName = "song1/vocals.mp3"
#orig_song = "song1/popsong.mp3"

#audio = AudioSegment.from_mp3(songName)
#audio2 = AudioSegment.from_mp3(songName2)

#load songs
def load_song(songName):

    return librosa.load(songName)
    
def get_tempo(beats):
    #beat_tracker = BeatNet(1, mode="online")
    #downbeat_frames = beat_tracker.process(y)
    intervals = np.diff(beats)

    tempo = 60/np.mean(intervals)
    
    #print(tempo)
    return tempo

def get_beats_and_downbeats(y, sr):

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    #beat_tracker = BeatNet(1, mode="online", inference_model="PF", thread=False)
    #beats_np = beat_tracker.process(y)
    #print(beats_np)
    print(tempo)
    sf.write("test.wav",y,sr)
    act = RNNDownBeatProcessor()("test.wav")

    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=100, min_bpm = tempo * 0.85, max_bpm = tempo * 1.15)

    downbeats = dbn(act)

    
    beats = np.array([time for time, beat in downbeats])
    downbeats = np.array([time for time, beat in downbeats if beat == 1])
    print(60/np.mean(np.diff(beats)))
    os.remove("test.wav")
    #print(beats_fixed)
    #return beats
    return (beats, downbeats)

def get_beats_and_downbeats_beatnet(y, sr):
    beat_tracker = BeatNet(1, mode="offline", inference_model="DBN", thread=False)
    beats_np = beat_tracker.process(y)
    beats = np.array([time for time, beat in beats_np])
    downbeats = np.array([time for time, beat in beats_np if beat == 1])
    print(60/np.mean(np.diff(beats)))
    return (beats, downbeats)



#def get_downbeats(y, beats):
    #beat_tracker = BeatNet(1, mode="online")
    #downbeat_frames = beat_tracker.process(y)
    
    #return librosa.time_to_frames([time for time, downbeat in downbeat_frames if downbeat == 1.])

def get_clicks(beats, sr):
    return librosa.clicks(times=beats, sr=sr)

def samples_to_time(beat_list, sr):
    return librosa.samples_to_time(beat_list, sr=sr)

#make tempos match
def match_tempos(y, y2, tempo, tempo2, stem1 = None, stem2 = None):
    rate1 = 1
    rate2 = tempo/tempo2
    print("new tempos", tempo * rate1, tempo2 * rate2)
    #if stem1 and stem2:
    #    if stem1 == "vocals":
    #        if tempo < 0.6 * tempo2:
    #            tempo = tempo * 2
    #        elif tempo2 < 0.6 * tempo:
    #            tempo2 = tempo2 * 2
    #        rate1 = 1
    #        rate2 = tempo/tempo2
    #    elif stem2 == "vocals":
    #        if tempo < 0.66 * tempo2:
    #            tempo = tempo * 2
    #        elif tempo2 < 0.66 * tempo:
    #            tempo2 = tempo2 * 2
    #        rate1 = tempo2/tempo
    #        rate2 = 1
    #else:
    #    if tempo < 0.66 * tempo2:
    #        tempo = tempo * 2
    #    elif tempo2 < 0.66 * tempo:
    #        tempo2 = tempo2 * 2
    #    rate1 = (((tempo + tempo2)/2)/tempo)
    #    rate2 = (((tempo + tempo2)/2)/tempo2)
    
    return (librosa.effects.time_stretch(y, rate = rate1), librosa.effects.time_stretch(y2, rate = rate2), rate1, rate2)

def frames_to_samples(beat_list, sr):
    return librosa.frames_to_samples(beat_list)
 
def frames_to_times(frames, sr):
    return librosa.frames_to_time(frames, sr=sr)
 
def times_to_samples(times, sr):
    return librosa.time_to_samples(times, sr=sr)

   


#find stft
def stft(y, sr):
  
    return np.abs(librosa.stft(y))


def mfcc(y, sr, downbeats):
    #y_harmonic = librosa.effects.harmonic(y=y)
    downbeats = times_to_samples(downbeats, sr)
    #print(len(downbeats))
    #downbeats = downbeats[::4]
    #print(len(downbeats))
    mfcc_array = []
    #for i in range(0,len(downbeats) - 1):
        #print(i)
    start = downbeats[0]
    if len(downbeats) < 5:
        end = downbeats[-1]
        y_segment = y[start:end]
        #y_segment = librosa.effects.harmonic(y_segment)
        mfcc = librosa.feature.mfcc(y=y_segment, sr=sr,n_mfcc=13)
        avg_mfcc_for_bar = np.mean(mfcc, axis=1)
        mfcc_array.append(avg_mfcc_for_bar)
    else:
        for i in range(4,len(downbeats)):
            start = downbeats[i-4]
            end = downbeats[i]
            y_segment = y[start:end]
            #y_segment = librosa.effects.harmonic(y_segment)
            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr,n_mfcc=80)
            avg_mfcc_for_bar = np.mean(mfcc, axis=1)
            #print(avg_mfcc_for_bar.shape)
            mfcc_array.append(avg_mfcc_for_bar)
            
    mfcc_array = np.array(mfcc_array)
    
        #print(mfcc.shape)
        #print(mfcc)
        #avg_mfcc_for_bar = np.mean(mfcc, axis=1)
        #if i == 10:
            ##print(total_mfcc_for_bar)
            #print(total_mfcc_for_bar.shape)
        #mfcc_array.append(avg_mfcc_for_bar)
    #mfcc_array = np.array(mfcc_array)
    #print(mfcc_array)
    #print(mfcc_array)
    return mfcc_array

#find chromagram
def chroma(y, sr, downbeats):
    y_harmonic = librosa.effects.harmonic(y=y)
    downbeats = times_to_samples(downbeats, sr)
    chroma_array = []
    for i in range(0, len(downbeats) - 1):
        start = downbeats[i]
        end = downbeats[i+1]
        y_segment = y_harmonic[start:end]
        chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr)
        avg_chroma_for_bar = np.mean(chroma, axis = 1)
        chroma_array.append(avg_chroma_for_bar)
    chroma_array = np.array(chroma_array)
    return chroma_array

#find spectral centroid
def centroid(y, sr):
    return librosa.feature.spectral_centroid(y=y, sr=sr)


#find tonal centroids
def tonnetz(y, sr, chroma):
    return librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)

#find rms
def rms(y, sr):
    return librosa.feature.rms(y=y)

#find correlation coefficients
def corr_coef(data1, data2):
    return np.corrcoef(data1, data2)
    #corr_shape_x = corr.shape[0]
    #corr_shape_y = corr.shape[1]
    #print(corr_shape_x, corr_shape_y)
    #shows correlation between variables in the different arrays
    #return corr[int(corr_shape_x/2):corr_shape_x, 0:int(corr_shape_y/2)]
    #tonnetz_corr_split = tonnetz_corr[6:12, 0:6]

def corr_coef_split(data1, data2):
    #print(data1.shape, data2.shape)
    corr = np.corrcoef(data1, data2)
    corr_shape_x = corr.shape[0]
    corr_shape_y = corr.shape[1]
    #print(corr_shape_x, corr_shape_y)
    #shows correlation between variables in the different arrays
    return corr[int(corr_shape_x/2):corr_shape_x, 0:int(corr_shape_y/2)]
    #tonnetz_corr_split = tonnetz_corr[6:12, 0:6]

def display(matrix, feature=None):
    if feature == "chroma" or feature == "tonnetz":
        return librosa.display.specshow(matrix, y_axis=feature, x_axis=feature)
    else:
        return librosa.display.specshow(matrix)

def find_matrix_avg(matrix):
    
    return np.mean(matrix)

def segment_song(feature):
    return librosa.segment.agglomerative(feature, 6)

def trim(y):
    return librosa.effects.trim(y)

def bar_by_bar_corr_coef_mfcc(mfcc1, mfcc2):
    corrArray = []
    #print(mfcc1.shape)
    for i in range(mfcc1.shape[0]):
        #print(mfcc2[:,i])
        corr_bar = corr_coef_split(mfcc1[i:,], mfcc2[i:,])
        #print(corr_bar)
        #if i == 5:
            #print(mfcc1[:,i])
        #print(corr_bar[0][0])
        corrArray.append(corr_bar[0][0])
    #print(corrArray)
    #print(len(corrArray))
    return np.array(corrArray)

def bar_by_bar_corr_coef_chroma(chroma1, chroma2):
    corrArray = []
    for i in range(chroma1.shape[0]):
        corr_bar = corr_coef_split(chroma1[i], chroma2[i])
        #print(mfcc1[:,i])
        #print(corr_bar[0][0])
        corrArray.append(corr_bar[0][0])
    #print(corrArray)
    #print(len(corrArray))
    return np.array(corrArray)

def find_silence(y, frame_length):
    y_db = librosa.amplitude_to_db(np.abs(y))
    mean_db = -np.mean(y_db)
    min_db = -np.min(y_db)
    max_db = -np.max(y_db)
    #print(min_db - mean_db)
    top_db = (min_db - mean_db)/100 * min_db
    #   top_db = 0

    non_silent = librosa.effects.split(y, top_db = top_db, frame_length=frame_length, hop_length=int(frame_length/2))
    silent = []
    if len(non_silent) == 0 or max_db > 40:
        silent.append([0, len(y)-1])
    else:
        if non_silent[0][0] != 0:
            silent.append([0, non_silent[0][0]])
        for i in range(0, len(non_silent) - 1):
            silent.append([non_silent[i][1], non_silent[i+1][0]])
        if non_silent[-1][1] < len(y) - 1:
            silent.append([non_silent[-1][1], len(y)-1])
    
    return np.array(silent)
    
def find_silent_downbeat_ranges(sr, silence, downbeats):
    downbeats = times_to_samples(downbeats, sr)
    silent_downbeats = []
    for interval in silence:
        #print(interval)
        try:
            pot_silence_start = [[downbeat for downbeat in downbeats if downbeat <= interval[0]][-1], [downbeat for downbeat in downbeats if downbeat >= interval[0]][0]]
        except:
            pot_silence_start = [0, downbeats[-1]]
        try:
            pot_silence_end = [[downbeat for downbeat in downbeats if downbeat <= interval[1]][-1], [downbeat for downbeat in downbeats if downbeat >= interval[1]][0]]
        except:
            pot_silence_end = [0, downbeats[-1]]
        #print(pot_silence_start, pot_silence_end)
        silence_start = pot_silence_start[np.argmin(np.array(abs(pot_silence_start - interval[0])))]
        silence_end = pot_silence_end[np.argmin(np.array(abs(pot_silence_end - interval[1])))]
        if silence_start != silence_end:
            #print(samples_to_time(silence_start), samples_to_time(silence_end))
            silent_downbeats.append([silence_start, silence_end])
    return silent_downbeats


def find_key(y,sr):
    y_harmonic = librosa.effects.harmonic(y=y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=12)
    pitch_activations = []

    for i in range(12):
        pitch_activations.append(np.sum(chroma[i]))

    pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    keyfreqs = {pitches[i]: pitch_activations[i] for i in range(12)}

    keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

    #krumhansl-schmuckler key-finding algorithm
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    min_key_corrs = []
    maj_key_corrs = []

    for i in range(12):
        key_test = [keyfreqs.get(pitches[(i + m)%12]) for m in range(12)]
        # correlation coefficients (strengths of correlation for each key)
        maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1,0], 3))
        min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1,0], 3))
        # names of all major and minor keys
    maj_key_dict = {keys[i]: maj_key_corrs[i] for i in range(12)}
    min_key_dict = {keys[i+12]: min_key_corrs[i] for i in range(12)}
        
        # this attribute represents the key determined by the algorithm
    
    bestcorr_maj = max(maj_key_dict.values())
    bestcorr_min = max(min_key_dict.values())
    if bestcorr_maj > bestcorr_min:
        bestkey = [key for key, value in maj_key_dict.items() if value==bestcorr_maj][0]
        bestcorr = bestcorr_maj
    else:
        bestkey = [key for key, value in min_key_dict.items() if value==bestcorr_min][0]
        bestcorr = bestcorr_min
    #print(maj_key_dict, min_key_dict)
        
        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
    #altkey = None
    #altbestcorr = None

    #for key, corr in key_dict.items():
    #    if corr > bestcorr*0.9 and corr != bestcorr:
    #        altkey = key
    #        altbestcorr = corr
    
    key_index = 0
    if "minor" in bestkey:
        for key in min_key_dict.keys():
            if min_key_dict[key] == bestcorr:
                break
            key_index += 1
    else:
        for key in maj_key_dict.keys():
            if maj_key_dict[key] == bestcorr:
                break
            key_index += 1
    
    major = 0

    if "minor" not in bestkey:
        major = 1

    return key_index, major

def shift_pitch(y, sr, key_y, key2):
    n_steps = key2 - key_y
    if n_steps > 6:
        n_steps = -12 + n_steps
    elif n_steps < -6:
        n_steps = 12 + n_steps
    #print(n_steps)
    if key_y == key2:
        return y
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def extract_chords(y, sr, fps=10):
    sf.write("temp1.wav",y,sr)
    fps = fps
    #print(fps)

    dcp = DeepChromaProcessor(fps=fps)
    decoder = DeepChromaChordRecognitionProcessor(fps=fps)
    chroma = dcp("temp1.wav")

    #hop_len = dcp.processors[1].hop_size

    targets = decoder(chroma)
    ann = encode_chords(targets)
    ann = merge_chords(ann)
    reduced = reduce_to_triads(ann["chord"])


    out = np.empty(len(reduced), dtype=ann.dtype)
    out["start"] = ann["start"] 
    out["end"] = ann["end"] 
    out["chord"] = reduced

    os.remove("temp1.wav")
    return out

def chord_matrix(ann, fps=10):
    T = int(np.ceil(ann['end'][-1] * fps))
    M = np.zeros((12,T), dtype=float)
    for segment in ann:
        r = int(segment["chord"]["root"])
        t0 = int(segment["start"] * fps)
        t1 = int(segment["end"] * fps)
        M[r, t0:t1] = 1.0
    return M

def chord_similarity(M1, M2, fps=10):
    F1 = M1.T
    F2 = M2.T

    C = cdist(F1, F2, metric="euclidean")

    D, wp = librosa.sequence.dtw(C=C, backtrack=True)
    dist = D[-1,-1]
    sim = 1.0 - dist / len(wp)
    return sim