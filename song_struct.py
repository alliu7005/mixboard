import matplotlib.pyplot as plt
import matplotlib.transforms as mpt
import matplotlib.cm as cm
import matplotlib.colors as colors
import librosa
import librosa_test
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import compress_dynamic_range
import numpy as np
from BeatNet.BeatNet import BeatNet
from ssmnet_ISMIR2023.ssmnet import core
import yaml
import os
import demucs.api
import soundfile as sf
import copy
import demucs.separate
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt, sosfiltfilt


class Song_Struct:
    def __init__(self, orig_y, sr, name, v_y=None, o_y=None, b_y=None, d_y=None, tempo=None, key=None, major=None, bounds=None, downbeats=None, beats=None, take_fields=True):

        self.name = name
        self.bounds = []

        self.orig_y = orig_y
        self.sr = sr

        if v_y is None or o_y is None or b_y is None or d_y is None:
            separator = demucs.api.Separator(model="htdemucs_ft")
            separator.update_parameter(progress=True)
            self.orig_y = librosa.resample(self.orig_y, orig_sr=self.sr, target_sr = separator.samplerate)
            self.sr = separator.samplerate
            sf.write("orig_song.mp3", self.orig_y, separator.samplerate)
            demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "htdemucs", "orig_song.mp3"])
            origin, separated = separator.separate_audio_file("./separated/htdemucs/orig_song/no_vocals.mp3")
            self.separated = separated
            os.remove("orig_song.mp3")

            self.v_y, _ = librosa.load("./separated/htdemucs/orig_song/vocals.mp3", sr=separator.samplerate)
            self.d_y = librosa.to_mono(separated["drums"].cpu().numpy())
            self.b_y = librosa.to_mono(separated["bass"].cpu().numpy())
            self.o_y = librosa.to_mono(separated["other"].cpu().numpy())
            
            

            self.orig_y = librosa.resample(self.orig_y, orig_sr=separator.samplerate, target_sr = sr)
            self.v_y = librosa.resample(self.v_y, orig_sr=separator.samplerate, target_sr = sr)
            self.d_y = librosa.resample(self.d_y, orig_sr=separator.samplerate, target_sr = sr)
            self.b_y = librosa.resample(self.b_y, orig_sr=separator.samplerate, target_sr = sr)
            self.o_y = librosa.resample(self.o_y, orig_sr=separator.samplerate, target_sr = sr)
            self.sr = sr

            #sf.write("o_y.wav", self.o_y, self.sr)
        else:
            self.v_y = v_y
            self.d_y = d_y
            self.b_y = b_y
            self.o_y = o_y

        #Order of EVERYTHING is 1. Vocals 2. Other 3. Bass 4. Drums

        if tempo is None or key is None or major is None or bounds is None or downbeats is None or beats is None:
        
            self.beats, self.downbeats = librosa_test.get_beats_and_downbeats(self.orig_y, self.sr)
            self.tempo = librosa_test.get_tempo(self.beats)
            self.key, self.major = librosa_test.find_key(orig_y,self.sr)
            self.set_bounds()
        else:
            self.beats = np.array(beats)
            self.downbeats = np.array(downbeats)
            self.tempo = tempo
            self.key = key
            self.major = major
            self.bounds = bounds

        self.song_len = len(self.downbeats)
        self.frame_len = int(np.mean(np.diff(librosa_test.times_to_samples(self.downbeats, self.sr))))

        self.vocals = Stem(self, self.v_y, self.sr, "vocals", self.tempo, self.key, self.major, self.bounds, self.downbeats, self.beats, take_fields=take_fields)
        self.other = Stem(self, self.o_y, self.sr, "other", self.tempo, self.key,  self.major, self.bounds, self.downbeats, self.beats, take_fields=take_fields)
        self.drums = Stem(self, self.d_y, self.sr, "drums", self.tempo, self.key,  self.major, self.bounds, self.downbeats, self.beats, take_fields=take_fields)
        self.bass = Stem(self, self.b_y, self.sr, "bass", self.tempo, self.key,  self.major, self.bounds, self.downbeats, self.beats, take_fields=take_fields)

        self.stems = [self.vocals, self.other, self.bass, self.drums]

    def set_bounds(self):
        config_path = os.path.join("ssmnet_ISMIR2023", "ssmnet", "weights_deploy","config_example.yaml")
        with open(config_path, "r", encoding="utf-8") as fid:
            config_d = yaml.safe_load(fid)
        ssmnet_deploy = core.SsmNetDeploy(config_d)

        feat_3m, time_sec_v = ssmnet_deploy.m_get_features(self.orig_y, self.sr)
        hat_ssm_np, hat_novelty_np = ssmnet_deploy.m_get_ssm_novelty(feat_3m)
        hat_boundary_sec_v, hat_boundary_frame_v = ssmnet_deploy.m_get_boundaries(hat_novelty_np, time_sec_v)
        self.bounds = []
        for bound in hat_boundary_sec_v:
            if bound != 0:
                #print(bound)
                
                try:
                    pot_bar = int(np.where(self.downbeats > bound)[0][0])
                except:
                    pot_bar = 0
                try:
                    pot_bar2 = int(np.where(self.downbeats < bound)[0][-1])
                except:
                    pot_bar2 = 0
                diff1 = abs(self.downbeats[pot_bar]-bound)
                diff2 = abs(self.downbeats[pot_bar2] - bound)
                print(pot_bar, diff1, pot_bar2, diff2)
                if diff1 < diff2:
                    self.bounds.append(pot_bar)
                else:
                    self.bounds.append(pot_bar2)
                #self.bounds.append(pot_bar)

        #self.bounds = [8, 24, 32, 48, 64, 72]
    
    def set_silence(self):
        #self.bass.set_silence()
        for stem in self.stems:
            stem.set_silence()

    #Must set silence FIRST
    def set_active(self):
        for stem in self.stems:
            stem.set_active()

    def get_chroma(self):
        for stem in self.stems:
            stem.get_chroma()
        #print(self.chroma)
    
    def get_mfcc(self):
        for stem in self.stems:
            stem.get_mfcc()

    def get_stft(self):
        for stem in self.stems:
            stem.get_stft()

    def get_rms(self):
        for stem in self.stems:
            stem.get_rms()

    #Need Chroma First
    def get_tonnetz(self):
        for stem in self.stems:
            stem.get_tonnetz()

    def get_specgram(self):
        for stem in self.stems:
            stem.get_specgram()


    def to_string(self):
        print("Active", [stem.active for stem in self.stems])

    #need active, specgram FIRST
    def visualize(self):
        fig, ax = plt.subplots()
        active_tracks = [stem.active for stem in self.stems]
        #tracks = [self.active[name] for name in self.names]
        

        names = [stem.name for stem in self.stems]
        for i in range(len(self.stems)):
        #for i in range(len(self.names)):
            y_position = i * 10  
            
            for j in range(len(active_tracks[i])):
                specgram = self.stems[i].specgram[j]
                #specgram = self.specgram[names[i]][j]
        
                s_db = librosa.power_to_db(specgram, ref=np.max)
                ax.imshow(s_db, aspect='auto', extent=[active_tracks[i][j][0], active_tracks[i][j][1],y_position,y_position + 8], cmap='viridis')


        ax.set_xlabel("Bars")
        ax.set_yticks([i * 10 + 4 for i in range(len(active_tracks))],names)
        max_y = (len(active_tracks) - 1) * 10 + 8
        ax.set_ylim(max_y, 0)
 
        ax.set_xlim(0, self.song_len-1)
        ax.set_title("Track Activity Visualization " + self.name)

        norm = colors.Normalize(vmin=s_db.min(), vmax=s_db.max())
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)

        cbar = fig.colorbar(sm, ax=ax, label="dB")
        cbar.ax.invert_yaxis()

        for bound in self.bounds:
            plt.axvline(x=bound)

        plt.tight_layout()
        plt.show()

    def play(self):
        sf.write("vocals.wav", self.vocals.y, self.sr)
        sf.write("other.wav", self.other.y, self.sr)
        sf.write("bass.wav", self.bass.y, self.sr)
        sf.write("drums.wav", self.drums.y, self.sr)
        vocal_audio = AudioSegment.from_wav("vocals.wav")
        other_audio = AudioSegment.from_wav("other.wav")
        bass_audio = AudioSegment.from_wav("bass.wav")
        drums_audio = AudioSegment.from_wav("drums.wav")

        os.remove("vocals.wav")
        os.remove("other.wav")
        os.remove("bass.wav")
        os.remove("drums.wav")

        vocal_audio.overlay(other_audio)
        vocal_audio.overlay(bass_audio)
        vocal_audio.overlay(drums_audio)
        play(vocal_audio)
    
    def to_dict(self):
        return {"name":self.name,
                "sr": self.sr,
                "tempo":self.tempo,
                "key":self.key,
                "major":self.major,
                "bounds":self.bounds,
                "downbeats":self.downbeats,
                "beats":self.beats,
                "y":self.o_y}


class Stem:
    def __init__(self, init_song, y, sr, name, tempo, key, major, bounds, downbeats, beats, silence = None, active = None, chroma = None, mfcc = None, stft = None, rms = None, tonnetz = None, specgram = None, take_fields=True):
        self.init_song = init_song
        self.y = y
        self.sr = sr
        self.name = name
        self.tempo = tempo
        self.key = key
        self.major = major
        self.bounds = bounds
        self.downbeats = downbeats
        self.beats = beats
        self.cluster = None

        self.frame_len = 0
        self.set_frame_len()
        self.song_len = len(self.downbeats)

        if take_fields:
            if silence is None or active is None or chroma is None or mfcc is None or stft is None or rms is None or tonnetz is None or specgram is None:
                self.silence = []
                self.active = []
                self.chroma = []
                self.mfcc = []
                self.stft = []
                self.rms = []
                self.tonnetz = []
                self.specgram = []

                print("getting silence")
                self.set_silence()
                print("getting active")
                self.set_active()
                #print("getting chroma")
                #self.get_chroma()
                #print("getting mfcc")
                #self.get_mfcc()
                #print("getting stft")
                #self.get_stft()
                #print("getting rms")
                #self.get_rms()
                #print("getting tonnetz")
                #self.get_tonnetz()
                print("getting specgram")
                self.get_specgram()
            else:
                self.silence = silence
                self.active = active
                self.chroma = chroma
                self.mfcc = mfcc
                self.rms = rms
                self.stft = stft
                self.tonnetz = tonnetz
                self.specgram = specgram

    def set_frame_len(self):
        self.frame_len = int(np.mean(np.diff(librosa_test.times_to_samples(self.downbeats, self.sr))))

    def set_silence(self):
        #print(self.name)
        self.set_frame_len()
        silence = librosa_test.find_silence(self.y, int(self.frame_len/8))
        if len(silence) != 0:
            silence_downbeats = librosa_test.samples_to_time(librosa_test.find_silent_downbeat_ranges(self.sr, silence, self.downbeats), self.sr)
            for j in range(len(silence_downbeats)):
                
                time_interval = np.around(silence_downbeats[j], 2)
                silence_start = [downbeat for downbeat in self.downbeats if downbeat >= time_interval[0]][0]

                try:
                    pot_silence_end = [[downbeat for downbeat in self.downbeats if downbeat >= time_interval[1]][0]]
                except:
                    pot_silence_end = [[downbeat for downbeat in self.downbeats if downbeat <= time_interval[1]][-1]]
                
                downbeats_of_interval = np.array([silence_start, pot_silence_end])
                
                
                start_bar = int(np.where(self.downbeats == downbeats_of_interval[0])[0][0])
                end_bar = int(np.where(self.downbeats == downbeats_of_interval[-1])[0][0])
                silence_downbeats[j] = [start_bar, end_bar]
            #print(self.name + "downbeats",silence_downbeats)

            for i in range(len(silence_downbeats) - 1):
                if silence_downbeats[i][1] == silence_downbeats[i+1][0]:
                    silence_downbeats[i+1] = [silence_downbeats[i][0],silence_downbeats[i+1][1]]
                    silence_downbeats[i] = 0

            self.silence = []
            for i in range(len(silence_downbeats)):
                if silence_downbeats[i][1] != 0:
                    self.silence.append(silence_downbeats[i])
                    
            self.silence = np.array(self.silence)
        else:
            self.silence = np.array([])

        #Must set silence FIRST
    def set_active(self):
        active = []
        if len(self.silence) != 0:
            if self.silence[0][0] > 0:
                active.append((0, self.silence[0][0]))
            for j in range(len(self.silence) - 1):
                silence_end = self.silence[j][1]
                silence_start = self.silence[j+1][0]
                active.append((silence_end, silence_start))
            if self.silence[-1][1] != self.song_len - 1:
                active.append([self.silence[-1][1], self.song_len - 1])
            self.active = np.array(active)
        else:
            self.active= np.array([[0, self.song_len-1]])
            

    def force_silence(self, silence):
        if len(silence) != 0:
            for i in range(len(silence)):
                silent_downbeats = (self.downbeats[int(silence[i][0])], self.downbeats[int(silence[i][1])])
                #print(silent_downbeats)
                interval = librosa_test.times_to_samples(silent_downbeats, self.sr)
                #print(interval)
                self.y[interval[0]:interval[1]] = 0
                #print(self.y[interval[0]:interval[1]], self.name)
        self.set_silence()
        self.set_active()

    def get_chroma(self):
        downbeats = librosa_test.times_to_samples(self.downbeats, self.sr)
        active = self.active
        chroma_array = []
        #for j in range(len(active)):
        #    bound1, bound2 = active[j]
        #    bound1 = int(bound1)
        #    bound2 = int(bound2)
        #    bound1 = downbeats[bound1]
        #    bound2 = downbeats[bound2]
            #chroma_array.append(librosa_test.chroma(self.y[bound1:bound2],self.sr, self.downbeats))
        self.chroma = librosa.feature.chroma_cqt(y=self.y,sr=self.sr)
        #print(self.chroma)

    def get_mfcc(self):
        downbeats = librosa_test.times_to_samples(self.downbeats, self.sr)
        active = self.active
        mfcc_array = []
        #for j in range(len(active)):
        #    bound1, bound2 = active[j]
        #    bound1 = int(bound1)
        #    bound2 = int(bound2)
        #    bound1 = downbeats[bound1]
        #    bound2 = downbeats[bound2]
            #mfcc_array.append(librosa_test.mfcc(self.y[bound1:bound2],self.sr, self.downbeats))
        self.mfcc = librosa.feature.mfcc(y=self.y,sr=self.sr, n_mfcc=13)
        #print(self.mfcc.shape)

    def get_stft(self):
        downbeats = librosa_test.times_to_samples(self.downbeats, self.sr)
        active = self.active
        stft_array = []
        #for j in range(len(active)):
        #    bound1, bound2 = active[j]
        #    bound1 = int(bound1)
        #    bound2 = int(bound2)
        #    bound1 = downbeats[bound1]
        #    bound2 = downbeats[bound2]
        #    stft_array.append()
        self.stft = librosa_test.stft(self.y,self.sr)

    def get_rms(self):
        downbeats = librosa_test.times_to_samples(self.downbeats, self.sr)
        active = self.active
        rms_array = []
        #for j in range(len(active)):
        #    bound1, bound2 = active[j]
        #    bound1 = int(bound1)
        #    bound2 = int(bound2)
        #    bound1 = downbeats[bound1]
        #    bound2 = downbeats[bound2]
        #    rms_array.append(librosa_test.rms(self.y[bound1:bound2],self.sr))
        self.rms = librosa_test.rms(self.y,self.sr)

        #Need Chroma First
    def get_tonnetz(self):
        downbeats = librosa_test.times_to_samples(self.downbeats, self.sr)
        active = self.active
        tonnetz_array = []
        #for j in range(len(active)):
        #    bound1, bound2 = active[j]
        #    bound1 = int(bound1)
        #    bound2 = int(bound2)
        #    bound1 = downbeats[bound1]
        #    bound2 = downbeats[bound2]
            #tonnetz_array.append(librosa_test.tonnetz(self.y[bound1:bound2],self.sr, self.chroma[j]))
        self.tonnetz = librosa_test.tonnetz(self.y,self.sr, self.chroma)

    def get_specgram(self):
        downbeats = librosa_test.times_to_samples(self.downbeats, self.sr)
        active = self.active
        specgram_array = []
        for j in range(len(active)):
            bound1, bound2 = active[j]
            bound1 = int(bound1)
            bound2 = int(bound2)
            #print(bound1, bound2)
            bound1 = downbeats[bound1]
            bound2 = downbeats[bound2]
            specgram_array.append(librosa.feature.melspectrogram(y=self.y[bound1:bound2],sr=self.sr))
        self.specgram = specgram_array

    def change_tempo(self, desired_beats, desired_downbeats):

        new_audio_segments = []
        new_downbeats = [desired_downbeats[0]]
        if(self.name == "other"):
            print(self.beats)

        for i in range(len(self.downbeats) - 1):
            start_sample = librosa.time_to_samples(self.downbeats[i], sr=self.sr)
            end_sample = librosa.time_to_samples(self.downbeats[i+1], sr=self.sr)
            segment = self.y[start_sample:end_sample]
            original_duration = self.downbeats[i+1] - self.downbeats[i]
            target_duration = desired_downbeats[i+1] - desired_downbeats[i]
            rate = original_duration/target_duration
            
            stretched = librosa.effects.time_stretch(segment, rate = rate)
            new_audio_segments.append(stretched)
            new_downbeats.append(desired_downbeats[i+1])

        start_sample = librosa.time_to_samples(self.downbeats[-1], sr=self.sr)
        end_sample = len(self.y) - 1
        end_time = np.around(librosa.samples_to_time(end_sample, sr=self.sr),2)
        segment = self.y[start_sample:end_sample]
        original_duration = self.downbeats[-1] - end_time
        target_duration = desired_downbeats[-1] - end_time
        #print(original_duration, target_duration, rate)
        if original_duration<=0 or target_duration<=0:
            new_audio_segments.append(segment)
        elif np.around(original_duration/target_duration,2) == 0:
            new_audio_segments.append(segment)
        else:
            #print(end_time)
            rate = original_duration/target_duration
            #print(self.downbeats, desired_downbeats)
            #print(original_duration, target_duration, rate)
            print(rate)
            stretched = librosa.effects.time_stretch(segment, rate = rate)
            new_audio_segments.append(stretched)

        self.y = np.concatenate(new_audio_segments)
        self.beats = desired_beats
        self.downbeats = np.array(new_downbeats)
        self.tempo = 60/np.mean(np.diff(self.beats))

    def set_key(self, key):
        self.y = librosa_test.shift_pitch(self.y, self.sr, self.key, key)
        self.key = key
    
    def to_dict(self):
        return {"name":self.name,
                "songname":self.init_song.name,
                "sr": self.sr,
                "tempo":self.tempo,
                "key":self.key,
                "major":self.major,
                "bounds":self.bounds,
                "downbeats":self.downbeats,
                "beats":self.beats,
                "y":self.y,
                "silent":self.silence,
                "active":self.active,
                "specgram":self.specgram,
                "cluster":self.cluster}

class Mashup:

    def __init__(self, init_song:Song_Struct, vocals:Stem, other:Stem, bass:Stem, drums:Stem):
        self.init_song = init_song
        self.vocals = copy.deepcopy(vocals)
        self.other = copy.deepcopy(other)
        self.bass = copy.deepcopy(bass)
        self.drums = copy.deepcopy(drums)
        self.stems = [self.vocals, self.other, self.bass, self.drums]
        self.section = 2
        self.sr = vocals.sr
        self.key = self.vocals.key

        vocal_bars = librosa_test.times_to_samples([self.vocals.downbeats[self.vocals.bounds[self.section]], self.vocals.downbeats[self.vocals.bounds[self.section] + 16]], self.sr)
        other_bars = librosa_test.times_to_samples([self.other.downbeats[self.other.bounds[self.section]], self.other.downbeats[self.other.bounds[self.section] + 16]], self.sr)
        bass_bars = librosa_test.times_to_samples([self.bass.downbeats[self.bass.bounds[self.section]], self.bass.downbeats[self.bass.bounds[self.section] + 16]], self.sr)
        drums_bars = librosa_test.times_to_samples([self.drums.downbeats[self.drums.bounds[self.section]], self.drums.downbeats[self.drums.bounds[self.section] + 16]], self.sr)

        self.vocals.y = self.vocals.y[vocal_bars[0]: vocal_bars[1]]
        self.bass.y = self.bass.y[bass_bars[0]: bass_bars[1]]
        self.drums.y = self.drums.y[drums_bars[0]: drums_bars[1]]
        self.other.y = self.other.y[other_bars[0]: other_bars[1]]

        sf.write("vocals.wav", self.vocals.y, self.sr, subtype='PCM_16')
        sf.write("other.wav", self.other.y, self.sr, subtype='PCM_16')
        sf.write("bass.wav", self.bass.y, self.sr, subtype='PCM_16')
        sf.write("drums.wav", self.drums.y, self.sr, subtype='PCM_16')

        

        self.vocals.downbeats = self.vocals.downbeats[self.vocals.bounds[self.section]:self.vocals.bounds[self.section]+17]
        self.other.downbeats = self.other.downbeats[self.other.bounds[self.section]:self.other.bounds[self.section]+17]
        self.bass.downbeats = self.bass.downbeats[self.bass.bounds[self.section]:self.bass.bounds[self.section]+17]
        self.drums.downbeats = self.drums.downbeats[self.drums.bounds[self.section]:self.drums.bounds[self.section]+17]

        #print(self.other.beats)

        self.vocals.beats = self.vocals.beats[np.where(self.vocals.beats == self.vocals.downbeats[0])[0][0]:np.where(self.vocals.beats == self.vocals.downbeats[-1])[0][0]+1]
        self.other.beats = self.other.beats[np.where(self.other.beats == self.other.downbeats[0])[0][0]:np.where(self.other.beats == self.other.downbeats[-1])[0][0]+1]
        self.bass.beats = self.bass.beats[np.where(self.bass.beats == self.bass.downbeats[0])[0][0]:np.where(self.bass.beats == self.bass.downbeats[-1])[0][0]+1]
        self.drums.beats = self.drums.beats[np.where(self.drums.beats == self.drums.downbeats[0])[0][0]:np.where(self.drums.beats == self.drums.downbeats[-1])[0][0]+1]

        #print(self.other.beats)
        
        self.vocals.downbeats -= self.vocals.downbeats[0]
        self.other.downbeats -= self.other.downbeats[0]
        self.bass.downbeats -= self.bass.downbeats[0]
        self.drums.downbeats -= self.drums.downbeats[0]

        self.vocals.beats -= self.vocals.beats[0]
        self.other.beats -= self.other.beats[0]
        self.bass.beats -= self.bass.beats[0]
        self.drums.beats -= self.drums.beats[0]

        self.song_len = len(self.vocals.downbeats)
        self.set_songlen()
       
        tempo_before_conversion = np.mean(np.diff(self.vocals.beats))
        self.tempo = 60/tempo_before_conversion
        #print(self.tempo)
        #self.vocals.change_tempo(self.vocals.beats)
        #desired_beats = np.around([i * tempo_before_conversion for i in range(len(self.vocals.beats))],2)
        #desired_downbeats = np.around([i * np.mean(np.diff(self.vocals.downbeats)) for i in range(len(self.vocals.downbeats))],2)
        #print(self.bass.beats, self.bass.downbeats)
        #print(len(self.bass.y))
        #self.vocals.change_tempo(desired_beats, desired_downbeats)
        self.bass.change_tempo(self.vocals.beats, self.vocals.downbeats)
        #print(self.bass.beats, self.bass.downbeats)
        #print(len(self.bass.y))
        self.other.change_tempo(self.vocals.beats, self.vocals.downbeats)
        self.drums.change_tempo(self.vocals.beats, self.vocals.downbeats)
        
        for stem in self.stems:
            stem.set_frame_len()

        
        self.set_silence()
        self.set_active()


        self.init_song_silence = self.match_silence()
        self.force_silence(self.init_song_silence)
        self.set_keys()
        self.get_specgram()


    def match_silence(self):
        init_song_silence = {}

        for stem in self.init_song.stems:
            silence_array = []
            stem_silence = stem.silence - self.init_song.bounds[self.section]
            #print(stem_silence)
            for interval in stem_silence:
                if interval[0] <= self.song_len - 1 and interval[1] > 0:
                    silence_array.append((max(interval[0],0), min(self.song_len - 1,interval[1])))
            init_song_silence[stem.name] = silence_array
        
        return init_song_silence

    def set_songlen(self):
        for stem in self.stems:
            stem.song_len = self.song_len

    def set_keys(self):
        for stem in self.stems:
            if stem.key != self.key and stem.name != "drums":
                stem.set_key(self.key)

    def set_silence(self):
        for stem in self.stems:

            stem.set_silence()

    #Must set silence FIRST
    def set_active(self):
        for stem in self.stems:
            stem.set_active()

    def get_specgram(self):
        for stem in self.stems:
            stem.get_specgram()
    
    def force_silence(self, silence):
        for stem in self.stems:
            if stem.name != "vocals":
                #print(silence[stem.name])
                stem.force_silence(silence[stem.name])


    def visualize(self, path):
        fig, ax = plt.subplots()
        active_tracks = [stem.active for stem in self.stems]
        #tracks = [self.active[name] for name in self.names]

        names = [stem.name for stem in self.stems]
        for i in range(len(self.stems)):
        #for i in range(len(self.names)):
            y_position = i * 10  
            
            for j in range(len(active_tracks[i])):
                specgram = self.stems[i].specgram[j]
                #specgram = self.specgram[names[i]][j]
        
                s_db = librosa.power_to_db(specgram, ref=np.max)
                ax.imshow(s_db, aspect='auto', extent=[active_tracks[i][j][0], active_tracks[i][j][1],y_position,y_position + 8], cmap='viridis')


        ax.set_xlabel("Bars")
        ax.set_yticks([i * 10 + 4 for i in range(len(active_tracks))],names)
        max_y = (len(active_tracks) - 1) * 10 + 8
        ax.set_ylim(max_y, 0)
 
        ax.set_xlim(0, self.song_len-1)
        ax.set_title("Track Activity Visualization " + "Mashup")

        norm = colors.Normalize(vmin=s_db.min(), vmax=s_db.max())
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)

        cbar = fig.colorbar(sm, ax=ax, label="dB")
        cbar.ax.invert_yaxis()

        #for bound in self.bounds:
        #    plt.axvline(x=bound)

        plt.tight_layout()
        plt.savefig(os.path.join(path, "vis.png"))
        plt.close()
    
    def norm_loudness(self, target = -14.0):
        print("normalizing loudness")
        meter = pyln.Meter(self.vocals.sr)
        v_loudness = meter.integrated_loudness(self.vocals.y)
        o_loudness = meter.integrated_loudness(self.other.y)
        b_loudness = meter.integrated_loudness(self.bass.y)
        d_loudness = meter.integrated_loudness(self.drums.y)



        self.vocals.y = np.clip(pyln.normalize.loudness(self.vocals.y, v_loudness, target), -1.0, 1.0)
        self.other.y = np.clip(pyln.normalize.loudness(self.other.y, o_loudness, target), -1.0, 1.0)
        self.bass.y = np.clip(pyln.normalize.loudness(self.bass.y, b_loudness, -12.0), -1.0, 1.0)
        self.drums.y = np.clip(pyln.normalize.loudness(self.drums.y, d_loudness, target), -1.0, 1.0)

    def gain_stage(self):
        peak = np.max(np.abs(self.drums.y))
        self.drums.y *= (10**(-6/20)) / (peak + 1e-6)
        self.drums.y = np.tanh(self.drums.y * 1.1) / 1.1

    def match_rms(self):
        print("matching rms")
        ref_rms = np.sqrt(np.mean(self.vocals.y**2))
        other_rms = np.sqrt(np.mean(self.other.y**2))
        bass_rms = np.sqrt(np.mean(self.bass.y**2))
        drums_rms = np.sqrt(np.mean(self.drums.y**2))
        self.other.y = self.other.y * (ref_rms) / (other_rms + 1e-6)
        self.bass.y = self.bass.y * (ref_rms) / (bass_rms + 1e-6)
        self.drums.y = self.drums.y * (ref_rms) / (drums_rms + 1e-6)
    
    def hp_filter(self):
        print("filtering")
        sos_v = butter(4, 80, 'highpass',fs=self.vocals.sr, output='sos')
        sos_b = butter(4, 30, 'highpass',fs=self.vocals.sr, output='sos')
        self.vocals.y = sosfiltfilt(sos_v, self.vocals.y)
        self.other.y = sosfiltfilt(sos_v, self.other.y)
        self.bass.y = sosfiltfilt(sos_b, self.bass.y)

    def noise_gate(self, threshold=0.02, attack=0.01, release=0.1, floor_gain=0.00, knee=0.01):
        env=np.abs(self.other.y)
        gain_raw = np.ones_like(env)
        low = threshold - knee
        high = threshold

        mask_low = env < low
        gain_raw[mask_low]=floor_gain
        mask_mid = (env >= low) & (env < high)
        gain_raw[mask_mid] = floor_gain + (env[mask_mid] - low) / knee * (1 - floor_gain)

        #gain_raw = np.clip((env - threshold) / (1 - threshold),0,1)
        sos = butter(2, 1.0/release, btype="low", fs=self.sr, output="sos")
        gain = sosfiltfilt(sos, gain_raw)
        self.other.y = self.other.y * gain
    
    def lp_filter(self):
        sos_o = butter(2, 11024.0, btype="lowpass", fs=self.sr, output="sos")
        self.other.y = sosfiltfilt(sos_o, self.other.y)

    def bell_cut(self, center, q=1.0, gain_db=-6.0):
        print("bell_cut")
        A = 10**(gain_db/40)
        w0 = 2*np.pi*center/self.sr
        alpha = np.sin(w0)/(2*q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha/A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha/A
        sos = np.array([[b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]])
        self.vocals.y = sosfiltfilt(sos, self.vocals.y)

    def bandstop(self, center, q=4, depth_db=-4):
        print("bandstop")
        bw = center / q
        low, high = center - bw/2, center + bw/2
        sos = butter(4, [low, high], btype='bandstop', fs = self.sr, output='sos')
        self.other.y = sosfiltfilt(sos, self.other.y)
        self.other.y = np.clip(self.other.y, -1.0, 1.0)
        
    
    def compress(self, segment, threshold=-20.0, ratio=4.0):
        print("compressing")
        return compress_dynamic_range(segment, threshold=threshold, ratio=ratio, attack=5.0, release=50.0)
    
    def boost_bass(self, cutoff = 200.0, order = 2, gain_db = 6.0):
        sos = butter(order, cutoff, btype='lowpass', fs=self.sr, output="sos")
        low = sosfiltfilt(sos, self.bass.y)
        gain = 10 ** (gain_db / 20.0)
        self.bass.y = self.bass.y + (gain - 1) * low
        self.bass.y = np.clip(self.bass.y, -1.0, 1.0)

    def boost_other(self, gain_db, peak_target=0.95):
        gain_lin = 10**(gain_db/20.0)
        self.other.y = self.other.y * gain_db
        peak = np.max(np.abs(self.other.y))
        self.other.y = self.other.y * (peak_target/peak)

        meter = pyln.Meter(self.sr)
        self.other.y = pyln.normalize.peak(self.other.y, -0.5)
    
    def auto_mix(self):
        #sf.write("other2.wav", self.other.y, self.sr, subtype='PCM_16')
        #sf.write("other2.wav", self.other.y, self.sr, subtype='PCM_16')
        #sf.write("bass2.wav", self.bass.y, self.sr, subtype='PCM_16')
        self.norm_loudness(target=-14.0)
        self.match_rms()
        self.gain_stage()
        self.noise_gate(threshold=0.02, attack=0.005, release=0.1)
        self.hp_filter()
        self.lp_filter()
        #self.bandstop(300, q=4.0)
        self.bell_cut(center=100, q=1.0, gain_db=-6.0)
        self.boost_bass(cutoff=200, order=2, gain_db = 2.0)

        #self.norm_loudness(target=-14.0)
        self.boost_other(gain_db=3.0, peak_target=0.85)

        silence = {"other": self.other.silence,
                   "bass": self.bass.silence,
                   "drums": self.drums.silence}
        self.force_silence(silence)


        self.other.get_specgram()
        self.vocals.get_specgram()
        self.drums.get_specgram()
        self.bass.get_specgram()

        
        sf.write("vocals.wav", self.vocals.y, self.sr, subtype='PCM_16')
        sf.write("other.wav", self.other.y, self.sr, subtype='PCM_16')
        sf.write("bass.wav", self.bass.y, self.sr, subtype='PCM_16')
        sf.write("drums.wav", self.drums.y, self.sr, subtype='PCM_16')

        vocal_audio = AudioSegment.from_wav("vocals.wav")
        other_audio = AudioSegment.from_wav("other.wav")
        #other_audio += 6
        bass_audio = AudioSegment.from_wav("bass.wav")
        drums_audio = AudioSegment.from_wav("drums.wav")

        audio = vocal_audio.overlay(other_audio)
        audio = audio.overlay(bass_audio)
        audio = audio.overlay(drums_audio)

        audio = self.compress(audio, threshold=-20.0, ratio=4.0)

        os.remove("vocals.wav")
        os.remove("other.wav")
        os.remove("bass.wav")
        os.remove("drums.wav")

        return audio


    def play(self, path):
        audio = self.auto_mix()
        audio.export(os.path.join(path, "mashup.wav"), format="wav")
    
