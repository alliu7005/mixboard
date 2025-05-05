from song_struct import Stem, Song_Struct
from models import db, SongModel, StemModel, stem_from_orm, song_from_orm, GraphModel, graph_from_orm
import librosa
from librosa_test import chord_similarity, shift_pitch, extract_chords, chord_matrix
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors

THRESHOLD = 0.8

def extract_stem_features(stem:Stem):
    #print(np.array(stem.mfcc).shape)
    chroma=np.mean(np.array(stem.chroma),axis=1).flatten()
    mfcc=np.mean(np.array(stem.mfcc), axis=1).flatten()
    tempo=stem.tempo
    onset_env = librosa.onset.onset_strength(y=stem.y, sr=stem.sr)
    onset_density=len(librosa.onset.onset_detect(onset_envelope=onset_env,sr=stem.sr))/librosa.get_duration(y=stem.y,sr=stem.sr)
    beats = stem.beats
    y = stem.y
    sr = stem.sr
    key = stem.key
    #print(mfcc.shape, chroma.shape)

    return {
        "tempo":tempo,
        "chroma":chroma,
        "mfcc":mfcc,
        "onset_density":onset_density,
        "y":y,
        "sr":sr,
        "key":key
    }

def cosine_similarity(vec1, vec2):
    return np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def vocals_other_compat(vocals:Stem,other:Stem):
    vocals_features = extract_stem_features(vocals)
    other_features = extract_stem_features(other)

    tempo_ratio = np.array(other_features["tempo"]) / np.array(vocals_features["tempo"])
    tempo_score = 1 - min(abs(tempo_ratio - 1), 1)

    mfcc_cosine = 1 - cosine(np.array(vocals_features["mfcc"]), np.array(other_features["mfcc"]),)

    y = shift_pitch(other.init_song.orig_y, other_features["sr"], other_features["key"], vocals_features["key"])

    chord_score = max(chord_similarity(vocals.init_song.orig_y, y, vocals_features["sr"]),0)

    chord_score = sharpen(chord_score)

    #print("VOCALS_OTHER")
    #print("chord score", vocals.init_song.name, other.init_song.name, chord_score)
    #print("mfcc", vocals.init_song.name, other.init_song.name, mfcc_cosine)
    #print("chroma", vocals.init_song.name, other.init_song.name, chroma_cosine)
    #print("tempo", vocals.init_song.name, other.init_song.name, tempo_score)

    weights = {"w1": 0.7, "w2": 0.1, "w3": 0.2}

    compat = (weights["w1"] * tempo_score + weights["w2"] * mfcc_cosine + weights["w3"] * chord_score)

    #print(compat)

    return compat

def other_bass_compat(bass:Stem,other:Stem):
    bass_features = extract_stem_features(bass)
    other_features = extract_stem_features(other)

    tempo_ratio = np.array(bass_features["tempo"]) / np.array(other_features["tempo"])
    tempo_score = 1 - min(abs(tempo_ratio - 1), 1)

    mfcc_cosine = 1 - cosine(np.array(other_features["mfcc"]), np.array(bass_features["mfcc"]))

    y = shift_pitch(bass.init_song.orig_y, bass_features["sr"], bass_features["key"], other_features["key"])

    chord_score = max(chord_similarity(other.init_song.orig_y, y, other_features["sr"]),0)
    chord_score = sharpen(chord_score)

    #print("OTHER_BASS")
    #print("chord score", bass.init_song.name, other.init_song.name, chord_score)
    #print("mfcc", bass.init_song.name, other.init_song.name, mfcc_cosine)
    #print("chroma", bass.init_song.name, other.init_song.name, chroma_cosine)
    #print("tempo", bass.init_song.name, other.init_song.name, tempo_score)

    weights = {"w1": 0.7, "w2": 0.1, "w3":0.2}

    compat = (weights["w1"] * tempo_score + weights["w2"] * mfcc_cosine + weights["w3"] * chord_score)

    #print(compat)

    return compat


def vocals_drums_compat(vocals:Stem,drums:Stem):
    vocals_features = extract_stem_features(vocals)
    drums_features = extract_stem_features(drums)

    tempo_ratio = np.array(drums_features["tempo"]) / np.array(vocals_features["tempo"])
    tempo_score = 1 - min(abs(tempo_ratio - 1), 1)

    onset_density_ratio = np.array(drums_features["onset_density"]) / np.array(vocals_features["onset_density"])
    onset_density_score = 1 - min(abs(onset_density_ratio - 1), 1)

    weights = {"w1": 0.85, "w2": 0.15}

    compat = (weights["w1"] * tempo_score + weights["w2"] * onset_density_score)

    return compat

def init_graph(dbsession):
    songs_orm = dbsession.query(SongModel).all()
    songs = [song_from_orm(s) for s in songs_orm]

    
    vocal_other_graph = [[]]
    other_bass_graph = [[]]
    vocal_drums_graph = [[]]


    for song1 in songs:
        for song2 in songs:
            if song1.name != song2.name:
                vo = vocals_other_compat(song1.vocals, song2.other)
                if vo > THRESHOLD:
                    vocal_other_graph[0].append((song1.name, song2.name, vo))
                ob = other_bass_compat(song1.other, song2.bass)
                if ob > THRESHOLD:
                    other_bass_graph[0].append((song1.name, song2.name, ob))
                vd = vocals_drums_compat(song1.vocals, song2.drums)
                if vd > THRESHOLD:
                    vocal_drums_graph[0].append((song1.name, song2.name, vd))
    
    print(vocal_other_graph)
    print(other_bass_graph)
    print(vocal_drums_graph)

    vocal_other = GraphModel(name="vocal_other",data=vocal_other_graph)
    dbsession.add(vocal_other)
    dbsession.commit()

    other_bass = GraphModel(name="other_bass",data=other_bass_graph)
    dbsession.add(other_bass)
    dbsession.commit()

    vocal_drums = GraphModel(name="vocal_drums",data=vocal_drums_graph)
    dbsession.add(vocal_drums)
    dbsession.commit()

def add_song_to_graph(dbsession, song):
    songs = dbsession.query(SongModel).all()
    vocal_other = dbsession.query(GraphModel).filter_by(name="vocal_other").all()[0]
    other_bass = dbsession.query(GraphModel).filter_by(name="other_bass").all()[0]
    vocal_drums = dbsession.query(GraphModel).filter_by(name="vocal_drums").all()[0]

    vocal_other_graph = graph_from_orm(vocal_other)
    other_bass_graph = graph_from_orm(other_bass)
    vocal_drums_graph = graph_from_orm(vocal_drums)

    songs_extracted = []
    for song_e in songs:
        song_ext = song_from_orm(song_e)
        songs_extracted.append(song_ext)

    for song2 in songs_extracted:
        if song.name != song2.name:
            vo = vocals_other_compat(song.vocals, song2.other)
            
            if vo > THRESHOLD:
                vocal_other_graph.append((song.name, song2.name, vo))
            vo2 = vocals_other_compat(song2.vocals, song.other)
            if vo2 > THRESHOLD:
                vocal_other_graph.append((song2.name, song.name, vo2))

            ob = other_bass_compat(song.other, song2.bass)
            
            if ob > THRESHOLD:
                other_bass_graph.append((song.name, song2.name, ob))
            ob2 = other_bass_compat(song2.other, song.bass)
            if ob2 > THRESHOLD:
                other_bass_graph.append((song2.name, song.name, ob2))

            vd = vocals_drums_compat(song.vocals, song2.drums)
            
            if vd > THRESHOLD:
                vocal_drums_graph.append((song.name, song2.name, vd))
            vd2 = vocals_drums_compat(song2.vocals, song.drums)
            if vd2 > THRESHOLD:
                vocal_drums_graph.append((song2.name, song.name, vd2))

    print(vocal_other_graph)
    print(other_bass_graph)
    print(vocal_drums_graph)

    dbsession.query(GraphModel).filter(GraphModel.name=="vocal_other").update(
        { GraphModel.data: vocal_other_graph }, 
        synchronize_session=False)
    dbsession.commit()

    dbsession.query(GraphModel).filter(GraphModel.name=="other_bass").update(
        { GraphModel.data: other_bass_graph }, 
        synchronize_session=False)
    dbsession.commit()

    dbsession.query(GraphModel).filter(GraphModel.name=="vocal_drums").update(
        { GraphModel.data: vocal_drums_graph }, 
        synchronize_session=False)
    dbsession.commit()

def sharpen(x, alpha=3):
    return (x**alpha) / (x**alpha + (1-x)**alpha)
