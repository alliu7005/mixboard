from song_struct import Stem, Song_Struct
from models import db, SongModel, StemModel, stem_from_orm, song_from_orm, GraphModel, graph_from_orm
import librosa
from librosa_test import chord_similarity, shift_pitch, extract_chords, chord_matrix, shift_pitch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

THRESHOLD = 0.75
BATCH_SIZE = 10

#extract stem features into dict for kmeans and 
def extract_stem_features(stem:Stem):
    #chroma=np.mean(np.array(stem.chroma),axis=1).flatten()
    mfcc=np.mean(np.array(librosa.feature.mfcc(y=stem.y, sr=stem.sr, n_mfcc=20)), axis=1).flatten()
    tempo=stem.tempo
    onset_env = librosa.onset.onset_strength(y=stem.y, sr=stem.sr)
    onset_density=len(librosa.onset.onset_detect(onset_envelope=onset_env,sr=stem.sr))/librosa.get_duration(y=stem.y,sr=stem.sr)
    y = stem.init_song.orig_y
    sr = stem.sr
    key = stem.key
    maj = stem.major
    y = shift_pitch(y, sr, key, 0)
    M = chord_matrix(extract_chords(y,sr,fps=10),fps=10)
    M_flattened = np.mean(np.array(M),axis=1).flatten()

    return {
        "tempo":tempo,
        "mfcc":mfcc,
        "onset_density":onset_density,
        "sr":sr,
        "key":key,
        "M": M,
        "M_flat": M_flattened,
        "major":maj
    }

def stem_embedding(stem: Stem):
    feat = extract_stem_features(stem)
    tempo = np.array([feat["tempo"]]) / 250.0
    maj = np.array([0.65 if feat["major"]== 1 else 0.35])
    #od = np.array([feat["onset_density"]]) / 8.0
    #mn, mx = feat["mfcc_stems"].min(), feat["mfcc_stems"].max()
    #mfcc = 2 * (feat["mfcc_stems"]-mn) / (mx - mn) -1
    #mn, mx = feat["M_flat"].min(), feat["M_flat"].max()
    #M_flat = (feat["M_flat"] - mn) / (mx - mn)

    #emb = np.concatenate([tempo, od, mfcc, M_flat])
    #emb = emb / np.sum(emb)
    #feat_weights = np.concatenate([[7.0], [0.1], np.ones(mfcc.shape[0]) * (0.9/np.sqrt(12)), np.ones(M_flat.shape[0]) * (2.0/np.sqrt(12))])

    return np.concatenate([tempo, maj])

def cosine_similarity(vec1, vec2):
    return np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def vocals_other_compat(vocals:Stem,other:Stem):
    vocals_features = extract_stem_features(vocals)
    other_features = extract_stem_features(other)

    tempo_ratio = np.array(other_features["tempo"]) / np.array(vocals_features["tempo"])
    tempo_score = 1 - min(abs(tempo_ratio - 1), 1)

    mfcc_cosine = 1 - cosine(np.array(vocals_features["mfcc"]), np.array(other_features["mfcc"]),)

    y = shift_pitch(other.init_song.orig_y, other_features["sr"], other_features["key"], vocals_features["key"])
    M2 = chord_matrix(extract_chords(y, other_features["sr"]))

    chord_score = max(chord_similarity(vocals_features["M"], M2),0)
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
    M2 = chord_matrix(extract_chords(y, bass_features["sr"]))

    chord_score = max(chord_similarity(other_features["M"], M2),0)
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

def cluster_stems(dbsession, songs_orm, songs_ext, n_clusters):
    all_ext = []
    all_orm = []
    for song_orm, song_ext in zip(songs_orm, songs_ext):
        for orm_stem in song_ext.stems:
            ext_stem = next(s for s in song_ext.stems if s.name == orm_stem.name)
            all_ext.append(ext_stem)
            all_orm.append(orm_stem)
    
    print("clustering stems")
    X = np.stack([stem_embedding(s) for s in all_ext])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    for i, c in enumerate(centers):
        print(f" Cluster {i}:", np.round(c, 3))


    print("labeling and uploading to db")
    for ext_stem, orm_stem, lbl in zip(all_ext, all_orm, labels):
        ext_stem.cluster = int(lbl)
        orm_stem.cluster = int(lbl)
        print(ext_stem.init_song.name, ext_stem.name, ext_stem.cluster)
        dbsession.query(StemModel).filter(StemModel.name==ext_stem.name, StemModel.songname==ext_stem.init_song.name).update(
        { StemModel.cluster: orm_stem.cluster }, 
        synchronize_session=False)
    dbsession.commit()

    return kmeans

def init_graph(dbsession):
    songs_orm = dbsession.query(SongModel).all()
    songs = [song_from_orm(s) for s in songs_orm]

    vocal_other_graph = []
    other_bass_graph = []
    vocal_drums_graph = []

    kmeans = cluster_stems(dbsession, songs_orm, songs, 2)

    print("building graph")
    for song_orm, song_ext in zip(songs_orm, songs):
        for other_orm, other_ext in zip(songs_orm, songs):
            if song_ext.name != other_ext.name:
                if song_ext.vocals.cluster == other_ext.other.cluster:
                    vo = vocals_other_compat(song_ext.vocals, other_ext.other)
                    print("vocal other", song_ext.name, other_ext.name, vo)
                    if vo > THRESHOLD:
                        vocal_other_graph.append((song_ext.name, other_ext.name, vo))
                if song_ext.other.cluster == other_ext.bass.cluster:
                    ob = other_bass_compat(song_ext.other, other_ext.other)
                    print("other bass", song_ext.name, other_ext.name, ob)
                    if ob > THRESHOLD:
                        other_bass_graph.append((song_ext.name, other_ext.name, ob))
                if song_ext.vocals.cluster == other_ext.drums.cluster:
                    vd = vocals_drums_compat(song_ext.vocals, other_ext.drums)
                    print("vocal drums", song_ext.name, other_ext.name, vd)
                    if vd > THRESHOLD:
                        vocal_drums_graph.append((song_ext.name, other_ext.name, vd))



    #for song1 in songs:
    #    for song2 in songs:
    #        if song1.name != song2.name:
    #            vo = vocals_other_compat(song1.vocals, song2.other)
    #            if vo > THRESHOLD:
    #                vocal_other_graph[0].append((song1.name, song2.name, vo))
    #            ob = other_bass_compat(song1.other, song2.bass)
    #            if ob > THRESHOLD:
    #                other_bass_graph[0].append((song1.name, song2.name, ob))
    #            vd = vocals_drums_compat(song1.vocals, song2.drums)
    #            if vd > THRESHOLD:
    #                vocal_drums_graph[0].append((song1.name, song2.name, vd))
    
    print(vocal_other_graph)
    print(other_bass_graph)
    print(vocal_drums_graph)

    vocal_other = GraphModel(name="vocal_other",data=vocal_other_graph, kmeans=kmeans)
    dbsession.merge(vocal_other)
    dbsession.commit()

    other_bass = GraphModel(name="other_bass",data=other_bass_graph, kmeans=kmeans)
    dbsession.merge(other_bass)
    dbsession.commit()

    vocal_drums = GraphModel(name="vocal_drums",data=vocal_drums_graph, kmeans=kmeans)
    dbsession.merge(vocal_drums)
    dbsession.commit()

def add_song_to_graph(dbsession, song:Song_Struct):
    songs_orm = dbsession.query(SongModel).all()
    stems_orm = dbsession.query(StemModel).all()
    vocal_other = dbsession.query(GraphModel).filter_by(name="vocal_other").all()[0]
    other_bass = dbsession.query(GraphModel).filter_by(name="other_bass").all()[0]
    vocal_drums = dbsession.query(GraphModel).filter_by(name="vocal_drums").all()[0]

    vocal_other_graph = graph_from_orm(vocal_other)["data"]
    other_bass_graph = graph_from_orm(other_bass)["data"]
    vocal_drums_graph = graph_from_orm(vocal_drums)["data"]
    

    stems_ext = [stem_from_orm(s) for s in stems_orm]
    print(stems_ext, stems_orm)

    if len(stems_ext) % 11 == 0:
        songs_ext = [song_from_orm(s) for s in songs_orm]
        kmeans = cluster_stems(dbsession, songs_orm, songs_ext, 2)
        return
    else:
        kmeans = graph_from_orm(vocal_other)["kmeans"]

    for stem in song.stems:
        emb = stem_embedding(stem).reshape(1,-1)
        label = int(kmeans.predict(emb)[0])
        stem.cluster = label
        dbsession.query(StemModel).filter(StemModel.name==stem.name, StemModel.songname==stem.init_song.name).update(
        { StemModel.cluster: stem.cluster }, 
        synchronize_session=False)
        dbsession.commit()

    for other_orm, other_ext in zip(stems_orm, stems_ext):
        print(song.vocals.cluster, other_ext.init_song.name, other_ext.cluster)
        if song.vocals.cluster != other_ext.cluster or song.name == other_ext.name:
            continue
        if other_ext.name == "other":
            vo = vocals_other_compat(song.vocals, other_ext)
            if vo > THRESHOLD:
                vocal_other_graph.append((song.name, other_ext.init_song.name, vo))
        elif other_ext.name == "drums":
            vd = vocals_drums_compat(song.vocals, other_ext)
            if vd > THRESHOLD:
                vocal_drums_graph.append((song.name, other_ext.init_song.name, vd))
        elif other_ext.name == "bass":
            ob = other_bass_compat(song.other, other_ext)
            if ob > THRESHOLD:
                other_bass_graph.append((song.name, other_ext.init_song.name, ob))



    #for song2 in songs_ext:
    #    if song.name != song2.name:
    #        vo = vocals_other_compat(song.vocals, song2.other)
            
    #        if vo > THRESHOLD:
    #            vocal_other_graph.append((song.name, song2.name, vo))
    #        vo2 = vocals_other_compat(song2.vocals, song.other)
    #        if vo2 > THRESHOLD:
    #            vocal_other_graph.append((song2.name, song.name, vo2))

    #        ob = other_bass_compat(song.other, song2.bass)
            
    #        if ob > THRESHOLD:
    #            other_bass_graph.append((song.name, song2.name, ob))
    #        ob2 = other_bass_compat(song2.other, song.bass)
    #        if ob2 > THRESHOLD:
    #            other_bass_graph.append((song2.name, song.name, ob2))

    #        vd = vocals_drums_compat(song.vocals, song2.drums)
            
    #        if vd > THRESHOLD:
    #            vocal_drums_graph.append((song.name, song2.name, vd))
    #        vd2 = vocals_drums_compat(song2.vocals, song.drums)
    #        if vd2 > THRESHOLD:
    #            vocal_drums_graph.append((song2.name, song.name, vd2))

    print(vocal_other_graph)
    print(other_bass_graph)
    print(vocal_drums_graph)

    dbsession.query(GraphModel).filter(GraphModel.name=="vocal_other").update(
        { GraphModel.data: vocal_other_graph, GraphModel.kmeans: kmeans }, 
        synchronize_session=False)
    dbsession.commit()

    dbsession.query(GraphModel).filter(GraphModel.name=="other_bass").update(
        { GraphModel.data: other_bass_graph, GraphModel.kmeans: kmeans }, 
        synchronize_session=False)
    dbsession.commit()

    dbsession.query(GraphModel).filter(GraphModel.name=="vocal_drums").update(
        { GraphModel.data: vocal_drums_graph, GraphModel.kmeans: kmeans }, 
        synchronize_session=False)
    dbsession.commit()

def sharpen(x, alpha=3):
    return (x**alpha) / (x**alpha + (1-x)**alpha)
