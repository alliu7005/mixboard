from flask_sqlalchemy import SQLAlchemy as _BaseSQLAlchemy
from song_struct import Song_Struct, Stem
import numpy as np
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, JSONB
from sqlalchemy.types import TypeDecorator, LargeBinary
import gzip, pickle

class SQLAlchemy(_BaseSQLAlchemy):
    def apply_pool_defaults(self, app, options):
        super(SQLAlchemy, self).apply_pool_defaults(self, app, options)
        options["pool_pre_ping"] = True

db = SQLAlchemy()


class CompressedNDArray(TypeDecorator):
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        arr = np.asarray(value)
        raw = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
        return gzip.compress(raw)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        raw = gzip.decompress(value)
        return pickle.loads(raw)
    
class CompressedPickle(TypeDecorator):
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return gzip.compress(raw)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        raw = gzip.decompress(value)
        return pickle.loads(raw)

class SongModel(db.Model):
    __tablename__ = 'songs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    sr = db.Column(db.Integer)
    tempo = db.Column(db.Float)
    key = db.Column(db.Integer)
    major = db.Column(db.Integer)
    bounds = db.Column(db.JSON)
    downbeats = db.Column(CompressedNDArray, nullable=False)
    beats = db.Column(CompressedNDArray, nullable=False)
    y = db.Column(CompressedNDArray, nullable=False)
    stems = db.relationship("StemModel", back_populates="song", cascade="all, delete-orphan")

class StemModel(db.Model):
    __tablename__ = 'stems'

    id = db.Column(db.Integer, primary_key=True)
    song_id = db.Column(db.Integer, db.ForeignKey('songs.id'))
    name = db.Column(db.String)
    songname = db.Column(db.String)
    sr = db.Column(db.Integer)
    tempo = db.Column(db.Float)
    key = db.Column(db.Integer)
    major = db.Column(db.Integer)
    bounds = db.Column(db.JSON)
    downbeats = db.Column(CompressedNDArray, nullable=False)
    beats = db.Column(CompressedNDArray, nullable=False)
    y = db.Column(CompressedNDArray, nullable=False)
    silent = db.Column(CompressedNDArray, nullable=False)
    active = db.Column(CompressedNDArray, nullable=False)
    specgram = db.Column(CompressedPickle, nullable=False)
    cluster = db.Column(db.Integer)
    song = db.relationship("SongModel", back_populates="stems")
    

class GraphModel(db.Model):
    __tablename__ = 'graph'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    kmeans = db.Column(db.PickleType)
    data = db.Column(db.PickleType)

def graph_from_orm(orm):
    data = orm.data
    kmeans = orm.kmeans
    return {"data": data, "kmeans": kmeans}

def song_from_orm(orm):
    name = orm.name
    sr = orm.sr
    orig_y = np.array(orm.y)
    tempo = orm.tempo
    key = orm.key
    major = orm.major
    bounds = orm.bounds
    downbeats = np.around(orm.downbeats,2)
    beats = np.around(orm.beats,2)
    stems = orm.stems
    for stem in stems:
        if stem.name == "vocals":
            v_y = np.array(stem.y)
        elif stem.name == "other":
            o_y = np.array(stem.y)
        elif stem.name == "bass":
            b_y = np.array(stem.y)
        else:
            d_y = np.array(stem.y)

    song = Song_Struct(orig_y, sr, name, v_y=v_y, o_y=o_y, b_y=b_y, d_y=d_y, tempo=tempo, key=key, major=major, bounds=bounds, downbeats=downbeats, beats=beats, take_fields=False)

    for i in range(len(song.stems)):
        song.stems[i].active = np.array(stems[i].active)
        song.stems[i].silence = np.array(stems[i].silent)
        song.stems[i].specgram = stems[i].specgram

    return song

def stem_from_orm(orm):
    init_song = song_from_orm(orm.song)
    y = orm.y
    sr = orm.sr
    name = orm.name
    tempo = orm.tempo
    key = orm.key
    major = orm.major
    bounds = orm.bounds
    downbeats = np.array(orm.downbeats)
    beats = np.array(orm.beats)
    silence = np.array(orm.silent)
    active = np.array(orm.active)
    specgram = [np.array(s) for s in orm.specgram]
    cluster=np.array(orm.cluster)
    
    stem = Stem(
        init_song=init_song,
        y=y,
        sr=sr,
        name=name,
        tempo=tempo,
        key=key,
        major=major,
        bounds=bounds,
        downbeats=downbeats,
        beats=beats,
        silence = silence,
        active = active,
        specgram = specgram
    )
    stem.cluster=cluster
    return stem