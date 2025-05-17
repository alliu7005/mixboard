from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from flask_session.__init__ import Session
import pickle
import os
import librosa_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import json
import shutil
import librosa
from song_struct import Song_Struct, Stem, Mashup
from models import db, SongModel, StemModel, GraphModel, song_from_orm, stem_from_orm, graph_from_orm
from graph import init_graph, add_song_to_graph
from flask_migrate import Migrate


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SESSION_FILE_DIR"] = "./flask_session_cache"
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = "filesystem"
Session(app)

db.init_app(app)
Migrate(app, db)

with app.app_context():
    db.create_all()
    

@app.route('/', methods = ['POST', 'GET'])
def index():

    if request.method == 'POST':

        #loads uploaded song and name
        stem_id = int(request.form.get('stem_id'))
        orm_stem = StemModel.query.filter_by(id=stem_id).all()[0]
        stem = stem_from_orm(orm_stem)
        song = stem.init_song

        vocal_other = GraphModel.query.filter_by(name="vocal_other").all()[0]
        other_bass = GraphModel.query.filter_by(name="other_bass").all()[0]
        vocal_drums = GraphModel.query.filter_by(name="vocal_drums").all()[0]

        #print(graph_from_orm(vocal_drums)["data"])

        vocal_other_graph = [t for t in graph_from_orm(vocal_other)["data"] if t[0]==song.name and t[1]!=song.name]
        vocal_drums_graph = [t for t in graph_from_orm(vocal_drums)["data"] if t[0]==song.name and t[1]!=song.name]

        vocal_other_graph = sorted(vocal_other_graph, key=lambda x: x[2], reverse=True)
        vocal_drums_graph = sorted(vocal_drums_graph, key=lambda x: x[2], reverse=True)
        print(vocal_drums_graph[0])

        other_name = vocal_other_graph[0][1]
        drums_name = vocal_drums_graph[0][1]

        other_orm = StemModel.query.filter_by(songname=other_name,name="other").all()[0]
        other_stem = stem_from_orm(other_orm)
        drums_orm = StemModel.query.filter_by(songname=drums_name,name="drums").all()[0]
        drums_stem = stem_from_orm(drums_orm)
        #print(graph_from_orm(other_bass), other_name, song.name)

        other_bass_graph = [t for t in graph_from_orm(other_bass)["data"] if t[0]==other_name and t[1]!=other_name]

        other_bass_graph = sorted(other_bass_graph, key=lambda x: x[2], reverse=True)

        bass_name = other_bass_graph[0][1]
        bass_orm = StemModel.query.filter_by(songname=bass_name,name="bass").all()[0]
        bass_stem = stem_from_orm(bass_orm)
        print(stem.init_song.name, other_stem.init_song.name, bass_stem.init_song.name, drums_stem.init_song.name)

        mashup = Mashup(song, stem, other_stem, bass_stem, drums_stem)

        session["mashup"] = pickle.dumps(mashup)
        mashup.play(app.config['UPLOAD_FOLDER'])
        mashup.visualize(app.config['UPLOAD_FOLDER'])

        
        #saves session variables for song file name
        #session['filename'] = filename
        #session['songname'] = name
        
        #redirects to results page
        return redirect(url_for('res'))
        
    else:
        #init_graph(db.session)
        orm_stems = StemModel.query.filter_by(name="vocals").all()
        return render_template('index.html', stems=orm_stems)

@app.route('/res', methods = ['POST', 'GET'])
def res():

    #loads section variables so processing can be done
    #filename = str(session.get('filename'))
    #name = str(session.get('songname'))

    #if request.method == 'GET':     

    #    y, sr = librosa_test.load_song(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    #    song = Song_Struct(y, sr, name)
    #    vocals = song.vocals
    #    other = song.other
    #    bass = song.bass
    #    drums = song.drums

    #    mashup = Mashup(song, vocals, other, bass, drums)
        
    
    #return results page
    return render_template('res.html')


@app.route('/load', methods = ['GET','POST'])
def load():

    if request.method == 'POST':

        #loads uploaded song and name
        song1 = request.files['song1']
        name = str(request.form.get('name'))
        filename = secure_filename(song1.filename)

        if song1:
            #saves original song into upload folder
            song_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            song1.save(song_path)

        
        y, sr = librosa_test.load_song(song_path)
        os.remove(song_path)

        song = Song_Struct(y, sr, name)
        vocals = song.vocals
        other = song.other
        bass = song.bass
        drums = song.drums

        song_dict = song.to_dict()
        vocals_dict = vocals.to_dict()
        other_dict = other.to_dict()
        bass_dict = bass.to_dict()
        drums_dict = drums.to_dict()
        stems = [vocals_dict, other_dict, bass_dict, drums_dict]
        store_song_from_dict(song_dict, stems, db.session)
        #init_graph(db.session)
        add_song_to_graph(db.session,song)
        
    
    return render_template('load.html')

def store_song_from_dict(song_dict, stems, dbsession):
    song_model = SongModel(
        name = song_dict["name"],
        sr = song_dict["sr"],
        tempo = song_dict["tempo"],
        key = song_dict["key"],
        major = song_dict["major"],
        bounds = song_dict["bounds"],
        downbeats = song_dict["downbeats"],
        beats = song_dict["beats"],
        y = song_dict["y"]
    )
    for stem_dict in stems:
        stem_model = StemModel(
            name = stem_dict["name"],
            songname = stem_dict["songname"],
            sr = stem_dict["sr"],
            tempo = stem_dict["tempo"],
            key = stem_dict["key"],
            major = stem_dict["major"],
            bounds = stem_dict["bounds"],
            downbeats = stem_dict["downbeats"],
            beats = stem_dict["beats"],
            y = stem_dict["y"],
            silent = stem_dict["silent"],
            active = stem_dict["active"],
            specgram = stem_dict["specgram"],
            cluster=stem_dict["cluster"])

        song_model.stems.append(stem_model)
    
    dbsession.add(song_model)
    dbsession.commit()
    print("stored song", song_model.name)

if __name__ == '__main__':
    app.run()
    