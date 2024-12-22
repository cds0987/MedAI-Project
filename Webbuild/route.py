from flask import Blueprint, render_template, request, redirect, url_for, session
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
routebp = Blueprint('route', __name__)

@routebp.route('/')
def index():
    return render_template('index.html')

@routebp.route('/about')
def about():
    return render_template('Introduce/about.html')

@routebp.route('/model_capacity')
def model_capacity():
    return render_template('Introduce/model.html')

@routebp.route('/media')
def media():
    return render_template('Introduce/media.html')

@routebp.route('/references')
def references():
    return render_template('Introduce/references.html')



@routebp.route('/MainPage')
def MainPage():
    return render_template('MainPage/userpage.html')


@routebp.route('/oral')
def move_oral():
    session['modality'] = 'oral'
    return render_template('ResearchFeatures/UploadImage.html')

@routebp.route('/cataract')
def move_cataract():
    session['modality'] = 'cataract'
    return render_template('ResearchFeatures/UploadImage.html')

@routebp.route('/brain_class')
def move_brain_class():
    session['modality'] = 'brain_class'
    return render_template('ResearchFeatures/UploadImage.html')

@routebp.route('/brain_detect')
def move_brain_detect():
    session['modality'] = 'brain_detect'
    return render_template('ResearchFeatures/UploadImage.html')

@routebp.route('/retina')
def move_retina():
    session['modality'] = 'retina'
    return render_template('ResearchFeatures/UploadImage.html')
 