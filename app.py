from flask import Flask
from Webbuild.route import routebp
from Webbuild.feature import featurebp
app = Flask(__name__)
app.config['SECRET_KEY'] = '1537'
app.config.from_pyfile('config.py')
# Register blueprints
app.register_blueprint(routebp, url_prefix='/')
app.register_blueprint(featurebp, url_prefix='/feature')


