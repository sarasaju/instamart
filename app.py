from flask import Flask ,render_template,request,url_for,redirect
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_login import UserMixin,login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
#from sqlalchemy import text

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import nltk
import xgboost as xgb
import streamlit as st

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///customer_reviews.db"
app.config['SQLALCHEMY_BINDS'] = {
    'newdb': 'sqlite:///database.db'
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['SECRET_KEY'] = 'thisisasecretkey'

db=SQLAlchemy(app)
bcrypt = Bcrypt(app)



login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
  id = db.Column(db.Integer, primary_key=True)
  username = db.Column(db.String(20), nullable=False, unique=True)
  password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
  username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

  password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

  submit = SubmitField('Register')

  def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')
        
class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


class CustomerReview(db.Model):
  sno=db.Column(db.Integer,primary_key=True)
  first_name=db.Column(db.String(20),nullable=False)
  last_name=db.Column(db.String(20),nullable=False)
  email_id=db.Column(db.String(30),nullable=False)
  review=db.Column(db.String(25),nullable=False)
  data_creates=db.Column(db.DateTime,default=datetime.utcnow)



  def __repr__(self) -> str:
    return f"{self.sno} - {self.first_name}"
  
# class NewModel(db.Model):  # Example model for new database
#     __bind_key__ = 'newdb'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(50), nullable=False)
  

   


@app.route('/addreview',methods=['GET','POST'])
def hello_world():
  if request.method=='POST':
     #print(request.form['first_name'])
      first_name = request.form['first_name']
      last_name = request.form['last_name']
      email_id = request.form['email_id']
      review = request.form['review']
     
      customerreview=CustomerReview(first_name=first_name, last_name=last_name, email_id=email_id, review=review)
      db.session.add(customerreview)
      db.session.commit()
  allreview=CustomerReview.query.all()
  print(allreview)
  return render_template('index.html',allreview=allreview)
  #return 'Hello, World!'

@app.route('/show')
def products():
   allreview=CustomerReview.query.all()
   print(allreview)
   return "the customer review"

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('index.html')   #try to change to index.html


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/')
def home():
    return render_template('home.html')
## here u can go to ur webpage and if u add /products then it will return the above msg


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/analysis", methods=["GET", "POST"])   
def analysis():
    return render_template("sentiment_analysis.html")      #probably have to change location

@app.route("/predict", methods=["POST"])
def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"

            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})
        
        else:
            return jsonify({"error": "No file or text input provided"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})
    
def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["reviews"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph

def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"




if __name__=="__main__":
   app.run(debug=True)
    # with app.app_context():
    #     db.create_all()
    