# web app packages
import requests
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response

import json

# for data loading and transformation
import numpy as np 
import pandas as pd

# for statistics output
from scipy import stats
from scipy.stats import randint

# for data preparation and preprocessing for model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# models
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
# Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Naive Bayes
from sklearn.naive_bayes import GaussianNB 
# Stacking
from mlxtend.classifier import StackingClassifier

# model evaluation and validation 
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score

# for db connection
import sqlite3
db_filename="database.db"

# for saving/loading the ML model
import pickle
model_filename="models/model.pkl"

# to bypass warnings in the jupyter notebook
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=PendingDeprecationWarning)

app=Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# instantiate index page
@app.route("/")
def index():
   	return render_template("index.html")

# return model predictions
@app.route("/api/predict", methods=["GET"])
def predict():
	msg_data={}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	f = open("models/X_test.json","r")
	X_test = json.load(f)
	f.close()
	all_cols=X_test
	input_df=pd.DataFrame(msg_data,columns=all_cols,index=[0])
	model = pickle.load(open(model_filename, "rb"))
	arr_results = model.predict(input_df)
	treatment_likelihood=""
	if arr_results[0]==0:
		treatment_likelihood="No"
	elif arr_results[0]==1:
		treatment_likelihood="Yes"
	return treatment_likelihood

if __name__ == "__main_":
	app.debug = False
	from werkzeug.serving import run_simple
	run_simple("localhost", 5000, app)