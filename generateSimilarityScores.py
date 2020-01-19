# Ver 0.2
# SpaCy NLP, Google Map API, sklearn k means clustering, Firebase

"""
FUTURE WORK - REFERENCES: 
# K Means clustering and clustering based recommendations
https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
https://github.com/ashishrana160796/Online-Course-Recommendation-System/blob/master/README.md
https://towardsdatascience.com/build-your-own-clustering-based-recommendation-engine-in-15-minutes-bdddd591d394

# Real estate recommendation with decision trees
https://github.com/hyattsaleh15/RealStateRecommender/blob/master/Case%20of%20Study.ipynb

# travel recommendation
https://github.com/jnatale11/CBR-Travel-Recommendation/blob/master/REPORT.pdf

# Hotel recommendation
https://github.com/saurabhwani4/Hotel-Recommendation-System/blob/master/Hotel%20Recommender.ipynb

# Movie recommender with cosine similarity
https://github.com/devpranjalsahu/Movie_Recommendation_system/blob/master/movie_recommender_model.py

# Full react - flask - ml - predict
https://github.com/ajaniv/react-flask-ml-predict

# Full ML Webapp: Keras ML + Flask backend + React frontend 
https://www.youtube.com/watch?v=6pcIyKIBEVQ

# Deploying ML flask app to Google Cloud:
https://www.youtube.com/watch?v=RbejfDTHhhg&list=PLJ39kWiJXSiyAFG2W3CUPWaLhvR5CQmTd&index=3

# Full ML Webapp: Keras/sklearn + Django backend forms + Bootstrap frontend forms + deploy to Heroku
https://www.youtube.com/watch?v=tDnAcbYROSI&list=PLM30lSIwxWOivNtja1_ztft1S_vrMyA0z&index=1

"""

# Variables and links 
FIREBASE_LINK = "" 
TUTOR_CSV = ""
GOOGLE_API_KEY = ""

# Import database module.
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
firebaseApp = firebase_admin.initialize_app(credential=None, options=None, name =FIREBASE_LINK)

from firebase import firebase 
firebase = firebase.FirebaseApplication(FIREBASE_LINK, None)

import random
import numpy as np
import pandas as pd
import math
import time
#from sklearn.feature_extraction.text import TfidfVectorizer
import spacy 
import en_core_web_sm
nlp = en_core_web_sm.load()

tutor_data = pd.read_csv(TUTOR_CSV, sep=',', engine='python')
tutor_data.head()
for i in range(len(tutor_data["TutorID"])):
    try:
        rate = tutor_data["Rate"][i]
        new_rate = rate.strip("/h").strip("$")
        if rate == "Negotiable":
            new_rate = 30
    except AttributeError:
        continue
    tutor_data["Rate"][i] = new_rate
    exp = tutor_data["Experience"][i]
    remarks = tutor_data["Remarks"][i]
    combined = str(exp) + " " + str(remarks)
    tutor_data["Experience"][i] = combined

tutor_data.drop("Remarks", axis =1)

#################### A. Gather input requirements from the parent / student ####################

## Get relevant data from firebase 
subject = firebase.get("/", 'Subject')
level = firebase.get("/", 'Level')
combined_level_subject = level + " - " + subject
experiences_input = firebase.get("/", 'Experiences') 
location_input = firebase.get("/", 'Postal Code')
gender_input = firebase.get("/", 'Gender')
budget_input = firebase.get("/", 'Budget')
education_input = firebase.get("/", 'Education')
moe_input = firebase.get("/", 'MOE')

#################### B. Initialize weights #################### 

preference_weights = {
    "Gender" : 0.05,
    "LevelSubject" : 0.3,
    "Duration" : 0.05,
    "Rate" : 0.15,
    "Education" : 0.2,
    "MOE" : 0.05,
    "Experiences" : 0.2
    #"Date" : 0
}

#################### C. Compute similarity scores based on requirements #################### 


####### 1. Compute sim for LevelSubject --> NLP #######
subcompare1 = nlp(combined_level_subject)
sub_scores = []

for i in range(len(tutor_data['LevelSubject'])):
    subcompare2 = nlp(tutor_data['LevelSubject'][i])
    simvalue = subcompare1.similarity(subcompare2)
    if simvalue >= 0.98: # set threshold for level subject 
        sub_scores.append(simvalue)
    else:
        sub_scores.append(0) # else set to 0

# add the sim score for Level Subject matching
tutor_data['LevelSubjectScore'] = sub_scores

####### 2. Compute sim for remarks and experiences --> NLP #######
expcompare1 = nlp(experiences_input)
exp_scores = []

for i in range(len(tutor_data['Experience'])):
    expcompare2 = nlp(tutor_data['Experience'][i])
    simvalue = expcompare1.similarity(expcompare2)
    exp_scores.append(simvalue)

# add the sim score for experiences matching
tutor_data['ExperiencesScore'] = exp_scores

####### 3. Compute sim for Gender, Budget, Required Education, and MOE certification --> Simple if-else #######

# scores are all from 0 to 1 scale
gender_scores = []
rate_diff_lst = []
rate_scores = []
moe_scores = []
edu_scores = []

# this edu level is ranked from lowest to highest
possible_education = [
    "PSLE", 
    "O Levels", 
    "A Levels", # or equivalent
    "Diploma",
    "Undergraduate",
    "Bachelors",
    "Masters",
    "PhD"
]

def getEduScore(required, tutor):
    required = required.strip()
    tutor = tutor.strip()
    required_level = possible_education.index(required)
    tutor_level = possible_education.index(tutor)
    if tutor_level >= required_level:
        edu_score = 1
    else:
        edu_score = 1 - 0.1*(required_level - tutor_level)
    return edu_score

for i in range(len(tutor_data['Gender'])):
    # Gender score calculation
    if gender_input in ['Male', 'Female']:
        if gender_input == tutor_data['Gender'][i]:
            gender_score = 1
        else:
            gender_score = 0.2
    else: # for no preference
        gender_score = 0
    gender_scores.append(gender_score)

    # MOE score calculation
    if moe_input == "True":
        moe_score = 1 if tutor_data['MOE'][i] == 1 else 0.2
    else: # for no preference
        moe_score = 0
    moe_scores.append(moe_score)
    
    # Edu score calculation
    edu_score = getEduScore(education_input, tutor_data['Education'][i])
    edu_scores.append(edu_score)

    # Budget diff calculation 
    tutor_rate = tutor_data['Rate'][i]
    rate_diff = float(budget_input) - float(tutor_rate)
    rate_diff_lst.append(rate_diff)

# append normalized rate scores
max_rate = max(rate_diff_lst)
min_rate = min(rate_diff_lst)
for i in rate_diff_lst:
    rate_scores.append((i - min_rate)/(max_rate - min_rate))
    
tutor_data['GenderScore'] = gender_scores
tutor_data['RateScore'] = rate_scores
tutor_data['MoeScore'] = moe_scores
tutor_data['EduScore'] = edu_scores


####### 4. Compute sim for postal code matching --> google maps API #######
import urllib.request
import json

def getMapDistance(START_POSTAL_CODE, END_POSTAL_CODE):
    FINAL_GOOGLE_MAP_URL = "https://maps.googleapis.com/maps/api/distancematrix/json?units=metric&mode=transit&origins=Singapore+{}&destinations=Singapore+{}&key={}".format(START_POSTAL_CODE, END_POSTAL_CODE, GOOGLE_API_KEY)
    # default to metres, seconds, postal code, and travelling mode by transit (MRT and bus)
    with urllib.request.urlopen(FINAL_GOOGLE_MAP_URL) as url:
        data = json.loads(url.read().decode())
        try:
            #distance_value = data['rows']['distance']['value']
            distance_text = data['rows'][0]['elements'][0]['distance']['text']
            duration_value = data['rows'][0]['elements'][0]['duration']['value']
            duration_text = data['rows'][0]['elements'][0]['duration']['text']
        except KeyError:
            distance_text = 'NAN'
            duration_value ='NAN'
            duration_text = 'NAN'
    return distance_text, duration_value, duration_text

distance_text_lst = []
duration_value_lst = []
duration_text_lst = []
duration_scores = []

for i in range(len(tutor_data['Location'])):
    START_POSTAL_CODE = location_input
    END_POSTAL_CODE = tutor_data['Location'][i]
    distance_text, duration_value, duration_text = getMapDistance(START_POSTAL_CODE, END_POSTAL_CODE)
    time.sleep(0.05)
    distance_text_lst.append(distance_text)
    duration_value_lst.append(duration_value)
    duration_text_lst.append(duration_text)

actual_duration_lst = [i for i in duration_value_lst if i != 'NAN']
max_duration = max(actual_duration_lst)
min_duration = min(actual_duration_lst)

for i in duration_value_lst:
    if i == 'NAN':
        duration_scores.append(0)
    else:
        # append normalized duration scores # aka related to the distance values
        duration_scores.append(1 - (i - min_duration)/(max_duration - min_duration)) # inverse, the faster the better

tutor_data['DurationScore'] = duration_scores
tutor_data['DurationTexts'] = duration_text_lst
tutor_data['DistanceTexts'] = distance_text_lst

####### 5. Compute sim for date matching --> schedule matching ??? #######



####### 6. Compute final score for every tutor #######
aggregate_lst = []
for i in range(len(tutor_data['TutorID'])):
    gender_score = tutor_data['GenderScore'][i] * preference_weights["Gender"]
    rate_score = tutor_data['RateScore'][i] * preference_weights["Rate"]
    moe_score = tutor_data['MoeScore'][i] * preference_weights["MOE"]
    edu_score = tutor_data['EduScore'][i] * preference_weights["Education"]
    exp_score = tutor_data['ExperiencesScore'][i] * preference_weights["Experiences"]
    subject_score = tutor_data['LevelSubjectScore'][i] * preference_weights["LevelSubject"]
    duration_score = tutor_data['DurationScore'][i] * preference_weights["Duration"]
    
    aggregate_score = gender_score + rate_score + moe_score + edu_score + exp_score + subject_score + duration_score
    aggregate_lst.append(aggregate_score)

tutor_data['AggregateScore'] = aggregate_lst

tutor_data.sort_values(by = 'AggregateScore', ascending = False)