import random
import numpy as np
import math
import pandas as pd
import datetime

possible_education = [
    "PSLE", 
    "O Level", 
    "A Level", # or equivalent
    "Diploma",
    "Undergraduate",
    "Bachelors",
    "Masters",
    "PhD"
]

considered_levels = [
    "Primary 1 and 2",
    "Primary 3 and 4",
    "Primary 5 and 6",
    "PSLE",
    "Secondary 1",
    "Secondary 2",
    "Secondary 3",
    "Secondary 4",
    "Secondary 5",
    "O Level",
    "N Level",
    "JC 1",
    "JC 2",
    "A Level"
]

considered_subjects = [
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "English",
    "Chinese",
    "Malay",
    "Tamil",
    "Literature",
    "Geography",
    "History",
    "Art",
    "Music",
    "Computing"
]

# using postal codes from mrt stations (140 of them)
import json
with open('mrt_stations.json') as json_file:
    data = json.load(json_file)
postal_lst = []
for i in range(len(data)):
    postal = data[i]['Possible Locations'][0]['POSTAL']
    postal_lst.append(postal)

data = {
    'TutorID': np.arange(1000,2000),
    'LevelSubject' : [random.choice(considered_levels) + ' - ' + random.choice(considered_subjects) for i in range(1000)],
    'Location': [random.choice(postal_lst) for i in range(1000)],
    'Gender' : [random.choice(["Male", "Female", "Prefer Not to Say"]) for i in range(1000)],
    'DateTime' : ['NA' for i in range(1000)],
    'LowestRate': [random.choice([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]) for i in range(1000)],
    'HighestEducation' : [random.choice(possible_education) for i in range(1000)],
    'MOE' : [random.choice([True, False]) for i in range(1000)],
    'EXP' : [random.randint(0, 100) for i in range(1000)],
    'Feedback' : [random.randint(0, 100) for i in range(1000)]
}

df = pd.DataFrame(data)
df.to_csv("dummy_tutor_data1.csv")
