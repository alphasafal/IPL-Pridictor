import json
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = FastAPI()

# Train model on startup
all_matches = []
folder = "/Users/safalgupta/Downloads/ipl_json"

for file in os.listdir(folder):
    if file.endswith(".json"):
        with open(f"{folder}/{file}") as f:
            all_matches.append(json.load(f))

rows = []
for match in all_matches:
    info = match['info']
    try:
        rows.append({
            'team1': info['teams'][0],
            'team2': info['teams'][1],
            'venue': info['venue'],
            'toss_winner': info['toss']['winner'],
            'toss_decision': info['toss']['decision'],
            'winner': info['outcome'].get('winner', None)
        })
    except:
        pass

df = pd.DataFrame(rows)
df = df.dropna(subset=['winner'])

le_team = LabelEncoder()
le_venue = LabelEncoder()
le_toss = LabelEncoder()

all_teams = pd.concat([df['team1'], df['team2'],
                       df['toss_winner'], df['winner']])
le_team.fit(all_teams)

df['team1_enc'] = le_team.transform(df['team1'])
df['team2_enc'] = le_team.transform(df['team2'])
df['venue_enc'] = le_venue.fit_transform(df['venue'])
df['toss_winner_enc'] = le_team.transform(df['toss_winner'])
df['toss_decision_enc'] = le_toss.fit_transform(df['toss_decision'])
df['winner_enc'] = le_team.transform(df['winner'])

X = df[['team1_enc', 'team2_enc', 'venue_enc',
        'toss_winner_enc', 'toss_decision_enc']]
y = df['winner_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Model ready!")

@app.get("/")
def home():
    return {"message": "IPL Predictor API is live!"}

@app.get("/predict")
def predict(team1: str, team2: str, venue: str,
            toss_winner: str, toss_decision: str):
    try:
        t1 = le_team.transform([team1])[0]
        t2 = le_team.transform([team2])[0]
        v = le_venue.transform([venue])[0]
        tw = le_team.transform([toss_winner])[0]
        td = le_toss.transform([toss_decision])[0]

        pred = model.predict([[t1, t2, v, tw, td]])[0]
        winner = le_team.inverse_transform([pred])[0]

        return {
            "team1": team1,
            "team2": team2,
            "predicted_winner": winner
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/teams")
def get_teams():
    return {"teams": list(le_team.classes_)}