import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
        team1 = info['teams'][0]
        team2 = info['teams'][1]
        toss_winner = info['toss']['winner']
        toss_decision = info['toss']['decision']
        winner = info['outcome'].get('winner', None)
        venue = info['venue']
        season = info['season']

        # Did toss winner win the match?
        toss_win_match = 1 if toss_winner == winner else 0
        # Did team batting first win?
        bat_first = 1 if toss_decision == 'bat' else 0

        rows.append({
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'season': str(season),
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'toss_win_match': toss_win_match,
            'bat_first': bat_first,
            'winner': winner
        })
    except:
        pass

df = pd.DataFrame(rows)
df = df.dropna(subset=['winner'])

# Encode
le_team = LabelEncoder()
le_venue = LabelEncoder()
le_toss = LabelEncoder()
le_season = LabelEncoder()

all_teams = pd.concat([df['team1'], df['team2'], 
                       df['toss_winner'], df['winner']])
le_team.fit(all_teams)

df['team1_enc'] = le_team.transform(df['team1'])
df['team2_enc'] = le_team.transform(df['team2'])
df['venue_enc'] = le_venue.fit_transform(df['venue'])
df['toss_winner_enc'] = le_team.transform(df['toss_winner'])
df['toss_decision_enc'] = le_toss.fit_transform(df['toss_decision'])
df['season_enc'] = le_season.fit_transform(df['season'])
df['winner_enc'] = le_team.transform(df['winner'])

X = df[['team1_enc', 'team2_enc', 'venue_enc',
        'toss_winner_enc', 'toss_decision_enc',
        'bat_first', 'season_enc']]
y = df['winner_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions)*100:.1f}%")