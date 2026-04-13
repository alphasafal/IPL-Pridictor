import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
all_matches = []
folder = "ipl_json"

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
            'winner': info['outcome'].get('winner', None),
            'date': info['dates'][0]
        })
    except:
        pass

df = pd.DataFrame(rows)
df = df.dropna(subset=['winner'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Total matches: {len(df)}")

# Build smart features chronologically
team1_winrate = []
team2_winrate = []
h2h_team1_rate = []
venue_team1_rate = []
team1_last5 = []
team2_last5 = []

win_count = {}
match_count = {}
h2h = {}
venue_stats = {}
recent_form = {}

for idx, row in df.iterrows():
    t1 = row['team1']
    t2 = row['team2']
    winner = row['winner']
    venue = row['venue']

    # Overall win rates
    t1_wr = win_count.get(t1, 0) / max(match_count.get(t1, 1), 1)
    t2_wr = win_count.get(t2, 0) / max(match_count.get(t2, 1), 1)
    team1_winrate.append(t1_wr)
    team2_winrate.append(t2_wr)

    # Head to head
    key = tuple(sorted([t1, t2]))
    h2h_data = h2h.get(key, {'t1_wins': 0, 'total': 1})
    h2h_t1 = h2h_data['t1_wins'] / max(h2h_data['total'], 1)
    h2h_team1_rate.append(h2h_t1)

    # Venue win rate for team1
    vkey = (venue, t1)
    vdata = venue_stats.get(vkey, {'wins': 0, 'total': 1})
    venue_team1_rate.append(vdata['wins'] / max(vdata['total'], 1))

    # Last 5 form
    t1_form = recent_form.get(t1, [])
    t2_form = recent_form.get(t2, [])
    team1_last5.append(sum(t1_form[-5:]) / max(len(t1_form[-5:]), 1))
    team2_last5.append(sum(t2_form[-5:]) / max(len(t2_form[-5:]), 1))

    # Update all stats
    win_count[winner] = win_count.get(winner, 0) + 1
    match_count[t1] = match_count.get(t1, 0) + 1
    match_count[t2] = match_count.get(t2, 0) + 1

    h2h.setdefault(key, {'t1_wins': 0, 'total': 0})
    h2h[key]['total'] += 1
    if winner == t1:
        h2h[key]['t1_wins'] += 1

    vkey1 = (venue, t1)
    vkey2 = (venue, t2)
    venue_stats.setdefault(vkey1, {'wins': 0, 'total': 0})
    venue_stats.setdefault(vkey2, {'wins': 0, 'total': 0})
    venue_stats[vkey1]['total'] += 1
    venue_stats[vkey2]['total'] += 1
    if winner == t1:
        venue_stats[vkey1]['wins'] += 1
    else:
        venue_stats[vkey2]['wins'] += 1

    recent_form.setdefault(t1, [])
    recent_form.setdefault(t2, [])
    recent_form[t1].append(1 if winner == t1 else 0)
    recent_form[t2].append(1 if winner == t2 else 0)

# Add features
df['team1_winrate'] = team1_winrate
df['team2_winrate'] = team2_winrate
df['h2h_team1'] = h2h_team1_rate
df['venue_team1'] = venue_team1_rate
df['team1_last5'] = team1_last5
df['team2_last5'] = team2_last5
df['toss_bat'] = (df['toss_decision'] == 'bat').astype(int)
df['toss_is_team1'] = (df['toss_winner'] == df['team1']).astype(int)
df['target'] = (df['winner'] == df['team1']).astype(int)

X = df[[
    'team1_winrate', 'team2_winrate',
    'h2h_team1', 'venue_team1',
    'team1_last5', 'team2_last5',
    'toss_bat', 'toss_is_team1'
]]
y = df['target']

# Random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"V2 Accuracy: {acc*100:.1f}%")

# Feature importance
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop Features:")
for i in range(len(features)):
    print(f"{features[indices[i]]}: {importances[indices[i]]:.3f}")