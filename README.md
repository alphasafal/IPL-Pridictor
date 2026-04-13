# 🏏 IPL Match Predictor API

A machine learning-powered REST API that predicts IPL match winners
using historical data from 2008–2026 (1188 matches).

## 🚀 Live Demo
**API:** https://ipl-pridictor.onrender.com

**Predict a match:**
https://ipl-pridictor.onrender.com/predict?team1=Chennai Super Kings&team2=Mumbai Indians&venue=Wankhede Stadium&toss_winner=Mumbai Indians&toss_decision=bat

**Interactive docs:**
https://ipl-pridictor.onrender.com/docs

## 📊 Project Overview
- Trained on 1188 IPL matches (2008–2026)
- Random Forest Classifier
- REST API built with FastAPI
- Deployed on Render (free tier)

## 🛠️ Tech Stack
- Python 3.13
- FastAPI
- Scikit-learn (RandomForestClassifier)
- Pandas
- Uvicorn
- Render (deployment)

## 📁 Project Structure
ipl-predictor/
├── api.py          # FastAPI application
├── work1.py        # Model training & exploration
├── ipl_json/       # 1188 match JSON files (2008-2026)
├── requirements.txt
└── README.md

## 🔌 API Endpoints

### GET /
Health check
{"message": "IPL Predictor API is live!"}

### GET /predict
Predict match winner
Parameters:

team1         : First team name
team2         : Second team name
venue         : Stadium name
toss_winner   : Who won the toss
toss_decision : bat or field


Example response:
```json
{
  "team1": "Chennai Super Kings",
  "team2": "Mumbai Indians",
  "predicted_winner": "Chennai Super Kings"
}
```

### GET /teams
Get all available team names
```json
{
  "teams": ["Chennai Super Kings", "Mumbai Indians", ...]
}
```

## 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/alphasafal/IPL-Pridictor.git
cd IPL-Pridictor

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn api:app --reload

# Open in browser
http://127.0.0.1:8000/docs
```

## 📈 Model Details
- Algorithm: Random Forest (100 estimators)
- Training data: 1188 matches
- Features: team1, team2, venue, toss winner, toss decision
- Data source: Cricsheet.org

## 🔮 Future Improvements
- [ ] Add player form features
- [ ] Head-to-head win rates
- [ ] Current season momentum
- [ ] Streamlit frontend dashboard
- [ ] XGBoost model comparison

## 👤 Author
**Safal Gupta**
- GitHub: @alphasafal
- VIT Vellore | CSE (IoT) | Batch 2027

## 📄 Data Source
Match data from [Cricsheet.org](https://cricsheet.org)
