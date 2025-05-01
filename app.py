from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os
from src.pipeline.predict_pipeline import run_pipeline
from src.pipeline.train_pipeline import run_pipeline as train_pipeline
import uvicorn

# Constants
CAPITAL_FILE = "static/capital_history.csv"
INITIAL_CAPITAL = 100000

# Load capital history
if os.path.exists(CAPITAL_FILE):
    capital_df = pd.read_csv(CAPITAL_FILE)
    capital_history = capital_df["capital"].tolist()
else:
    capital_history = [INITIAL_CAPITAL]
    pd.DataFrame({"capital": capital_history}).to_csv(CAPITAL_FILE, index=False)

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Replace with your React dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/capital-json")
def get_capital_data():
    df = pd.read_csv(CAPITAL_FILE)
    return df["capital"].tolist()

@app.post("/reset-capital")
def reset_capital(initial_capital: float = Form(...)):
    global capital_history
    capital_history = [initial_capital]
    pd.DataFrame({"capital": capital_history}).to_csv(CAPITAL_FILE, index=False)
    return {"message": "Capital reset", "initial_capital": initial_capital}

@app.post("/predict")
def predict_pipeline_route():
    returns, actions_dict = run_pipeline()

    daily_return = returns / 100
    new_capital = capital_history[-1] * (1 + daily_return)
    capital_history.append(new_capital)

    pd.DataFrame({"capital": capital_history}).to_csv(CAPITAL_FILE, index=False)

    return {
        "returns": returns,
        "actions": actions_dict,
        "new_capital": new_capital
    }

@app.post("/train")
def train_pipeline_route():
    train_pipeline()
    return {"message": "Training completed"}
