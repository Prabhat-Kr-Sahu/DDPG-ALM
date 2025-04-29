from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from src.pipeline.predict_pipeline import run_pipeline
from src.pipeline.train_pipeline import run_pipeline as train_pipeline
import os
import os
import pandas as pd

# File to store capital history
CAPITAL_FILE = "static/capital_history.csv"

# Load existing capital history if file exists, else initialize
if os.path.exists(CAPITAL_FILE):
    capital_df = pd.read_csv(CAPITAL_FILE)
    capital_history = capital_df['capital'].tolist()
else:
    initial_capital = 100000
    capital_history = [initial_capital]

app = FastAPI()
templates = Jinja2Templates(directory="templates")
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/reset-capital", response_class=HTMLResponse)
def reset_capital_form(request: Request):
    return templates.TemplateResponse("reset_capital.html", {"request": request})

@app.post("/reset-capital", response_class=HTMLResponse)
def reset_capital_submit(request: Request, initial_capital: float = Form(...)):
    # Overwrite capital history
    global capital_history
    capital_history = [initial_capital]
    capital_df = pd.DataFrame({"capital": capital_history})
    capital_df.to_csv(CAPITAL_FILE, index=False)
    # Redirect to home
    return RedirectResponse(url="/", status_code=303)

# ➡️ Route to train the model
@app.get("/train", response_class=HTMLResponse)
def train_pipeline_route(request: Request):
    # 👇 Call your actual train_pipeline function here
    train_pipeline()

    return templates.TemplateResponse("train_done.html", {"request": request})

# ➡️ Route to predict and show actions
@app.get("/predict", response_class=HTMLResponse)
def predict_pipeline_route(request: Request):
    returns, actions_dict = run_pipeline()

    daily_return = returns / 100
    new_capital = capital_history[-1] * (1 + daily_return)
    capital_history.append(new_capital)

    # Save to file
    capital_df = pd.DataFrame({"capital": capital_history})
    capital_df.to_csv(CAPITAL_FILE, index=False)

    return templates.TemplateResponse(
        "predict_done.html",
        {"request": request, "actions": actions_dict, "returns": returns}
    )

# ➡️ New route for capital page
@app.get("/capital", response_class=HTMLResponse)
def capital_page(request: Request):
    return templates.TemplateResponse(
        "capital.html",
        {"request": request, "capital_history": capital_history}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
