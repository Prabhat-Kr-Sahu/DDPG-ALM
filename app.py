from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from src.pipeline.predict_pipeline import run_pipeline
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

@app.get("/reset")
def reset_capital():
    global capital_history
    capital_history = [100000]
    capital_df = pd.DataFrame({"capital": capital_history})
    capital_df.to_csv(CAPITAL_FILE, index=False)
    return {"message": "Capital reset to 100000"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Run your model pipeline and generate the plot
    returns, actions = run_pipeline()
        # Update capital based on today's return
    daily_return = returns / 100  # assuming returns is in %
    new_capital = capital_history[-1] * (1 + daily_return)
    capital_history.append(new_capital)

    # Save updated capital history to file
    capital_df = pd.DataFrame({"capital": capital_history})
    capital_df.to_csv(CAPITAL_FILE, index=False)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "actions": actions,
        "returns": returns,
        "capital_history": capital_history
    })

# ➡️ New route for capital page
@app.get("/capital", response_class=HTMLResponse)
def capital_page(request: Request):
    return templates.TemplateResponse(
        "capital.html",
        {"request": request, "capital_history": capital_history}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
