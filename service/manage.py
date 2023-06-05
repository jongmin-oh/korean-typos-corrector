from pydantic import BaseModel

import uvicorn

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from inference import infer

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class TextRequest(BaseModel):
    sentence: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def submit(request: Request):
    form = await request.form()
    sentence = form["sentence"]
    corrected = infer(sentence)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sentence": sentence,
            "corrected": corrected,
        },
    )


if __name__ == "__main__":
    uvicorn.run("manage:app", host="0.0.0.0", port=5900, reload=True)
