import utils
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Request
import os
from os import getcwd
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTasks
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory = "static"), name = "static")
templates = Jinja2Templates(directory = "templates")

PATH_FILES = getcwd() + "/"

@app.get("/")
# def read_root():
#     return {"msg": "Hello World"}
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/superres.html")
# def read_root():
#     return {"msg": "Hello World"}
def home(request: Request):
    return templates.TemplateResponse("superres.html", {"request": request})

@app.post("/superres.html")
async def predict(request: Request, file: UploadFile = File(...)):
    # with open(f'{PATH_FILES}{file.filename}', "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    # with open(PATH_FILES + file.filename, "wb") as myfile:
    content = await file.read()
        # myfile.write(content)
        # myfile.close()

    results = utils.superres(content)
    return templates.TemplateResponse("superres.html", {"request": request, "results": results})

@app.get("/colorizer.html")
# def read_root():
#     return {"msg": "Hello World"}
def home(request: Request):
    return templates.TemplateResponse("colorizer.html", {"request": request})

@app.post("/colorizer.html")
async def predict(request: Request, file: UploadFile = File(...)):
    # with open(f'{PATH_FILES}{file.filename}', "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    # with open(PATH_FILES + file.filename, "wb") as myfile:
    content = await file.read()
        # myfile.write(content)
        # myfile.close()
        
    results = utils.colorize(content)
    return templates.TemplateResponse("colorizer.html", {"request": request, "results": results})

