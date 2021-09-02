from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, ValidationError, validator
from typing import List, Optional
import utils

app = FastAPI()


@app.get("/")
def home():
    """
    Return html template render for home page form
    """

    return {"message": "Homepage"}


@app.post("/predict")
async def predict_from_api(file: UploadFile = File(...)
                           ):
    """
    Requires an image file upload and Optional image size parameter.
    Intended for API users.
    Return: JSON results of running YOLOv5 on the uploaded image.
    """

    file_str = str(await file.read())
    result = utils.predict(file_str)
    return result


if __name__ == '__main__':
    app_str = 'server:app'
    uvicorn.run(app_str, host='localhost', port=8008, workers=1)
