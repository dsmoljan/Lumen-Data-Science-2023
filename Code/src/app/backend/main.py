from io import BytesIO

import fastapi
import torch
import librosa as lr
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.logger import logger
from starlette.middleware.cors import CORSMiddleware

from src.app.backend.model_wrapper import ModelWrapper

app = FastAPI()

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.on_event("startup")
async def startup_event():
    """
    Initalize FastAPI server and the base model used for prediction.
    """
    model_wrapper = ModelWrapper()

    # https://stackoverflow.com/questions/71298179/fastapi-how-to-get-app-instance-inside-a-router
    app.model = model_wrapper


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/api/predict")
# ako koristis request, ne mozes i audio_file
async def do_predict(request: Request, audio_file: UploadFile = File(...)):
    """
    Perform prediction on input data
    :param audio_file:
    :param request: HTTP request
    :param body: body of the HTTP request, containing the audio file to classify
    :return:
    """
    # Load the audio file from the request body
    contents = await audio_file.read()
    audio_signal, sr = lr.load(BytesIO(contents), sr=None, mono=True)
    # TODO: premjestiti u preprocessing
    if sr != 44100:
        audio_signal = lr.resample(audio_signal, sr, 16000)
        sr = 44100

    # https://stackoverflow.com/questions/71298179/fastapi-how-to-get-app-instance-inside-a-router
    model = request.app.model
    logger.info('API predict method called')
    logger.info(f'input: {audio_file}')
    return model.predict(audio_signal, sr)





if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)