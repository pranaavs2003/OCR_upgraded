from fastapi import FastAPI, HTTPException
import requests
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from detector import detector

import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs


#Initialize FastAPI
app = FastAPI()

# Set up allowed origins
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://checkd.vercel.app"
    "https://ocr-a4g8.onrender.com"
]

# Add the CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    msg: str

class ImageDetail(BaseModel):
    image_url: str
    

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


@app.post("/test")
def test(message: Message):
    print('Route working successfully! 🌄')
    res = {"status": 200,
            "msg": message.msg
           }
    return res

# Define the POST endpoint
@app.post("/predict")
async def predict_image(image_details: ImageDetail):
    
    try:
        # Download the image
        response = requests.get(image_details.image_url)
        response.raise_for_status()
        image_bytes = BytesIO(response.content)

        # Convert the image to a format suitable for OpenCV
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Initialize your model (this might need adjustments)
        configs = BaseModelConfigs.load("202301111911/configs.yaml")
        model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

        # Predict
        prediction_text = model.predict(image)
        ocrData = detector(image_details)
        print('🆑' ,ocrData)
        return {"prediction": ocrData}

    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Error fetching the image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.post('/extract')
def extract(image_details: ImageDetail):
    ocrData = detector(image_details)
    print('🆑' ,ocrData)
    return ocrData