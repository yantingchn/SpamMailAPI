# !pip install fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model import Modelling

spam_check_api = FastAPI()

# Load the deployed model from the mlflow file
modelling = Modelling()
model = modelling.load_deployed_model()

class Request(BaseModel):
    message: str

@spam_check_api.post("/predict")
def predict_spam(req: Request):
    text = req.dict()['message']
    # Preprocess your data and make a prediction
    prediction = model.predict([text])
    prediction = modelling.le.inverse_transform(prediction)
    # Return the prediction
    return {"prediction": str(prediction)}