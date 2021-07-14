import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

"""
Note: you need to run the app from the root folder otherwise the models folder will not be found
- To run the app
$ uvicorn serving.model_as_a_service.main:app --reload

- To make a prediction from terminal
$ curl -X 'POST' 'http://127.0.0.1:8000/predict_obj' \
  -H 'accept: application/json' -H 'Content-Type: application/json' \
  -d '{ "age": 0, "sex": 0, "bmi": 0, "bp": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0, "s6": 0 }'
"""

app = FastAPI()
model = joblib.load("../../models/diabetes_model.joblib")


class DiabetesInfo(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/")
async def root():
    return {"message": "Hello World"}

"""
@app.post("/predict_single_patient")
async def predict_single_patient(
        age: float, sex: float, bmi: float, bp: float, s1: float, s2: float, s3: float, s4: float, s5: float,
        s6: float = 18):
    # age, sex, body_mass_index, average_blood_pressure, total_serum_cholesterol, low_density_lipoproteins,
    # high_density_lipoproteins, total_cholesterol, possibly_log_of_serum_triglycerides_level, blood_sugar_level
    model_input_data = np.array([age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]).reshape(1, -1)
    progression = model.predict(model_input_data)[0]
    return progression
"""

@app.post("/predict")
async def predict_diabetes_progress_1(diabetes_info: DiabetesInfo):
    # print(diabetes_info.dict())
    model_input_data = pd.DataFrame([diabetes_info.dict()])
    progression = model.predict(model_input_data)[0]
    return progression
