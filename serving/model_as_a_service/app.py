import codecs
import csv
import io
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import json
from sklearn import preprocessing

"""
Run application from root folder using the below command 
$ uvicorn serving.model_as_a_service.app:app --reload

Run the frontend also this is just the API. Run it from inside the folder
$ streamlit frontend.py
"""

app = FastAPI()
model = joblib.load("models/diabetes_rf.pkl")

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


@app.get("/model_meta")
async def get_model_meta_data():
    return model.get_params()

@app.post("/predict_single")
async def predict_single_patient(diabetes_info: DiabetesInfo):
    model_input_data = [diabetes_info.age, diabetes_info.sex, diabetes_info.bmi, diabetes_info.bp, diabetes_info.s1,
                        diabetes_info.s2, diabetes_info.s3, diabetes_info.s4, diabetes_info.s5, diabetes_info.s6]

    df = pd.DataFrame([model_input_data], columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])

    df = preprocessing.normalize(df)
    prediction = model.predict(df)

    return {'result': prediction[0]}


@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    read = await file.read()

    converted = str(read, 'utf-8')
    converted = converted.replace('\r', '')
    patient_diabetes_df = pd.read_csv(io.StringIO(converted), lineterminator='\n', index_col=0)
    print(patient_diabetes_df.head())

    patient_diabetes_df = patient_diabetes_df.drop(['target'], axis=1)
    progression = model.predict(patient_diabetes_df)
    progression = list(progression)

    return {'result': progression}
