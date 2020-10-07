from typing import Optional
from fastapi import FastAPI
#from . import predict_synthese_sanitaire as pr 
from predict_synthese_sanitaire import clf 
from predict_synthese_sanitaire import vectorizer
#from . import predict_synthese_sanitaire as pr

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/prediction/{restaurant_name}")
def read_item(restaurant_name: str):
    return {"restaurant " : restaurant_name, "prediction": str(clf.predict(vectorizer.transform([restaurant_name])))}
    