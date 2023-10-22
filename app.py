import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import logging


from bird_classifier import BirdClassifier

# get root logger
logger = logging.getLogger(__name__)

# initialise app
app = FastAPI()

# initialise classifier
classifier = BirdClassifier(logger)

def convert_dict_to_strings(input_dict):
    if isinstance(input_dict, dict):
        return {str(key): convert_dict_to_strings(value) for key, value in input_dict.items()}
    elif isinstance(input_dict, list):
        return [convert_dict_to_strings(item) for item in input_dict]
    elif isinstance(input_dict, tuple):
        return tuple(convert_dict_to_strings(item) for item in input_dict)
    else:
        return str(input_dict) 



class InputData(BaseModel):
    data: dict

class OutputData(BaseModel):
    data: dict

@app.post("/classify", response_model=OutputData)
async def classify_birds(data: InputData):
    url_dict = data.data
    output_dict = classifier.classify(url_dict)
    output_dict = convert_dict_to_strings(output_dict)
    return {"data": output_dict}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)