import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import os

# set tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from bird_classifier import BirdClassifier
from bird_classifier.utils import Utils

# get root logger
logger = logging.getLogger(__name__)

# initialise app
app = FastAPI()

# initialise classifier
classifier = BirdClassifier(logger)
class InputData(BaseModel):
    data: dict

class OutputData(BaseModel):
    data: dict

@app.post("/classify", response_model=OutputData)
async def classify_birds(data: InputData):
    """Classify birds in images
    
    Args:
        data (InputData): Input data
        
    Returns:
        OutputData: Output data 
    """
    # get urls from input data
    url_dict = data.data

    # classify images
    output_dict = classifier.classify(url_dict)

    # convert all values in output_dict to strings
    output_dict = Utils.convert_dict_to_strings(output_dict)

    return {"data": output_dict}

logger.info("Bird classifier app swagger docs: http://127.0.0.1:8000/docs")

if __name__ == "__main__":

    # run app
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # # use logger to give hyperlink to swagger docs