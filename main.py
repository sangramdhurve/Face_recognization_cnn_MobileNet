import time
import uvicorn
import logging
import numpy as np
from fastapi import FastAPI
from models.load_model import classifier
from fastapi import File, UploadFile, HTTPException
from preprocessing import read_imagefile, Preprocessing


# Let's create log file
logging.basicConfig(filename="log.txt", level=logging.INFO, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
                    , filemode="a")
logging.debug("Debug logging test...")


# initialize app and call the model
app = FastAPI(debug=True)
model = classifier()


# Let's give some Classes for predicton
lst = ['Feraz','Ganesh','Lakshmee','Pooja','Roshan','Sathish','Shrikant','Shruthi']


# The Homepage of Api
@app.get('/')
async def index():
    mess = "We are happy to serve you."
    return "Welcome to Face Detection API", mess


# Start recording the time
start = time.time()
@app.post("/prediction")
async def preprocess_api(file: UploadFile = File(...)):

    # Checking The Image Extension whether it is right extension or not
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    print(f"Image Extension is :{extension} ")
    if not extension:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an right extension of image.')
    logging.info(f"Image must be jpg or png format! : {extension}")


    try:
        image = read_imagefile(await file.read())
        logging.info(f"Image is Read as expected , {image}")

        img = Preprocessing(image)
        logging.info(f"Image Preprocessing is Successful : {img}")

        prediction = model.predict(img)[0]

        max_value = np.max(prediction)
        name = lst[np.argmax(prediction)]
        return {"Name of the person is => ": name,
                "prediction percentage => ": float(max_value)}


    except Exception as E:
        logging.exception(f"Warning, the image preprocessing may not successful , Exception is {E}")
        end = time.time()
        logging.info(f"Took {end - start} secs for Whole Image Preprocessing")
        logging.info("\n")


if __name__ == '__main__':
    uvicorn.run(app)

