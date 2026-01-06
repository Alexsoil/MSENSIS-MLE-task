from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import shutil
from Classifier import Classifier

app = FastAPI()

@app.post('/upload')
async def create_upload_file(file: UploadFile = File(...)):
    with open('images/your_image.jpg', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {'filename': file.filename}

@app.get('/')
async def root():
    return {'Message': 'Image Recognition Running'}

@app.get('/analyze/finetuned')
async def analyze_file():
    classifier = Classifier(model_name='finished_models/cats_dogs_finetune')
    category, confidence = classifier.classify('images/your_image.jpg')
    return {f'Image Class: {category}': f'Confidence: {confidence * 100:.2f}%'}

@app.get('/analyze/pretrained')
async def analyze_file():
    classifier = Classifier()
    category, confidence = classifier.classify('images/your_image.jpg')
    return {f'Image Class: {category}': f'Confidence: {confidence * 100:.2f}%'}