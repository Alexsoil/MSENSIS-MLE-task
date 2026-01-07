from fastapi import FastAPI, File, UploadFile
import shutil
import os
from Classifier import Classifier

app = FastAPI()

@app.post('/upload')
async def create_upload_file(file: UploadFile = File(...)):
    os.makedirs('images', exist_ok=True)
    with open(os.path.join('images', 'your_image.jpg'), 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {'Message': 'Upload Successful',
            'filename': file.filename}

@app.get('/analyze/finetuned')
async def analyze_file_finetuned():
    try:
        classifier = Classifier(model_name='Alexsoil/ViT-cats-dogs')
        category, confidence = classifier.classify(os.path.join('images', 'your_image.jpg'))
        return {'Message': 'Analysis complete using fine-tuned ViT Transformer.',
                f'Image Class: {category}': f'Confidence: {confidence * 100:.2f}%'}
    except FileNotFoundError as err:
        return {'Error': 'Please upload an image via the /upload endpoint.'}

@app.get('/analyze/pretrained')
async def analyze_file_pretrained():
    try:
        classifier = Classifier()
        category, confidence = classifier.classify(os.path.join('images', 'your_image.jpg'))
        return {'Message': 'Analysis complete using pretrained ViT Transformer.',
                f'Image Class: {category}': f'Confidence: {confidence * 100:.2f}%'}
    except FileNotFoundError as err:
        return {'Error': 'Please upload an image via the /upload endpoint'}

@app.get('/')
async def root():
    return {'Message': 'Image Recognition Running'}