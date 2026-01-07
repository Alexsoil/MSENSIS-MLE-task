# MSENSIS-MLE-task
A simple web application that classifies images of cats and dogs using pre-trained models and allows users to test custom uploads.

## Goal

Build a small ML application that:

- 1. Accepts an image of a cat or dog
- 2. Uses a pre-trained or fine-tuned deep learning model to classify the image (Based on user selection)
- 3. Returns the predicted class and confidence score
- 4. Web UI allowing the pipeline to run from browser

## Installation Instructions

Note! Python 3.13 or newer is required.

- Clone the repositiory to local machine.
```
git clone https://github.com/Alexsoil/MSENSIS-MLE-task
```

- Create a virtual environment and activate it.
```
python -m venv .venv
```
```
source .venv/bin/activate
```

- Install required python modules using pip.
```
pip install -r requirements.txt
```

- Start the API.
```
fastapi run src/model_api.py
```

- Open the Documentation page on your browser at http://127.0.0.1:8000/docs.

- Use the '/upload' endpoint to upload an image.
(Click 'Try it out', select a file through the 'Browse...' section, and finally click 'Execute'. The Response body should show a message upon successful uploading.)

- Use the '/analyze/pretrained' or '/analyze/finetuned' endpoints to classify the uploaded image using the respective model.
(Click 'Try it out', and then 'Execute'. The Response body should show the uploaded image's class, as well as the model's confidence in the prediction.)

- NOTE: The first time you run each model's endpoint, it could take from a few seconds to several minutes to receive an answer, depending on your machine's performance and internet connection.