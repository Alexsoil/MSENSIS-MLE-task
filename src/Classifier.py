from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

class Classifier():
    r"""
        Class info
    """

    def __init__(self, model_name='google/vit-base-patch16-224'):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
    
    def classify(self, image_path=''):
        try:
            image_rgb = Image.open(image_path).convert('RGB')

            inputs = self.processor(images=image_rgb, return_tensors='pt')
            outputs = self.model(**inputs)

            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            confidence = logits.softmax(-1)[0, predicted_class_idx]
            category = self.model.config.id2label[predicted_class_idx]
            return category, confidence
        
        except FileNotFoundError as err:
            print(f'Image path incorrect: {err}')