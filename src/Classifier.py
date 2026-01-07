from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

class Classifier():
    r"""
        This class is desined to classify images using the pretrained transformer model
        ViT or a finetuned version of it.

        Args:
            model_name (`str`, defaults to `vit-base-patch16-224`):
                The path to the finetuned model of ViT, either locally or on the HuggingFace Hub.
                If left blank the pretrained ViT model is loaded.
    """

    def __init__(self, model_name='google/vit-base-patch16-224'):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained(model_name)
    
    def classify(self, image_path=''):
        r"""
        Classifies an image using the transformer model loaded during initialization.

        Args:
            image_path (`str`, defaults to `''`):
                The path to the image to be classified.
        
        Returns:
            category (`str`):
                The label of the associated class the image has been classified as.
            confidence (`float`):
                The classification confidence, sourced as the softmax value
                of the image class on final layer of the classifier.
    """
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