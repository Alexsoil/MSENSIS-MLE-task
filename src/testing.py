from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

image_path = 'akita_image.png'

image = Image.open(image_path).convert('RGB')
image.show()

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors='pt')
outputs = model(**inputs)

logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

print("Predicted class:", model.config.id2label[predicted_class_idx])


