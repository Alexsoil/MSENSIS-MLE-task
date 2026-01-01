from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_path = 'akita_image.png'

image = Image.open(image_path).convert('RGB')
# image.show()

model_name = 'google/vit-base-patch16-224'

processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

inputs = processor(images=image, return_tensors='pt')
outputs = model(**inputs)

logits = outputs.logits
print(logits)
predicted_class_idx = logits.argmax(-1).item()

print("Predicted class:", model.config.id2label[predicted_class_idx])
