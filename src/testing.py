from Classifier import Classifier

# classifier = Classifier()
classifier = Classifier(model_name='finished_models/cats_dogs_finetune')
category, confidence= classifier.classify('cat_image.png')
print(f'Image Class: {category} | Confidence: {confidence * 100:.2f}%')