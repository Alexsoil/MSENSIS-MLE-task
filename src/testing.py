from Classifier import Classifier

# classifier = Classifier()
classifier = Classifier(model_name='Alexsoil/ViT-cats-dogs')
category, confidence= classifier.classify('cat_image.png')
print(f'Image Class: {category} | Confidence: {confidence * 100:.2f}%')