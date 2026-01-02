from Classifier import Classifier

classifier = Classifier()
category, confidence= classifier.classify('cat_image.png')
print(f'Image Class: {category} | Confidence: {confidence:.4f}')