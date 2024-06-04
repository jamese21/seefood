import os
from os.path import join

from IPython.display import Image, display
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def decode_predictions(preds, top=5, class_list_path='../input/resnet50/imagenet_class_index.json'):
  """Decodes the prediction of an ImageNet model.
  Arguments:
      preds: Numpy tensor encoding a batch of predictions.
      top: integer, how many top-guesses to return.
      class_list_path: Path to the canonical imagenet_class_index.json file
  Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.
  Raises:
      ValueError: in case of invalid shape of the `pred` array
          (must be 2D).
  """
  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  CLASS_INDEX = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results

# Get paths to images
hotdog_paths = ["../data/train/hot_dog/1000288.jpg", "../data/train/hot_dog/127117.jpg"]
not_hotdog_paths = ["../data/train/not_hot_dog/823536.jpg", "../data/train/not_hot_dog/99890.jpg"]
img_paths = hotdog_paths+not_hotdog_paths
print(img_paths)

# Prep images
img_size = 224
imgs = [load_img(img_path, target_size=(img_size, img_size)) for img_path in img_paths]
img_array = np.array([img_to_array(img) for img in imgs])
test_data = preprocess_input(img_array)

# Train
my_model = ResNet50(weights=None)
preds = my_model.predict(test_data)
print(preds)

# most_likely_labels = decode_predictions(preds, top=3)