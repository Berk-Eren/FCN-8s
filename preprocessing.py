import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

from lookup import color_class
from settings import INPUT_SHAPE, TRAINIG_IMAGE_PATH, \
                        TRAINING_SEMANTIC_PATH, NUM_CLASSES

training_images = []
training_semantic = []

width_list = []
height_list = []

is_grey = False if INPUT_SHAPE[-1]>1 else True

print("Preprocessing is started.")
#---------------------------------#

#Each image in TRAINING_IMAGE_PATH are being resized and appended to a list.
for count, img in enumerate(os.listdir(TRAINIG_IMAGE_PATH)):
    if (count+1)%20==0:
      print("{}. image is being processed.".format(count+1))
    img_path = os.path.join(TRAINIG_IMAGE_PATH, img)
    sem_path = os.path.join(TRAINING_SEMANTIC_PATH, img)

    img = imread(img_path, as_gray=is_grey)
    sem = imread(sem_path)
    
    img = cv2.resize(img, dsize=INPUT_SHAPE[:-1], interpolation=cv2.INTER_NEAREST)
    sem = cv2.resize(sem, dsize=INPUT_SHAPE[:-1], interpolation=cv2.INTER_NEAREST)
    
    training_images.append(img)
    training_semantic.append(sem)

training_images = np.array(training_images)    
training_semantic = np.array(training_semantic)
    

assert len(training_images) == len(training_semantic)


semantic_label_shape = tuple([len(training_semantic)]) + INPUT_SHAPE[:-1] + tuple([NUM_CLASSES])
training_semantic_label = np.zeros(semantic_label_shape)


#Categorate each semantic image with one-hot encoding.
for img_ind, img in enumerate(training_semantic):
  if (img_ind+1)%20==0:
    print("{}. image is being processed.".format(img_ind+1))
  for row_ind, row in enumerate(img):
      for col_ind, col in enumerate(row):
          pixel_val = col
          class_id = color_class[tuple(pixel_val)]

          one_hot_imp = to_categorical(class_id, num_classes=NUM_CLASSES)

          training_semantic_label[img_ind][row_ind, col_ind] = one_hot_imp.tolist()

#---------------------------------#
print("Preprocessing is done.")          