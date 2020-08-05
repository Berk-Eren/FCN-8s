import os

PATH = "<path_to_kitti_dataset>/data_semantics"

if not PATH:
    raise NameError("Please set the PATH in the {} file".format(os.getcwd() + "/settings.py"))

#Training and Test image path
TRAINIG_IMAGE_PATH = PATH + "training/image_2/"
TRAINING_SEMANTIC_PATH = PATH + "training/semantic_rgb/"
TESTING_PATH = PATH + "testing"

assert (os.path.exists(TRAINIG_IMAGE_PATH) 
        and os.path.exists(TRAINING_SEMANTIC_PATH) 
         and os.path.exists(TESTING_PATH)) == True

#Number of classes for semantics.
NUM_CLASSES = 35

#I have set the input image size to 125, 125 in gray scale.
INPUT_SHAPE = (125, 125, 1)

#Test size for the validatition.
TEST_SIZE=0.2
