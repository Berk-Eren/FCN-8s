from sklearn.model_selection import train_test_split

from model import model
from settings import TEST_SIZE
from preprocessing import training_images, training_semantic_label


train_img, test_img, train_sem, test_sem = train_test_split(training_images, 
                                                                training_semantic_label, 
                                                                    test_size=TEST_SIZE, 
                                                                        random_state=45)

results = model.fit(train_img, 
                      train_sem, epochs=300, 
                        batch_size=32, 
                          validation_data=(test_img, test_sem))

model.save("model.h5")                                      