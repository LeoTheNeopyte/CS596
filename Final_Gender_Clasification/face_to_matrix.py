import os, sys
import numpy as np
import PIL.Image as Image
from sklearn import svm
import matplotlib.pyplot as plt

def read_images(path, id, sz=None):
    c = id
    y = []
    for pathname, dirnames, filenames in os.walk(path):
            im_number = len(filenames)
            X = np.empty((im_number,80,53),dtype="float32")
            counter = 0
            for file in filenames:
                try:
                    im_location = os.path.join(pathname, file)
                    #print("image path {} ".format(im_location))
                    im = Image.open(im_location)
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X[counter,:,:] = np.asarray(im,dtype="float32")
                    y.append(c)  
                except IOError as e:
                    print ("I/O error({}): {}".format(e.errno, e.strerror))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
                      #c = c+1
                counter += 1

    return [X,y]
resize = (53,80)
[male_test, y_male_test] = read_images("./faces/male/test_set",1,resize)
[male_training, y_male_training] = read_images("./faces/male/training_set",1,resize)
[female_test, y_female_test] = read_images("./faces/female/test_set",-1,resize) 
[female_training, y_female_training] = read_images("./faces/female/training_set",-1,resize)
train_images = []
train_labels = []
test_images = []
test_labels = []
train_images.append(male_training)
train_images.append(female_training)
train_images = np.concatenate(train_images)
train_labels.append(y_male_training)
train_labels.append(y_female_training)
train_labels = np.concatenate(train_labels)


test_images.append(male_test)
test_images.append(female_test)
test_images = np.concatenate(test_images)
test_labels.append(y_male_test)
test_labels.append(y_female_test)
test_labels = np.concatenate(test_labels)

n = 474
S = np.random.permutation(n)
train_images= train_images[S,:,:]
train_labels= train_labels[S]

n2 = 472
S2 = np.random.permutation(n2)
test_images= test_images[S2,:,:]
test_labels= test_labels[S2]

#print("male test: {}\n shape : {}".format(test_images.shape,test_labels))
data_dict = {
    'images_train': train_images,
    'labels_train': train_labels,
    'images_test': test_images,
    'labels_test': test_labels,
  }
np.save("faces",data_dict)