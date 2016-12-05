"""
This program is used to recognize the human frances among 15 persons using Yale Face Dataset.
"""

import cv2
import numpy as np
import os
from skimage import io
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# the paths for the face images
image_path = []
for file_name in os.listdir("C:\\data\\Yale_faces"):
    image_path.append(os.path.join("C:\\data\\Yale_faces", file_name))

imageData = []
imageLabels = []

# read face images and class labels
for img in image_path:
    imgRead = io.imread(img, as_grey=True)
    imageData.append(imgRead) 
    labelRead = int(os.path.split(img)[1].split(".")[0].replace("subject", "")) - 1
    imageLabels.append(labelRead)

# face detector
faceDetectClassifier = cv2.CascadeClassifier('C:\\opencv3\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

# croped face images
imageDataFin = []
for image in imageData:
    facePoints = faceDetectClassifier.detectMultiScale(image)
    x,y = facePoints[0][:2]
    cropped = image[y: y + 150, x: x + 150]
    resized_image = cv2.resize(cropped, (50, 50)) 
    imageDataFin.append(resized_image)

# split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(np.array(imageDataFin), np.array(imageLabels), train_size=0.9, 
                                                    random_state = 123)

# fix the random seed for reproducibility
seed = 123
np.random.seed(seed)

# configuration
batch_size = 32
nb_classes = 15
nb_epoch = 50
# image dimensions
img_rows, img_cols = 50, 50
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# resdhape the data matrix
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# scale the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one-hot encoding for the class label
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# define the convolutional neural network model
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# compile the CNN model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# train the CNN model
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

# evaluation
score = model.evaluate(X_test, Y_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])

# find out which images are classified wrongly
predicted_classes = model.predict_classes(X_test)
correct_classified_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_classified_indices = np.nonzero(predicted_classes != y_test)[0]
if not incorrect_classified_indices:
    print('\nAll test samples are correctly recognized.')
else:
    print('The incorrect indices are:', incorrect_classified_indices) 
    
    
