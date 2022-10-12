# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

### Functions ###

# Images Visualization. 
def plot_images(x_df, y_df, y_clases, img_numer, number_images, color_map):
    # create figure
    fig_cols = 5
    color_map = ('gray' if color_map == 1 else 'viridis')
    fig_rows = -(-number_images//fig_cols)
    fig = plt.figure(figsize=(2*fig_cols, 2*fig_rows), linewidth=5, edgecolor="black")

    acc = 1
    for i in range(img_numer, img_numer+number_images):
        # Adds a subplot at the 1st position
        fig.add_subplot(-(-number_images//fig_cols), fig_cols, acc)

        # showing image
        plt.imshow(x_df[i], interpolation='spline16', cmap=color_map)
        plt.axis('off')
        plt.title(y_clases[y_df[i][0]])
        acc +=1

    plt.show()

######

# Load the data
(x_train, y_train),(x_test,y_test)= cifar10.load_data()

# Target variable Classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', ' dog', 'frog', 'horse', 'ship', 'trunk']

# convert x_train and x_test from BGR to Grayscale
x_train_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_train])
x_test_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_test])

# print first 10 images 
plot_images(x_train_gray, y_train, classes, 0, 10, 1)

# Data Normalization
x_train_gray, x_test_gray = x_train_gray/255, x_test_gray/255

# Label preprocessing
one_hot_encoder  = OneHotEncoder(sparse=False)
one_hot_encoder.fit(y_train)

y_train_ohe = one_hot_encoder.transform(y_train)
y_test_ohe = one_hot_encoder.transform(y_test)


# Construction the CNN
# reshape grayscale data to fit the Conv2D format
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], x_train_gray.shape[1], x_train_gray.shape[2], 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], x_test_gray.shape[1], x_test_gray.shape[2], 1)
# neural net input shape 
input_shape = (x_train_gray.shape[1], x_train_gray.shape[2], 1)

# Model creation
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiling model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'])

# stop the training process once the loss value reaches its minimum point
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

history_gray = model.fit(x_train_gray, y_train_ohe, epochs=20, batch_size=32, validation_data=(x_test_gray, y_test_ohe), callbacks=[es])

# Saving/loading model
model.save('CNN_CIFAR.h5')
model = load_model('CNN_CIFAR.h5')

# Model Evaluation
model.evaluate(x_test_gray, y_test_ohe)

plt.plot(history_gray.history['acc'], label='acc')
plt.plot(history_gray.history['val_acc'], label='val_acc')
plt.legend(loc='upper left')
plt.show()

plt.plot(history_gray.history['loss'], label='loss')
plt.plot(history_gray.history['val_loss'], label='val_loss')
plt.legend(loc='upper left')
plt.show()


# Predicting Data
predictions = model.predict(x_test_gray)
predictions = one_hot_encoder.inverse_transform(predictions)
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(9,9))
sns.heatmap(cm, cbar=False, xticklabels=classes, yticklabels=classes, fmt='d', annot=True, cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# return to format to visualize the images
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], x_test_gray.shape[1], x_test_gray.shape[2])
y_test = y_test.astype(int)
predictions = predictions.astype(int)


fig, axes = plt.subplots(ncols=7, nrows=3, sharex=False,
    sharey=True, figsize=(17, 8))
index = 0
for i in range(3):
    for j in range(7):
        axes[i,j].set_title('actual:' + classes[y_test[index][0]] + '\n' 
                            + 'predicted:' + classes[predictions[index][0]])
        axes[i,j].imshow(x_test_gray[index], cmap='gray')
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)
        index += 1
plt.show()