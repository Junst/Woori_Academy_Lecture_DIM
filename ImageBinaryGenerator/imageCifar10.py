# 구현한 데이터로 CIFAR10 방식으로 적용해보기

import numpy as np
import os
import matplotlib.pyplot as plt

# data directory
input = os.getcwd() + "/data/data.bin"
imageSize = 32
labelSize = 1
imageDepth = 3
debugEncodedImage = True


# show given image on the window for debug
def showImage(r, g, b):
    temp = []
    for i in range(len(r)):
        temp.append(r[i])
        temp.append(g[i])
        temp.append(b[i])
    show = np.array(temp).reshape(imageSize, imageSize, imageDepth)
    plt.imshow(show, interpolation='nearest')
    plt.show()


def load_one_data(data, offset):
    eachColorSize = imageSize * imageSize
    offset = labelSize + (labelSize + eachColorSize * 3) * offset

    rgb = []
    for i in range(3):
        color = eachColorSize * i
        rgbData = data[offset + color: offset + color + eachColorSize]
        rgb.append(rgbData)

    # showImage(rgb[0], rgb[1], rgb[2])

    retData = np.array([rgb[0], rgb[1], rgb[2]])
    retData = retData.reshape(32, 32, 3)

    return retData, data[offset - 1]


def load_batch(path, num_train_samples):
    data = np.fromfile(path, dtype='u1')

    retData = []
    retLabels = []

    for i in range(num_train_samples):
        d, l = load_one_data(data, i)
        retData.append(d)
        retLabels.append(l)

    retData = np.array(retData)
    retLabels = np.array(retLabels)

    print(retData.shape)

    return retData, retLabels


def load_data():
    dirname = 'downloads'
    path = os.path.join(dirname, "data2.bin")

    # 저는 총 671개의 데이터로 진행했습니다.
    num_train_samples = 671

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    x_train, y_train = load_batch(path, num_train_samples)

    y_train = np.reshape(y_train, (len(y_train), 1))

    return x_train, y_train

import tensorflow
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

batch_size = 32
num_classes = 10
epochs = 50
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'lion_tiger_cifar10_trained_model.h5'

# The data, split between train and test sets:yr
x_train, y_train = load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# Convert class vectors to binary class matrices.
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_train /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=671)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# 임의 데이터 뽑아서 결과 확인하기
# %matplotlib inline
import math
import matplotlib.pyplot as plt
import random

# Get one and predict
r = random.randint(0, 671 - 1)
input_val = x_train[r:r+1]
output_val = model.predict(input_val)

print(r, "Prediction : ", np.argmax(output_val))
# Selected sample showing
print(input_val[0].shape)
plt.imshow(
    input_val[0],
)
plt.show()