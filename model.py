
# coding: utf-8

# In[1]:
#imports
import csv
import cv2
import random
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

#parameters used in data augmentation
TRANS_X_RANGE = 100
TRANS_Y_RANGE = 40
TRANS_ANGLE = .3
CORRECTION_FIX_FOR_TURN = .12

#### helper methods ####
def fix_path(original_path, dir_name='data'):
    file_name = original_path.split("/")[-1]
    new_path = '../{}/IMG/'.format(dir_name) + file_name
    return new_path

def brighten_image(image, low_limit=.5, high_limit=1.5):
    image = np.array(image, dtype = np.uint8)
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = np.random.uniform(low=low_limit, high=high_limit)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255] = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def xy_affine_transform(img, angle, bias, threshold):
    x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
    new_angle = angle + ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE
    y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    if (abs(new_angle) + bias) < threshold or abs(new_angle) > 1.:
        return None, None
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0])), new_angle

### data load ####
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for idx, line in enumerate(reader):
        if idx == 0:
            continue
        lines.append(line)
print("done reading example data: {} samples".format(len(lines)))
for line in lines:
    line[0] = fix_path(line[0])
    line[1] = fix_path(line[1])
    line[2] = fix_path(line[2])

recovery_lines = []
with open('../purerecovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for idx, line in enumerate(reader):
        if idx == 0:
            continue
        recovery_lines.append(line)
print("done reading recovery data: {} samples".format(len(recovery_lines)))
for line in recovery_lines:
    line[0] = fix_path(line[0], 'purerecovery')
    line[1] = fix_path(line[1], 'purerecovery')
    line[2] = fix_path(line[2], 'purerecovery')
lines.extend(recovery_lines)

curve_lines = []
with open('../curve2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for idx, line in enumerate(reader):
        if idx == 0:
            continue
        curve_lines.append(line)
print("done reading curve data: {} samples".format(len(curve_lines)))
for line in curve_lines:
    line[0] = fix_path(line[0], 'curve2')
    line[1] = fix_path(line[1], 'curve2')
    line[2] = fix_path(line[2], 'curve2')
lines.extend(curve_lines)

rec_lines = []
with open('../recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for idx, line in enumerate(reader):
        if idx == 0:
            continue
        rec_lines.append(line)
print("done reading recovery data: {} samples".format(len(rec_lines)))
for line in rec_lines:
    line[0] = fix_path(line[0], 'recovery')
    line[1] = fix_path(line[1], 'recovery')
    line[2] = fix_path(line[2], 'recovery')
lines.extend(rec_lines)

bridge_lines = []
with open('../bridge/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for idx, line in enumerate(reader):
        if idx == 0:
            continue
        bridge_lines.append(line)
print("done reading recovery data: {} samples".format(len(bridge_lines)))
for line in bridge_lines:
    line[0] = fix_path(line[0], 'bridge')
    line[1] = fix_path(line[1], 'bridge')
    line[2] = fix_path(line[2], 'bridge')
lines.extend(bridge_lines)
print("total data samples: {}".format(len(lines)))

#split data
train_samples, validation_samples = train_test_split(lines, test_size=0.3)
print("done splitting data into training/validation. training_samples: {} , validation_samples: {}".format(len(train_samples),len(validation_samples)))

#main data augmentation method that generator calls
def add_data(line, camera, bias, correction=.10, validation=False):
    images = []
    measurements = []
    skip_count = 0
    threshold = np.random.uniform()
    angle = float(line[3])
    measurement = angle
    if validation:
        img = cv2.imread(line[0])
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        images.append(image)
        measurements.append(angle)
    else:
        if camera == 'left':
            image = cv2.imread(line[1])
            if angle == 0:
                measurement = angle + correction
            else:
                measurement = angle + correction + CORRECTION_FIX_FOR_TURN
        elif camera == 'right':
            image = cv2.imread(line[2])
            if angle == 0:
                measurement = angle - correction
            else:
                measurement = angle - correction - CORRECTION_FIX_FOR_TURN
        else:
            image = cv2.imread(line[0])
        if (abs(measurement) + bias) < threshold or abs(measurement) > 1.:
            skip_count+=1

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        images.append(image)
        measurements.append(measurement)
        flip_prob = np.random.random()
        if flip_prob > 0.3:
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurement_flipped = -measurement
            measurements.append(measurement_flipped)
        image_trans, angle_trans = xy_affine_transform(image, angle, bias, threshold )
        if image_trans is not None:
            images.append(image_trans)
            measurements.append(angle_trans)
        if abs(measurement) > 0.0 and np.random.randint(2) == 0:
            images.append(brighten_image(image))
            measurements.append(measurement)
            if flip_prob > 0.3:
                images.append(brighten_image(image_flipped))
                measurements.append(measurement_flipped)
    return images, measurements

#generator method used to batch the data for training.
def generator(lines, bias, validation, batch_size=32):
    num_samples = len(lines)
    while 1:
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images = []
            measurements = []
            camera = np.random.choice(['center','left','right'])
            for line in batch_samples:
                ims, angles = add_data(line, camera, bias, validation)
            images.extend(ims)
            measurements.extend(angles)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[ ]:

#Model hyperparams
EPOCHS = 5
runs = 0

#NVIDIA Model definition
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
while(True):
    bias = 1. / (runs + 1.)
    print('bias: {}'.format(bias))
    history_object = model.fit_generator(generator(train_samples, bias, False, batch_size=32), samples_per_epoch=len(train_samples)*2, validation_data=generator(validation_samples, bias, True, batch_size=32), nb_val_samples=len(validation_samples), nb_epoch=1, verbose=1)
    runs+=1
    model.save('model_{}.h5'.format(runs))
    print('done saving model')
    print(history_object.history.keys())
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    if runs >= EPOCHS:
        break
