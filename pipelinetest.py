import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random, os, glob
import matplotlib.pyplot as plt
import tensorflow as tf

dir_path = "E:\Praneeth\Python\_trashidentify\_allmaterials"
img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))
len(img_list)

data = tf.keras.utils.image_dataset_from_directory('E:\Praneeth\Python\_trashidentify\_allmaterials')

class_names = data.class_names
labels = {class_name: index for index, class_name in enumerate(class_names)}
print(labels)

labels = dict((v, k) for k, v in labels.items())
print(labels)

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.msubplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


model = Sequential()
    # Convolution blocks

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
    # model.add(SpatialDropout2D(0.5)) # No accuracy

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
    # model.add(SpatialDropout2D(0.5))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

    # Classification layers
model.add(Flatten())

model.add(Dense(64, activation='relu'))
    # model.add(SpatialDropout2D(0.5))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))

filepath = "trained_model.keras"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
optimizer='adam',
metrics=['acc'])

logdir='loghist'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=1, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['acc'], color='teal', label='acc')
plt.plot(hist.history['val_acc'], color='orange', label='val_acc')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

from keras.preprocessing import image

img_path = "E:\Praneeth\Python\_trashidentify\_allmaterials\metal\metal25.jpg"

img = image.load_img(img_path, target_size=(256, 256))
img = image.img_to_array(img, dtype=np.uint8)
img = np.array(img) / 255.0

plt.title("Loaded Image")
plt.axis('off')
plt.imshow(img.squeeze())

p = model.predict(img[np.newaxis, ...])

# print("Predicted shape",p.shape)
print("Maximum Probability: ", np.max(p[0], axis=-1))
predicted_class = labels[np.argmax(p[0], axis=-1)]
print("Classified:", predicted_class)
