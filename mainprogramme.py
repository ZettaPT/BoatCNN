import os
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False)
conv_base.summary(line_length=200)

base_dir = 'datayo'
train_dir = 'datayo/sc5'
test_dir = 'datayo/sc5-2013-Mar-Apr-Test-20130412'

datagen = ImageDataGenerator()
batch_size = 24

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 8, 8, 512))
    labels = np.zeros(shape=())
    generator = datagen.flow_from_directory(
        directory,
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 4774)
# validation_features, validation_labels = extract_features(validation_dir, 1000)
# test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (4774,  8 * 8 * 512))
print(train_features)
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=32768))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.2)


# if __name__ == "__main__":


