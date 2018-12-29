import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import compute_class_weight
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras.applications import MobileNetV2

#conv_base = VGG19(include_top=False, weights = None)
#conv_base.summary()

data_gen = ImageDataGenerator(rescale=0.5, validation_split=0.2)
test_gen = ImageDataGenerator(rescale=0.5)

batch_size = 16

train_gen = data_gen.flow_from_directory('datayo/sc5', target_size=(224,224), batch_size=16, class_mode='categorical', subset='validation')
test_gen = test_gen.flow_from_directory('datayoMKII/sc5 test', target_size=(224,224), batch_size=16, class_mode='categorical', shuffle=False)

yo = train_gen.classes
d = {}

weights = compute_class_weight(class_weight='balanced', classes=np.unique(yo), y = yo)

for i in range(len(weights)):
    d[i] = weights[i]

print(d)

conv_full = MobileNetV2(include_top=True, weights=None, classes=24)

conv_full.summary()

conv_full.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['categorical_accuracy',  'top_k_categorical_accuracy'])

stepsxepoch = len(train_gen) // batch_size
stepsxepoch1 = len(test_gen) // batch_size

conv_full.load_weights('boats4.h5')
conv_full.fit_generator(train_gen, epochs=50, steps_per_epoch=stepsxepoch, validation_steps=20, class_weight=d)
h = conv_full.predict_generator(test_gen, steps = len(test_gen))
f = conv_full.predict_generator(train_gen, steps = len(train_gen))

Y_pred = h
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(test_gen.classes, y_pred))
confmatr = confusion_matrix(test_gen.classes, y_pred)
print('Classification Report')
print(classification_report(test_gen.classes, y_pred))
classrep = classification_report(test_gen.classes, y_pred)


g = conv_full.evaluate_generator(train_gen, steps=200)
l = conv_full.evaluate_generator(test_gen, steps=200)
print(g, l)