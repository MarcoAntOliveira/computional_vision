from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (BatchNormalization, Lambda, Input, Dense,
                                     Conv2D, MaxPooling2D, AveragePooling2D,
                                     ZeroPadding2D, Dropout, Flatten, Activation)
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow.keras.backend as K


def beer_net(num_classes):
    input_image = Input(shape=(224,224,3))

    # Exemplo: primeira branch top
    top_conv1 = Conv2D(48, (11,11), strides=(4,4), activation='relu')(input_image)
    top_conv1 = BatchNormalization()(top_conv1)
    top_conv1 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(top_conv1)

    # ... resto igual ao seu modelo ...

    # Flatten + FC
    flatten = Flatten()(conv_output)
    FC_1 = Dense(1024, activation='relu')(flatten)   # reduzido para estabilidade
    FC_1 = Dropout(0.5)(FC_1)
    FC_2 = Dense(512, activation='relu')(FC_1)
    FC_2 = Dropout(0.5)(FC_2)
    output = Dense(num_classes, activation='softmax')(FC_2)

    model = Model(inputs=input_image, outputs=output)
    sgd = SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Config
img_rows, img_cols = 224, 224
num_classes = 9
batch_size = 32
nb_epoch = 5

model = beer_net(num_classes)

filepath = 'color_weights.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.3,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train/',
                                                 target_size=(img_rows, img_cols),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('test/',
                                            target_size=(img_rows, img_cols),
                                            batch_size=batch_size,
                                            class_mode='categorical')

model.fit(training_set,
          steps_per_epoch=len(training_set),
          epochs=nb_epoch,
          validation_data=test_set,
          validation_steps=len(test_set),
          callbacks=callbacks_list)

model.save('color_model.h5')
