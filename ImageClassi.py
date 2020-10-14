import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.60
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

from tensorflow.keras.layers import Dense,Flatten,Input,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

IMAGE_SIZE = [510,510]
train_path = r'C:\Users\Jeev\Untitled Folder\data\train'
test_path =  r'C:\Users\Jeev\Untitled Folder\data\test'

mod = InceptionResNetV2(include_top=False, weights='imagenet', pooling='max', input_shape=(510,510,3))
for layers in mod.layers:
    layers.trainable = False
for layers in mod.layers[-5:]:
    layers.trainable = True

folders = glob(r'C:\Users\Jeev\Untitled Folder\data\train\*')

model = Flatten()(mod.output)

model = Dense(4, activation='softmax')(model)

model = Model(inputs = mod.input,outputs=model)
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='reflect',
                                   data_format='channels_last',
                                   brightness_range=[0.5, 1.5])
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=64,
                                                    class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(test_path,
                                                              target_size=IMAGE_SIZE,
                                                              batch_size=32,
                                                              class_mode='categorical')


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_generator,validation_data=validation_generator, epochs=10, steps_per_epoch=len(train_generator),
                     validation_steps=len(validation_generator))
tf.keras.models.save_model(
    model, r'C:\Users\Jeev\Untitled Folder\data', overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None
)