# Training of Resnet18 with the original architecture (no preactivation)
import math
import asn
from asn.resnets.resnet_asn import ResnetBuilder
import os
import keras
from asn.utils import save_pickle

from keras.optimizers import SGD
from keras import callbacks
from keras.models import load_model
from keras.utils import multi_gpu_model
import numpy as np

from asn.training.preprocess_crop import load_and_crop_img
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import LearningRateScheduler


def step_decay(epoch):
    # based on the fb implementation, https://github.com/mlosch/fb.resnet.torch/blob/master/train.lua
    base_lr = 0.1
    return base_lr * math.pow(0.1, math.floor((epoch - 1) / 30))

# Swap the loading function to introduce random cropping with Monkey patch.
keras.preprocessing.keras_preprocessing.image.iterator.load_img = load_and_crop_img

directory = '/mnt/Googolplex/imagenet/'
epochs = 90
batch_size = 240
train_dir = '/mnt/Googolplex/imagenet/train/'
val_dir = '/mnt/Googolplex/imagenet/val/'
weight_dir = '/mnt/Googolplex/imagenet/weights/'
train_datagen = ImageDataGenerator(rescale=1/255., horizontal_flip=True)

train_img_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        interpolation='lanczos:random',  # <--------- random crop
        shuffle=True)

#tmp = train_img_generator.next()

validate_datagen = ImageDataGenerator(rescale=1/255.)

validate_img_generator = validate_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    interpolation='lanczos:center',  # <--------- center crop
    shuffle=False)

# Prepare callbacks:
saveFile = weight_dir + 'asnweights.{epoch:02d}-{val_acc:.2f}.h5'
checkpointer = callbacks.ModelCheckpoint(filepath=saveFile, monitor='val_acc', verbose=1, save_best_only=True)
lr_reducer = LearningRateScheduler(step_decay)
# and the optimizer
sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)

# Check if the directory is empty, training was started before...
files =[filename for filename in os.listdir(weight_dir) if filename.startswith("asnweights.")]
if len(files) > 0: # Continue training where you left off

    epochs_done = np.zeros(len(files))
    for idx, file in enumerate(files):
        epochs_done[idx] = int(file.split('-')[0][-2:])

    last_epoch_idx = np.where(epochs_done == np.max(epochs_done))[0][0]
    last_ckpt = files[last_epoch_idx]
    print('Resuming training at epoch ' + str(epochs_done[last_epoch_idx]))
    epoch_start = int(epochs_done[last_epoch_idx])
    model = load_model(weight_dir + last_ckpt, custom_objects={'ASNTransfer': asn.training.ASNTransfer})
    # Fill in the best model from memory
    checkpointer.best = float(last_ckpt[-7:-3])

else:

    model = ResnetBuilder.build_resnet_18_v0((224,224,3),1000) #https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    epoch_start = 0

model_gpu = multi_gpu_model(model,2)
model_gpu.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model_gpu.__setattr__('callback_model',model) # This makes sure that the checkpoint is saving the template model (incl. weights & optimizer state) and not the GPU version.

if epoch_start > 0:
    model_gpu._make_train_function() # to make the weights for the optimizer
    model_gpu.optimizer.set_weights(model.optimizer.get_weights()) # load in the optimizer state

history = model_gpu.fit_generator(train_img_generator,
        steps_per_epoch=1281167//batch_size,  # Total amount of images 1,281,167 # samples_per_epoch=1200012,
        epochs=epochs,
        validation_data=validate_img_generator,
        validation_steps=50000//batch_size,
        callbacks=[checkpointer,lr_reducer], initial_epoch=epoch_start, use_multiprocessing=False, nb_worker=6)



save_pickle(history.history,'/mnt/Googolplex/imagenet/weights/history')
# Save weights
model.save_weights('/mnt/Googolplex/imagenet/weights/FinalWeights.h5')

model.save('/mnt/Googolplex/imagenet/weights/Single_model.h5')

model_gpu.save('/mnt/Googolplex/imagenet/weights/GPU_model.h5')

model_json = model.to_json()
with open("/media/lynn/Googolplex/imagenet/weights/model.json",
                  "w") as json_file:
            json_file.write(model_json)

print("Done.")
#
#

