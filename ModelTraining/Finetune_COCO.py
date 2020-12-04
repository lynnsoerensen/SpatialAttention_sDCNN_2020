import numpy as np
# import tensorflow as tf
from asn.resnets.resnet_asn import ResnetBuilder
from keras.preprocessing import image
from asn.training.imagenet_utils import decode_predictions
from keras.models import Model
from keras.layers import Dense
from asn.utils import load_pickle, save_pickle
from keras.utils import multi_gpu_model
from asn.training.callbacks import LossHistory_binary, ModelCheckpoint, CyclicLR
from asn.layers.training import ASNTransfer
from asn.evaluation.generators import coco_squared
from sklearn.metrics import f1_score
from keras.callbacks import ReduceLROnPlateau

np.random.seed(3)
# %% Finetune a pre-trained model on two curated COCO datasets

for dataset in ['1_street', '2_food']:
    print(dataset)
    freezing = -2
    trained = True
    spike_test = False
    multi_gpu = True

    # Load in the datasets.
    img_dir = '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/Datasets/' + dataset + '/images/'
    img_dir_train = img_dir + '/train2017_multi_radial/'
    img_dir_test = img_dir + '/val2017_single_radial/'

    dataset_base = load_pickle(
        '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/Datasets/' + dataset+ '/train2017_multi_radial.pickle')
    dataset_test = load_pickle(
        '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/Datasets/' + dataset + '/val2017_single_radial.pickle')
    gen_test = coco_squared(dataset_test, batch_size=len(dataset_test['x_ids']) - 1, img_dir=img_dir_test)

    learning_mode = 'constant'
    optimizer = 'Adam'
    optimizer_func = eval('optimizers.' + optimizer)
    lr = 1e-4
    training_name = '_' + optimizer + '_' + learning_mode + '_' + str(lr)
    weight_path = '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/ModelTraining/' + dataset + '/weights' + training_name

    batch_size = 200
    epochs = 100
    (x_test, y_test) = next(gen_test)
    num_classes = len(dataset_base['categories'])

    # %%
    if trained == True:
        from keras.models import model_from_json

        json_file = open(
            '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/ModelTraining/' + dataset + "/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        modelFT = model_from_json(loaded_model_json, custom_objects={'ASNTransfer': ASNTransfer})
        modelFT.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr), metrics=['binary_accuracy'])
        modelFT.load_weights(weight_path + '.h5')

        scores = modelFT.evaluate(x_test, y_test)

    else:
        modelBase = ResnetBuilder.build_resnet_18_v0((224, 224, 3), 1000)
        modelBase.summary()
        weight_dir = '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/ModelTraining/ImageNet/'
        weight_path_resnet = weight_dir + 'FinalWeights.h5'

        modelBase.load_weights(weight_path_resnet)  # , by_name=True)

        img_path = '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/ModelTraining/ImageNet/Elephant.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255
        x = np.expand_dims(x, axis=0)

        preds = modelBase.predict(x)

        print('Predicted:', decode_predictions(preds))
        # %% Replace the top layer (10 units instead of 1000 & sigmoid instead of softmax)
        temp = Dense(num_classes, activation='sigmoid')(modelBase.get_layer('flatten_1').output)
        modelFT = Model(inputs=modelBase.input, outputs=temp)

        modelFT.summary()
        # %% Freeze the rest
        modelFT.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr), metrics=['binary_accuracy'])
        if multi_gpu == True:
            model_gpus = multi_gpu_model(modelFT, gpus=2)
            for l in range(len(model_gpus.layers[:freezing])):
                model_gpus.layers[l].trainable = False

            model_gpus.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                               metrics=['binary_accuracy'])
        else:
            for l in range(len(modelFT.layers[:freezing])):
                modelFT.layers[l].trainable = False

            modelFT.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                            metrics=['binary_accuracy'])
        # %%
        checkpointer = ModelCheckpoint(filepath=weight_path + '.h5', monitor='val_loss', verbose=1,
                                       save_best_only=True)

        history_call = LossHistory_binary()
        if learning_mode == 'clr':
            lr_scheduler = CyclicLR(base_lr=lr, max_lr=1e-4,
                                    step_size=264., mode='exp_range')
        elif learning_mode == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                             patience=5, min_lr=1e-6)
        else:
            lr_scheduler = None

        # %% Make a train & val set
        # np.random.seed(3)
        proportion = 0.2
        idx = np.arange(len(dataset_base['x_ids']))
        np.random.shuffle(idx)
        dataset_val = {}
        dataset_val['x_ids'] = dataset_base['x_ids'][idx][:int(np.ceil(len(dataset_base['x_ids']) * proportion))]
        dataset_val['y'] = dataset_base['y'][idx][:int(np.ceil(len(dataset_base['x_ids']) * proportion))]

        dataset_train = {}
        dataset_train['x_ids'] = dataset_base['x_ids'][idx][int(np.ceil(len(dataset_base['x_ids']) * proportion)):]
        dataset_train['y'] = dataset_base['y'][idx][int(np.ceil(len(dataset_base['x_ids']) * proportion)):]

        gen = coco_squared(dataset_val, batch_size=len(dataset_val['x_ids']) - 1, img_dir=img_dir_train)
        (x_val, y_val) = next(gen)
        if lr_scheduler is not None:
            callbacks = [checkpointer, history_call, lr_scheduler]
        else:
            callbacks = [checkpointer, history_call]

        if multi_gpu == True:
            history = model_gpus.fit_generator(
                coco_squared(dataset_train, batch_size=batch_size, img_dir=img_dir_train),
                steps_per_epoch=int(np.ceil(len(dataset_train['x_ids']) / batch_size)),
                epochs=epochs, validation_data=(x_val, y_val), shuffle=True, callbacks=callbacks)
        else:
            history = modelFT.fit_generator(
                coco_squared(dataset_train, batch_size=batch_size, img_dir=img_dir_train),
                steps_per_epoch=int(np.ceil(len(dataset_train['x_ids']) / batch_size)),
                epochs=epochs, validation_data=(x_val, y_val), shuffle=True, callbacks=callbacks)

        save_pickle(history.history,
                    '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/ModelTraining/' + dataset + '/history' + training_name)
        # Save model architecture
        model_json = modelFT.to_json()
        with open('/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/ModelTraining/' + dataset  + "/model.json",
                  "w") as json_file:
            json_file.write(model_json)
        # Save weights
        modelFT.save_weights(weight_path + '.h5')

        # %% Evaluate on the test set
        scores = modelFT.evaluate(x_test, y_test)
        preds = modelFT.predict(x_test)
        # %%
        print('Accuracy analog: ' + str(scores[1]))
        tracker = {}

        tracker["batch_size"] = batch_size
        tracker["epochs"] = epochs
        tracker["weights"] = weight_path + '.h5'
        tracker["accuracy_analog_center"] = scores[1]
        num_samples = 300
        tracker['F1_analog_centre'] = f1_score(y_test[:num_samples, :], preds[:num_samples, :] > 0.5,
                                               average='weighted')
        shuffle_idx = np.arange(y_test.shape[1])
        n_permutations = 10000
        permutation_scores = np.zeros(n_permutations)
        for p in range(n_permutations):
            np.random.shuffle(shuffle_idx)
            permutation_scores[p] = f1_score(y_test[:num_samples, :], preds[:num_samples, shuffle_idx] > 0.5,
                                             average='weighted')

        tracker['F1_permutationScores'] = permutation_scores

        save_pickle(tracker,
                    '/mnt/Googolplex/PycharmProjects/SpatialAttention_sDCNN_2020/ModelTraining/' + dataset + '/tracker' + training_name)



