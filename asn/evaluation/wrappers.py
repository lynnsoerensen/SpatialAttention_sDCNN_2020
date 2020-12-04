import numpy as np
from keras.utils import multi_gpu_model
from asn.evaluation.metrics import top_k_categorical_accuracy_asn
from asn.evaluation.generators import TimeReplication

def predict_generator(model_test, X_test, y_test, time_steps, batch_size, generator = None, steps= None, num_gpus= None):
    if steps == None:
        steps = int(np.ceil(X_test.shape[0]/batch_size))
    if num_gpus != None:
        print("Predicting on " + str(num_gpus) + ' GPUs')
        model_test = multi_gpu_model(model_test, gpus=num_gpus)
        model_test.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_test.predict_generator(generator=generator(X_test, y_test, time_steps, batch_size=batch_size),
                                               steps=steps, verbose=True)

def evaluate_predictions(predictions,y_test,start_eval, stop_eval, eval_mode, num_targets=1):
    y_pred = np.mean(predictions[:, start_eval:stop_eval, :], axis=1)
    if eval_mode =='timesteps':
        print('Evaluating timestep-wise accuracy')
        prediction_max = np.argmax(predictions[:, start_eval:stop_eval, :], axis=2)
        eval_duration = stop_eval-start_eval
        target = np.expand_dims(np.argmax(y_test, axis=1), axis=1)
        target = np.repeat(target, eval_duration, axis=1)
        accuracy = np.sum(prediction_max == target)/ (target.shape[0]*target.shape[1])

    elif eval_mode=='binary-timesteps':
        print('Evaluating binary accuracy')
        predictions_binary = np.round(predictions[:, start_eval:stop_eval, :])
        eval_duration = stop_eval - start_eval
        target = np.expand_dims(y_test, axis=1)
        target = np.repeat(target, eval_duration, axis=1)
        accuracy = np.sum(predictions_binary == target) / (target.shape[0] * target.shape[1])
    else:
        print('Evaluating mean accuracy with top' + str(eval_mode))
        accuracy = top_k_categorical_accuracy_asn(y_pred, y_test, k=eval_mode, num_targets=num_targets)
    return accuracy

def evaluate_generator(model_test, X_test, y_test, time_steps, batch_size, generator = None, steps= None,
                       start_eval= None, stop_eval=None, num_gpus=None, eval_mode='timesteps', num_targets=1):
    """ This function is used to evaluate spiking DNN performance
    model_test: spiking/digital DNN
    X_test: test images (n, n_row,n_col,channel)
    y_test: target label (n,n_classes)
    time_steps: time_steps of the model_test
    batch_size: How many images to process at a time?
    generator: cf. asn.generators, default is TimeReplication
    steps: Total number of steps (batches of samples) to yield from generator before stopping. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
    start_eval: Starting time point of accuracy assessment of prediction time courses, default:100 ms
    stop_eval: Stoping time point. default is the end of the prediction time course
    num_gpus: number of GPUs to evaluate on
    eval_mode: evaluate predictions for every time steps or based on average, then give k=1

    acc: mean accuracy over all evaluate time points (start_eval:stop_eval)
    predictions: prediction time course of sDNN
    """
    if steps == None:
        if isinstance(X_test,list):
            steps = int(np.ceil(X_test[0].shape[0] / batch_size))
        else:
            steps = int(np.ceil(X_test.shape[0]/batch_size))
    if stop_eval==None:
        stop_eval=time_steps
    if start_eval == None:
        start_eval = 200 # the first 100 ms or so are dominated by transient activity
    if generator == None:
        generator=TimeReplication

    predictions = predict_generator(model_test, X_test, y_test, time_steps, batch_size,
                                    generator=generator, steps=steps, num_gpus=num_gpus)
    if len(predictions) > 1:
        accuracy = evaluate_predictions(predictions[0],y_test[0:batch_size*steps],start_eval, stop_eval, eval_mode, num_targets=num_targets)

    return [accuracy, predictions]
