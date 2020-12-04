import pickle

#  I/O: These have been replaced by joblib.

def save_pickle(obj, name):
    try:
        filename = open(name + ".pickle","wb")
        pickle.dump(obj, filename)
        filename.close()
        return(True)
    except:
        return(False)


def load_pickle(filename):
    return pickle.load(open(filename, "rb"))


def getLayerIndexByName(model, layername):
    """
    from https://stackoverflow.com/questions/50151157/keras-how-to-get-layer-index-when-already-know-layer-name
    """
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
    


