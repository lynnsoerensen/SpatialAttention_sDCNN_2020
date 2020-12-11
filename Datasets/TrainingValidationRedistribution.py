"""
In this script, I created a version of coco training and validation set in which a random subsample of 20 % of the
training set will be moved to be part of the validation set

20.02.2019
"""

from pycocotools.coco import COCO
import numpy as np
import json
import os
from copy import deepcopy
dataDir='/media/lynn/Googolplex/coco/'
dataTypes=['val2017','train2017']
redistribution_proportion = 0.2

makeImageDir = False

annFileTrain='{}/annotations/instances_train2017.json'.format(dataDir)
annFileVal='{}/annotations/instances_val2017.json'.format(dataDir)

cocoTrain=COCO(annFileTrain)
cocoVal=COCO(annFileVal)

# Make pseudo-random selection
np.random.seed(5)

all_train = cocoTrain.getImgIds()

idx = np.arange(len(cocoTrain.imgs))
np.random.shuffle(idx)

idx_sel = idx[:np.ceil(redistribution_proportion * len(idx)).astype('int')]

datasetTrain = deepcopy(cocoTrain.dataset)
datasetVal = deepcopy(cocoVal.dataset)

count = 1
all = len(idx_sel)
for i in idx_sel:
    print(str(count) + '/' + str(all))
    img = cocoTrain.dataset['images'][i]
    if makeImageDir == True:
        # move the image file
        os.rename(dataDir+'images/train2017_redistributed/'+img['file_name'],dataDir+'images/val2017_redistributed/'+img['file_name'])

    # reconfigure the annotation files
    datasetVal['images'].append(img)
    for c in range(len(datasetTrain['images'])-1):
        if datasetTrain['images'][c]['id'] == img['id']:
            print('True')
            del datasetTrain['images'][c]
    # add annotations to validation set
    annList = cocoTrain.imgToAnns[img['id']]
    datasetVal['annotations'].extend(annList)
    # delete it from the training set
    annId = [ann['id'] for ann in annList]
    for c in range(len(datasetTrain['annotations'])-len(annId)):
        if datasetTrain['annotations'][c]['id'] in annId:
            del datasetTrain['annotations'][c]

    if np.divmod(count, 5000)[1] == 0:
        print("Saving...")
        with open("{}annotations/instances_redistributed_train2017.json".format(dataDir),"w") as write_file:
            json.dump(datasetTrain, write_file)

        with open("{}annotations/instances_redistributed_val2017.json".format(dataDir),"w") as write_file:
            json.dump(datasetVal, write_file)

    count = count + 1

print("Final saving...")
with open("{}annotations/instances_redistributed_train2017.json".format(dataDir),"w") as write_file:
    json.dump(datasetTrain, write_file)

with open("{}annotations/instances_redistributed_val2017.json".format(dataDir),"w") as write_file:
    json.dump(datasetVal, write_file)