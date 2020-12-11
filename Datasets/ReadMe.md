# COCO Dataset curation & feature selection

This is an overview of how the datasets for all later spatial attention experiments were selected and named.

For this to work, both the general annotations as well as stuff annotations from the COCO database 2017 need to be available and can be downloaded from [here](https://cocodataset.org/#download).

Further dependencies are:[`cocoapi`](https://github.com/cocodataset/cocoapi),[`Deep Gaze II`](https://deepgaze.bethgelab.org/), [`LGN statistics`](https://github.com/irisgroen/LGNstatistics/tree/master/CEandSCmatlab)


## TrainingValidationRedistribution.py  
This script took a random 20% of the training set and added it to the validation set of the coco dataset. This served the purpose to have more images to select from during the image selection, which was very strict for the validation set but less strict for the training dataset.

* This resulted in the `instances_redistributed_train2017.json`, `instances_redistributed_val2017.json`, `stuff_redistributed_train2017.json`, `stuff_redistributed_val2017.json` annotations in the `coco/annotations` folder.
* This can result in the `/val2017_redistributed/` & `/train2017_redistributed/` image directories, if a copy of the images should be obtained (optional)
* Note that these were obtained in two steps with first using `TrainingValidationRedistribution.py` and then `TrainingValidationRedistribution_stuff.py`
* Please also note that hereafter I will call `val2017_redistributed` `val2017` and `train2017_redistributed` `train2017` 

## CreateDataset.py
This script contains a set of functions and commands to select eligible images from the COCO dataset. The settings can be found in the documentation of the function.
The arguments for the paper datasets are specified in there.

This script produces the following outputs within the dataset folders (e.g. `1_street`):

* a folder termed: `val2017_single_radial` in the image directory: 
	* This means that the images stem from the val2017 dataset, contain a single target object and had a radial spatial constraint ( this means no target object lies in a circle around the center of the image).  
	* Note that the complete collection of all computed datasets will be stored in the specified data directory (e.g. `mnt/Googolplex/coco/images/datasetName/`)
* a pickle termed `val2017_single_radial.pickle` in the result directory:
	* This pickle has all the meta-info on the dataset selection such as image-ids, target ids and the settings of `CreateDataset.py`
	* There is such a pickle for all computed datasets (which are in the result directory `/mnt/Googolplex/PycharmProjects/coco/`)
	
	
## ComputeDatasetFeatures.py
The first script was a preselection of images. With this collection of functions and commands, we analyse every image for a couple of quality criteria such as 

* image and target saliency, target area, target centre of Mass, Eccentricity, LocationTypicality, sceneComplexity. Those were used to make a final selection of images ('selection 1') obtained with `FilterFeatures.py`. 
* Please note that the dataset features have only been computed for the validation images.
* The scene complexity measures are merely read in and need to be prepared in `Matlab` based on the `LGN statistics`. These have been obtained with the scripts developed by Iris Groen and were then stored in the folder `SceneComplexity`
* This script returns the pickles such as `Features_single_radial.pickle`


## FilterFeatures.py
This scripts makes the final selection of images from the dataset based on the criteria dataframe. 

* The current chosen selection is: Pick images with a sceneComplexity spatial coherence of lower than 1.2, and those with a targetSalienceSum of higher than 0.04. 
* This scripts returns:
	* an extra `selection` column to both the `Features_single_radial.pickle` that specifies the selection as a boolean.
	* An overview of the current counts of examples for a given category & dataset `SelectionCounts_single_radial.csv` in the Dataset folder.

