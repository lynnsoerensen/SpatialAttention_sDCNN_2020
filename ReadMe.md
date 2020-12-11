# Leveraging spiking deep neural networks to model neural mechanisms underlying selective attention
** by Lynn K.A. Sörensen, Davide Zambrano, Heleen A. Slagter, Sander M. Bohté,& H.Steven Scholte **


### Overview
This is the code that accompagnies the [bioRxiv - paper](link) and consists of three parts:
* the `asn` package for DCNN to sDCNN conversion with and without spatial attention 
* a set of function (`Datasets`) to replicate the dataset curation from the [COCO database](https://cocodataset.org/#home)
* the code to reproduce the results in the paper (`ModelTraining`, `ModelEvaluation`, `ModelAnalysis`)
* the code to reproduce the [bioRxiv - paper](link) figures (`Figures`)

### Dependencies
The `asn` package relies on Keras (2.2.4) with a TensorFlow backend (1.10). 


The dataset curation relies on [`COCO API`](https://github.com/cocodataset/cocoapi) as well as [`Deep Gaze II`](https://deepgaze.bethgelab.org/).


For the analysis part, results files can be downloaded [here](link) to follow these analyses. 
Please make sure to add the right files to the `ModelEvaluation` and `ModelAnalysis` folder to reproduce the `Figures`.

The model training scripts can be found in `ModelTraining`. The resulting weights are provided [here](link).












