# Leveraging spiking deep neural networks to understand neural mechanisms underlying selective attention
by Lynn K.A. Sörensen, Davide Zambrano, Heleen A. Slagter, Sander M. Bohté, & H. Steven Scholte


### Overview
This is the code that accompagnies this [paper](https://www.biorxiv.org/content/10.1101/2020.12.15.422863v4) and consists of three parts:
* the `asn` package for DCNN to sDCNN conversion with and without spatial attention 
* a set of function (`Datasets`) to replicate the dataset curation from the [COCO database](https://cocodataset.org/#home)
* the code to reproduce the results in the paper (`ModelTraining`, `ModelEvaluation`, `ModelAnalysis`)
* the code to reproduce the [paper](https://www.biorxiv.org/content/10.1101/2020.12.15.422863v4) figures (`Figures`)

<br/>


![Using a sDCNN for naturalistic visual search with spatial cues from [paper](link)](https://surfdrive.surf.nl/files/index.php/s/CSuFQPOxiCehrVt/download)


### Dependencies
The `asn` package relies on Keras with a TensorFlow backend. 

The `dataset curation` relies on [`COCO API`](https://github.com/cocodataset/cocoapi) as well as [`Deep Gaze II`](https://deepgaze.bethgelab.org/).

For the analysis part, results files can be downloaded [here](https://uvaauas.figshare.com/projects/Leveraging_spiking_deep_neural_networks_to_understand_neural_mechanisms_underlying_selective_attention/94406) to follow these analyses. 
Please make sure to add the right files to the `ModelEvaluation` and `ModelAnalysis` folder to reproduce the `Figures`.

The model training scripts can be found in `ModelTraining`. The resulting weights are provided [here](https://uvaauas.figshare.com/projects/Leveraging_spiking_deep_neural_networks_to_understand_neural_mechanisms_underlying_selective_attention/94406).


Last updated: 10.11.2021








