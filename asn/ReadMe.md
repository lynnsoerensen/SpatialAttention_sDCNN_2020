# Adaptive Spiking Deep Convolutional Neural Networks (sDCNNs) with Spatial Attention 

Implementation for Keras (2.2.4) with a tensorflow backend (1.10).

This package is used to convert ResNets into their ASN equivalent. In addition, it has a functionality to apply different kinds of spatial attention to its computations.

## Features

### Training of ANNs for SNN compatibility

We here provide a non-linearity that can be used to train most standard feedforward architectures (e.g. AlexNet, VGG, ResNet).
To do so, the ReLUs in the architecture have to replaced by the `ASNTransfer`-layer (`layer/training.py`). 

In addition, one additional non-linearity should be introduced at the beginning of the architecture to ensure that the image pixel intensities are encoded as currents at the very start of the network.

After these two modifications, the architecture can be trained with standard training approaches (e.g. SGD, data augmentation)

`resnet_asn.py` is an example for such an architecture. 

During inference with ASN, the chosen architecture will be evaluated over the amount of specified timesteps. This can easily lead to very large networks. We here mainly use ResNet18 for that reason.  

### Conversion from ANN to SNN
After training, the obtained weights can be introduced in a spiking DCNN. 
Any layer without a non-linearity can be applied to both standard values but also to temporal spike trains.

In  `conversion/convert.py` a collection of functions can be used for this. The conversion can be initiated with `convert_model()`.
This function converts both the architecture as well as the weights and return the spiking DCNN.

The conversion can be done for different levels of precision by changing the speed of adaptation in the neurons (`mf`)

### Conversion to a spatial Attention SNN
During conversion, attentional setting can be implemented that will apply these choices during model conversion. 
The choices can be specified via `attention/attn_param.py`

For spatial attention modulation, it can be chosen between precision, input gain or connection gain (termed output gain). 
 
### Evaluation of SNNs with and without spatial attention
The obtained spiking DCNN can be evaluated with the functions in `evaluation/wrapper.py`













