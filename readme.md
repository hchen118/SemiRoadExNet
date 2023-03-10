# SemiRoadExNet

SemiRoadExNet: A Semi-Supervised Network for Road Extraction from Remote Sensing Imagery via Adversarial Learning

# Network

![](./other/framework.jpg 'framework')
Fig. 1. The overall architecture of SemiRoadExNet.

SemiRoadExNet is based on a Generative Adversarial Network (GAN),  containing one generator with two discriminators. Firstly, both labeled and unlabeled images are put into the generator network for road extraction, and the outputs of the generator not only include road segmentation results but also the corresponding entropy maps. The entropy maps represent the confidence of prediction (road or non-road) for each pixel. Then, the two discriminators enforce the feature distributions keeping the consistency of road prediction maps and entropy maps between the labeled and unlabeled data. During the adversarial training, the generator is continuously regularized by exploiting the potential information from unlabeled data, thus the generalization capacity of the proposed model can be improved effectively.

# code

The code file contains folder 'data', 'network', 'other' and file 'compare_2.py', 'other_information.py', 'overlap_tes.py', 'tes_and_compare.py', 'train_12.py'. 

The 'data' folder is used for dataloader. The 'betwork' folder contains network. The 'other' folder contains other files. The 'compare_2.py' and 'overlap_tes.py' is used by 'tes_and_compare.py', which generates results. 'other_information.py' is used for setting important parameters. 'train_12.py' is a file for training.