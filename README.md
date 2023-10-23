# AntennaDesign
Code for Antenna Design
## Contents:
- losses.py - various loss function to train/evaluate different models.
- models - forward/inverse/forward & inverse concatenated model architectures.
- utils.py - different utilities like nearest neighbor benchmark, data preprocessing, etc. 
- trainer.py - training and evaluating functions.
- antenna_training.py - main script. loads data,preprocess it, defines hyper-parameters and runs it all.

* Important: if inverse model is hypernetwork, then batch size MUST be 1. Learning process can be modified by
gradient accumulation parameter in the main script. If inverse is not hypernetwork, then batch size can be larger and no gradient accumulation is needed.