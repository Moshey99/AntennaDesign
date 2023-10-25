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

## checkpoints:
- INVERSE_GammaRad_concat_HuberCyclic_loss_lamda1_radphasefac1_lr0.0002_loss0.674.pth - trained inverse_forward model 
that predicts both gamma and radiation pattern.
- FORWARD_small_10layers_dB_linpred.pth - trained forward model that predicts gamma linearly.
- FORWARD_radiation_HuberCyclic_loss_range[-55,5].pth - trained forward model that predicts radiation pattern [dB].
## models:
 - baseline_regressor.small_deeper_baseline_forward_model - architecture of forward model for gamma.  
 - forward_radiation.Radiation_Generator - architecture of forward model for radiation pattern.
 - forward_GammaRad.forward_GammaRad - architecture of forward model for both gamma and radiation pattern.
 - inverse_hypernet.small_inverse_radiation_no_hyper - inverse model
 - inverse_hypernet.inverse_forward_concat - inverse and forward concatenated