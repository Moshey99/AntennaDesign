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
## how to train:
  - #### training models is done in antenna_training.py . The code receives multiple arguments (data path, hyper-parameters, etc.) and trains model's weights (and saves them, if wanted). In addition, this script plots the value of the loss function VS epochs
    1. Define the required argument in arg_parser as you wish (by changing the value in "default") or leave them as they are.
       <img width="572" alt="image" src="https://github.com/Moshey99/AntennaDesign/assets/104683567/a2a107e4-85e0-440a-a326-e8e3c4432484">
    2. Define manually model's architecture and the loss function it will try to minimize. There are various models and loss functions, so recommended for a start to do that:
       ```
           model = inverse_hypernet.inverse_forward_concat(inv_module=inverse_hypernet.small_inverse_radiation_no_hyper(),
                                                    forw_module=forward_GammaRad.forward_GammaRad(rad_range=radiation_range),
                                                    forward_weights_path_rad=args.forward_model_path_radiation,
                                                    forward_weights_path_gamma=args.forward_model_path_gamma)
           loss_fn = GammaRad_loss(lamda=GammaRad_lambda,rad_phase_fac=rad_phase_fac)
       ```
       that defines the model to be a inverse-forward concatenation with loaded&frozen forward weights, and the loss function to be GammaRad_loss (takes into account both radiation pattern and gamma)
       - IMPORTANT: the --inv_or_forw argument must match the chosen architecure. In the case above it must be ``` inverse_forward_GammaRad ```.
         for all the options look into the function create_dataloader in utils.py
    4. If i
