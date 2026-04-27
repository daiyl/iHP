# iHP: Intelligent Healthcare Process Simulation Framework #
A simulation framework for generating realistic, mechanism-driven healthcare processes based on Partially Observable Markov Decision Processes (POMDP). This framework enables systematic evaluation of sequence prediction algorithms for disease progression modeling and clinical decision support.


## Overview ##
iHP provides a standardized, reproducible, and extensible foundation for evaluating intelligent healthcare algorithms. It addresses the critical challenge of data scarcity in healthcare by generating high-quality synthetic multimodal data that respects underlying physiological mechanisms.

Key Features:
- Mechanism-driven simulation of patient health state dynamics
- Interpretable state transition models
- Shared latent state encoding for naturally aligned multimodal data
- Standardized benchmark for sequence prediction algorithms

## Generation Pipeline ##
The framework follows a three-stage generation pipeline:


1. **Latent State Generation**: Generate intervention-state sequences using predefined transition models
2. **Multimodal Synthesis**: Map latent states to high-fidelity images (chest X-ray, fundus photography, OCT) via conditional diffusion models
3. **Temporal Interpolation**: Create continuous transitions between pathological states using latent diffusion-based interpolation

## Evaluation Benchmark ##
The benchmark includes 9 state-of-the-art sequence prediction models across 4 categories:

| Category | Models |
|:--------:|:------:|
| **Transformer** | iTransformer, PatchTST |
| **RNN** | LSTM, xLSTM, Mamba |
| **CNN** | SCINet, TimesNet |
| **MLP** | DLinear, TiDE |


## Quick Start ##

    # python

    from Patient.models.delayed_model import DelayedStateTransitionModel
    
    # Initialize model with parameters
    model = DelayedStateTransitionModel(n=2, m=4, k=2)
    
    # Define intervention sequence
    interventions = [
	    (0, 0, 1),  # apply intervention
	    (0, 0, 1),  # continue intervention
	    (0, 0, 0),  # stop intervention
    ]
    
    # Simulate health state evolution
    result = model.simulate(interventions, return_details=True)
    print(f"Health states: {result['states']}")
    print(f"Events: {result['events']}")


