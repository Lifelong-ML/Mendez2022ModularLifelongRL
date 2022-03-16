# Modular Lifelong RL

This code base reproduces the experiments of the ICLR 2022 paper "Modular Lifelong Reinforcement Learning via Neural Composition". 

## Discrete 2-D tasks

Primary dependencies: 
- gym-minigrid: install local version via `pip install -e gym-minigrid`
- torch-ac-composable: install local version via `pip install -e torch-ac-composable`
- pytorch version 1.5.1

To reproduce the results of our compositional agent, execute the following command from the `torch-ac-composable/torch_ac_composable/` directory:

`python -m experiments.ppo_minigrid_lifelong --algo comp-ppo --learning-rate 1e-3 --steps-per-proc 256 --batch-size 64 --procs 16 --num-tasks 64 --num-steps 1000000 --max-modules 4`

## Robotic Manipulation

Primary dependencies:
- Spinning Up: install local version via `pip install -e spinningup`
- Robosuite: install local version via `pip install -e robosuite`
- pytorch version 1.8.1

To reproduce the results of our compositional agent, execute the following command from the `training/` directory:

`python train_lifelong_ppo.py --algo comp-ppo --num-tasks 48 --cpu 40 --gamma 0.995 --epochs 150 --steps 8000`

## Citing our work

If you use this work, please make sure to cite our paper:

```
@inproceedings{
    mendez2022modular,
    title={Modular Lifelong Reinforcement Learning via Neural Composition},
    author={Jorge A Mendez and Harm van Seijen and Eric Eaton},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=5XmLzdslFNN}
}
```