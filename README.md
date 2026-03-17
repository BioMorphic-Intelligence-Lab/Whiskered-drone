# Whisker-Based Tactile Flight for Tiny Drones

This repository contains the **simulation and real-world implementation** for the paper:

**Whisker-Based Tactile Flight for Tiny Drones**

Paper:  
https://arxiv.org/abs/2510.03119

The goal of this project is to enable **tiny drones to perform tactile navigation exploration using whisker-like sensors** and active exploration strategies.

---

# Repository Overview
```
Whiskered-drone
│
├── simulation
│ └── WhiskeredDroneSim
│   ├── IsaacLab
│   └── OmniDrones_whiskers
│
└── real_world
  └── crazyflie-firmware
```
### Simulation

Simulation code is located in:

`simulation/WhiskeredDroneSim/`

It contains two main components:

**IsaacLab**  
The base simulation framework built on top of **Isaac Sim**.

**OmniDrones_whiskers**  
A modified drone simulation framework based on **OmniDrones**, integrated with **Isaac Sim 4.1** and **Isaac Lab**, and extended with modules for whisker-based exploration.

Note that **Isaac Sim 4.1 is not included in this repository** and must be downloaded separately by users.

Detailed installation and setup instructions are provided in:

`simulation/README.md`

---

### Real-world

The implementation used on the physical drone platform is located in:
real_world/

It includes:

- Modified Crazyflie firmware  
- Whisker sensor interation  
- Onboard tactile navigation and exploration  
- Python scripts for running experiments  

See:

real_world/README.md

---


# Citation

If you use this code in your research, please cite:
```bibtex
@article{ye2025whisker,
  title={Whisker-based Tactile Flight for Tiny Drones},
  author={Ye, Chaoxiang and de Croon, Guido and Hamaza, Salua},
  journal={arXiv preprint arXiv:2510.03119},
  year={2025}
}
```
If you use the [OmniDrones](https://github.com/btx0424/OmniDrones) framework, please also cite their work.


---

## Acknowledgement

This project builds upon the following open-source frameworks:

- [Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [OmniDrones](https://github.com/btx0424/OmniDrones)

We thank the original authors for their contributions.

---

# License

This repository follows the **MIT License** inherited from the OmniDrones project.

See the LICENSE file for details.
