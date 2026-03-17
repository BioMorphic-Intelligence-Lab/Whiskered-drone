# Simulation Setup

This document describes how to install and run the simulation environment used in this project.

The simulation framework is built on:

- **Isaac Sim 4.1**
- **Isaac Lab**
- **OmniDrones_whiskers**

---

# Prerequisites

Recommended system:

- Ubuntu 20.04 / 22.04
- NVIDIA GPU
- Python 3.10
- Conda

---

# 1. Install Isaac Sim 4.1

Download Isaac Sim from NVIDIA:

https://developer.nvidia.com/isaac-sim

After downloading, extract Isaac Sim into the following directory:
```
/home/<your_username>/Whiskered-drone/sim/
```

Then rename the extracted folder to:
isaacsim


The final directory structure should look like:
```
/home/<your_username>/Whiskered-drone/sim/isaacsim
```


Add the environment variable:

```bash
export ISAACSIM_PATH="/home/<your_username>/Whiskered-drone/sim/isaacsim"
```
Apply the changes:
```
source ~/.bashrc
```

# 2. Create Conda Environment
```
conda create -n sim python=3.10
conda activate sim
```

# 3. Configure Isaac Sim Python Environment
From the OmniDrones_whiskers directory:
```
cp -r conda_setup/etc $CONDA_PREFIX
conda activate sim
```
re-activate the environment
```
conda activate sim
```
Verify Isaac Sim Python:
```
python -c "from isaacsim import SimulationApp"
```
Verify PyTorch:
```
python -c "import torch; print(torch.__path__)"
```

# 4. Install Isaac Lab
Install dependencies:
```
sudo apt install cmake build-essential
pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash
```
Install Isaac Lab:
```
cd WhiskeredDroneSim/IsaacLab
./isaaclab.sh --install
```

# 5. Install OmniDrones_whiskers
```
cd WhiskeredDroneSim/OmniDrones_whiskers
pip install -e .
```

Verify Installation

Run the example:
```
cd examples
python whiskered_drone_exploration_random_CF.py
```

Running Simulation

Example command:
```
python whiskered_drone_exploration_GPIS.py env_rotation_deg=135

```
