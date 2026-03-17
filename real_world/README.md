# Real-World Experiments

This directory contains the real-world implementation of our work:

**Whisker-Based Tactile Flight for Tiny Drones**

It includes:
- Modified Crazyflie firmware
- Python scripts for running experiments

---

# 1. Hardware Requirements

- Crazyflie **Brushless**
- Two whisker sensors (left & right)
- Crazyradio
- Host computer (Ubuntu recommended)

---

# 2. Flash Firmware

### Step 1: Enter firmware directory

cd real_world/crazyflie-firmware/examples/demos/app_whiskers_CF


### Step 2: Enter bootloader mode

1. Plug in the battery  
2. Power off the Crazyflie  
3. Press and hold the power button  
4. Release after a few seconds  

### Step 3: Build and flash
``
make clean
make -j 12
make cload
```

---

# 3. Run Experiments

## 3.1 Set Crazyflie URI

Edit the Python script to match your Crazyflie address:

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')


---

## 3.2 Tactile Navigation

python CF_OA.py


---

## 3.3 Active Tactile Exploration

python CF_OA_GPIS.py


---

# 4. Parameters

## FSM Parameters

set_initial_params(scf.cf, 50.0, 90.0, 50.0, 90.0, 0.2, 25.0)


Controls:
- whisker thresholds  
- maximum velocity  
- turning rate  

---

## Post Calibration (Optional)

post_cal_params(scf.cf, 1.00, 1.00, 0.00, 0.00)


Used for correcting sensor bias. Default is no correction.

---

## MLP Recalibration (Important)

If you use a different whisker setup, you **must update the onboard MLP model**.

Modify the following file:


real_world/crazyflie-firmware/examples/demos/app_whiskers_CF/src/mlp.c


Specifically:
- Replace the **MLP weights and biases** with your trained parameters  
- Update the **input normalization parameters** to match your whisker calibration  

The MLP is a **3-layer neural network** used for tactile signal processing.  
Incorrect parameters will lead to poor or unstable behavior.

---

# 5. What Was Modified

The `crazyflie-firmware` is based on the official firmware with the following changes:

## 1. Whisker Data Transmission
- Two whiskers (left & right)  
- UART1 and UART2  
- 50 Hz streaming  

## 2. Onboard Control Framework
- Implemented in Crazyflie app layer (RTOS-based)  
- Supports tactile navigation and exploration  

## 3. Custom App Layer
- Finite-state machine (FSM)  
- Tactile sensing integration (TDORC, MLP, KF)  
- Real-time active decision making onboard (e.g., GPIS)

---

# 6. Troubleshooting

## 1. Cannot connect
- Check Crazyradio is plugged in  
- Check URI  
- Make sure Crazyflie is powered on  

---

## 2. No whisker data
- Check wiring (UART1 / UART2)  
- Check firmware flashed correctly  

---

## 3. Drone does not move

Make sure:

scf.cf.param.set_value("app.stateOuterLoop", 1)


---

# Acknowledgment

This project builds upon the official [crazyflie-firmware](https://github.com/bitcraze/crazyflie-firmware) by Bitcraze.
