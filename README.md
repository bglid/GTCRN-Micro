# GTCRN-Micro: *Microcontroller Speech Enhancement* 
<div align="center">

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/bglid/SERTime)](https://github.com/bglid/SERTime/blob/main/LICENSE)
[![Actions status](https://github.com/bglid/SERTime/workflows/build-desktop/badge.svg)](https://github.com/bglid/SERTime/actions)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
- - -
### NOTE: Still need to train new model architecture
*If you're reading this and wondering if the models are good to go, I am working through full toolchain validation and stil need to run the newly architected model through training*
- - -
</div>

## Roadmap / to-dos 

##### *Update 11/29/2025:*

Current model architecture in [gtcrn_micro.py](./gtcrn_micro/models/gtcrn_micro.py) is being trained on the **DNS3** dataset. Next to-dos are:

- [ ] Depending on the TCN-based results of the trained model, try a simple LSTM architecture in place of the Dual paths.
- [ ] Setup Streaming converter for chosen architecture
- [ ] Create QAT training setup to run on HPC

Chosen hardware for deployment will the **STM32H7**

##### *Update 11/26/2025:*

Currently the model architecture needs to be stripped drown significantly, not just changing incompatible units, to invoke without tons of latency on the *ESP32-S3*.

With that, the current focus is to train the adjusted, but quantizable architecture in [gtcrn_micro.py](./gtcrn_micro/models/gtcrn_micro.py), and target different hardware depending on the model performance.

Overall project roadmap can be found: [docs/plan.md](docs/plan.md)

List of to-dos can be found: [docs/TODO.md](docs/TODO.md)

- - -
## Acknowledgements

###### 1. The original model this is based off of is [GTCRN](https://github.com/Xiaobin-Rong/gtcrn), as well as a notable amount of the setup to train and change the model was based off of the same authors project [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus). They have some seriously impressive SE research! Please check out their research and throw some of their work a star!
###### 2. The project also requires moving from **PyTorch $\rightarrow$ ONNX $\rightarrow$ .tflite** to run inference on a microcontroller. None of this could have been possible without the direct help and work of [PINTO0309](https://github.com/PINTO0309) & their awesome project [onnx2tf](https://github.com/PINTO0309/onnx2tf). I highly recommend you check out their work if you are reading this and want to do a similar project. Please consider throwing some of their work a star!
- - - 

*This project would not have been possible without their efforts. Please consider giving them a star first before this project's!* 

- - - 