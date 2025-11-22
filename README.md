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

Overall project roadmap can be found: [docs/plan.md](docs/plan.md)

List of to-dos can be found: [docs/TODO.md](docs/TODO.md)

## Acknowledgements
- - -

###### 1. The original model this is based off of is [GTCRN](https://github.com/Xiaobin-Rong/gtcrn), as well as a notable amount of the setup to train and change the model was based off of the same authors project [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus). They have some seriously impressive SE research! Please check out their research and throw some of their work a star!
###### 2. The project also requires moving from **PyTorch $\rightarrow$ ONNX $\rightarrow$ .tflite** to run inference on a microcontroller. None of this could have been possible without the direct help and work of [PINTO0309](https://github.com/PINTO0309) & their awesome project [onnx2tf](https://github.com/PINTO0309/onnx2tf). I highly recommend you check out their work if you are reading this and want to do a similar project. Please consider throwing some of their work a star!
- - - 

*This project would not have been possible without their efforts. Please consider giving them a star first before this project's!* 
