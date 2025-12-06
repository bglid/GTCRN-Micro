# GTCRN-Micro: *Microcontroller Speech Enhancement* 
<div align="center">

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/bglid/SERTime)](https://github.com/bglid/SERTime/blob/main/LICENSE)
[![Actions status](https://github.com/bglid/SERTime/workflows/build-desktop/badge.svg)](https://github.com/bglid/SERTime/actions)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
- - -

*Project is still underway. The first non-quantized model checkpoints can be found [here](./gtcrn_micro/ckpts/). This checkpoint is ready to be used and it's performance can be found [here](./gtcrn_micro/README.md)*.

*First quantized model .tflite files can be found [here](./gtcrn_micro/models/tflite/)*

*For updates, check out either the issues, or the project roadmap [here](./docs/plan.md) and [here](./docs/TODO.md)*. 

*My plan is to get a full version with the deployment code up before the end of December!*
- - - 
</div>

## Project Background

The goal of this project is to walk through adjusting a modern, powerful, lightweight speech ehancement model to quantize it to an int8 representation and deploy it to a MCU, while trying to preserve as much performance as possible.

The motivation for this project comes from a general interest in designing speech processing (mainly speech enhancement) models that can run on microcontrollers. Impresive models such as GTCRN showcase significant advancements in designing speech enhancement that maintain great performance whilst being very lightweight. I have been generally curious in working through the process of quantizing and deploying a model like GTCRN to a microcontroller for quite some time. Ultimately, it's a passion project that allows me to build skills in this area of interest, and help provide insight for anyone else looking to do the same. 

Please check out the [acknowledgements!](#acknowledgements)
- - -

## How to use

### Setup
<details>

#### Clone the project:
```bash
git clone https://github.com/benjaminglidden/GTCRN-Micro.git
cd GTCRN-Micro
```
This project uses uv as the dependency manager. 
- To get setup with the project dependencies, first thing is to make sure [uv](https://docs.astral.sh/uv/) is installed on your device:

#### Installing uv:
**Linux & Mac OS**

From the **terminal**:
 - Use curl to download the script and execute it with sh:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
 - If for some reason you don't have `curl`, use `wget`

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows**

From **PowerShell:**

 - Use `irm` to download the script and execute it with `iex`:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

####  Verify UV installation

 - To verify you've installed it correctly, from your Terminal (or PowerShell), run:
```
uv --version
```
 - You should be returned a version of UV

#### Install the dependencies
```bash
uv sync
```
</details>

### Using the offline non-quantized model
*WIP*

- Trained model checkpoints can be found in [ckpts](./gtcrn_micro/ckpts/)
- Onnx files can be found int [onxx](./gtcrn_micro/models/onnx/)
...

### Quantized model
*WIP*

- Quantized tflite files can be found in [tflite](./gtcrn_micro/models/tflite/)
     - The necessary [replacement .json](./gtcrn_micro/models/tflite/replace_gtcrn_micro.json) is there for example if you want to recreate the quantization
...
- - - 

## Current Results (DNS3)

**GTCRN-Micro** evaluated on synthetically generated DNS3 test subset:
| Metric | Score |
|--------|-------|
| SDR    | 8.72 |
| SI-SNR | 8.39 |
| PESQ   | 1.79 |
| STOI   | 0.82 |

More details can be found in [./gtcrn_micro](./gtcrn_micro/README.md)
- - - 
## Roadmap / to-dos 

##### *Update 12/06/2025:*

Based on some issues getting QAT to work without changing some model architecture, decided it was premature to implement. Will revisit if needed

Current model architecture in [gtcrn_micro.py](./gtcrn_micro/models/gtcrn_micro.py) is being trained on the **DNS3** dataset. Next to-dos are:

- [ ] Setup Streaming converter for chosen architecture

Chosen hardware for deployment will be either the **ESP32-S3** or the **STM32H7** if the model is too large still for the ESP32

- - -
## Acknowledgements

###### 1. The original model this is based off of is [GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources](https://ieeexplore.ieee.org/document/10448310), leveraging the implementation code at [GTCRN](https://github.com/Xiaobin-Rong/gtcrn). A notable amount of the setup to train and change the model was based off of the same authors project [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus). They have some seriously impressive SE research! Please check out their research and throw some of their work a star!
###### 2. The project also requires moving from **PyTorch $\rightarrow$ ONNX $\rightarrow$ .tflite** to run inference on a microcontroller. None of this could have been possible without the direct help and work of [PINTO0309](https://github.com/PINTO0309) & their awesome project [onnx2tf](https://github.com/PINTO0309/onnx2tf). I highly recommend you check out their work if you are reading this and want to do a similar project. Please consider throwing some of their work a star!
- - - 

*This project would not have been possible without their efforts. Please consider citing them and giving them a star first before this project's!* 

- - - 
