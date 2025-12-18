# To-Dos:
- - -
 * ###### Check out the [plan.md](./plan.md) file to see the goals at a larger bird's eye view.
- - - 
## Organization:

 - Each set of Todos that I leave myself will be loosely tied to a goal from [plan.md](./plan.md). 
 - There is likely to be some bleed between these goals, as things are rarely built completely linearly. I will try to place these smaller to-dos as tasks of sub-goals where it makes sense. However, I want to also keep a trail of my process that is honest. 
- - - 
 ### List of Goals

 - [Project Setup](#project-setup)
 - [ESP32](#esp32)
 - [STM32](#esp32)

- - - 

### Project Setup

 - [x] Get UV setup for python and workflow scripts
 - [x] Setup pre-commit hooks
 - [x] Search for models to work off of (*GTCRN*)
 - [x] setup datasets for offline evaluation
 - [x] Test PyTorch Converters
 - [x] Setup datasets for training
 - [x] Setup basic .sh scripts/makefile
 - [x] Scope out datasheets for $I^{2}S$ Protocol 
 - [x] Setup offline evaluation metrics in [/python](/python/)

- - - 
## ESP32

### Model Design

 - [x] Reimplement model replacing GRUs and for TFLM
 - [x] Make model adjustments to comply with **Torch $\rightarrow$ ONNX $\rightarrow$ TFLite $\rightarrow$ TFLM**
 - [x] Fix parameter replacement for JSON
 - [ ] Train new model
 - [ ] Setup Streaming converter for chosen architecture 
 - [ ] ~~Create QAT training setup to run on HPC~~
 - [ ] Add examples of how to run the model in main README

- - - 

### Export and Quantization

 - [x] Convert PyTorch Adjusted model with TFLite Micro using PTQ 
 - [ ] Report measure of # of params and MMACs, ensure can fit on-device or adjust
 - [ ] Profile and measure accuracy of quanitzed model


- - - 

### Deployment

*NOTE: Subject to change to only MCU dependent on time*

 - [x] Setup build toolchain for ESP32-S3 deployment
 - [ ] Setup $I^2S$ for ESP32
 - [ ] Write SE inference program in C for MCU using ```.tflite``` model
 - [ ] Get performance for both models in terms of **Latency, Power Consumption, Accuracy**

- - - 

- - - 
## STM32

### Model Design

 - [ ] Reimplement model for Cube AI supported Ops
 - [ ] Test out STM32 Cube AI conversion
 - [ ] Train new model (once shapes are confirmed)
 - [ ] Setup Streaming converter for chosen architecture 
 - [ ] ~~Create QAT training setup to run on HPC~~
 - [ ] Add examples of how to run the model in main README

- - - 

### Export and Quantization

 - [ ] Export trained model with CubeAI 
 - [] Report measure of # of params and MMACs, ensure can fit on-device or adjust
 - [ ] Profile and measure accuracy of quanitzed model


- - - 

### Deployment

 - [ ] Setup build for STM32
 - [ ] Setup $I^2S$ for STM32
 - [ ] Write SE inference program in C for MCU 
 - [ ] Get performance for both models in terms of **Latency, Power Consumption, Accuracy**

- - - 