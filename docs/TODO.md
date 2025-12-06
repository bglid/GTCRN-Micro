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
 - [Model Design](#model-design)
 - [Export and Quantization](#export-and-quantization)
 - [On-device Deployment](#deployment)
 <!-- - [Bela Deployment](#bela) -->

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

- - - 

### Model Design

 - [x] Setup offline evaluation metrics in [/python](/python/)
 <!-- - [ ] load .onnx files -->
 - [x] Reimplement model replacing GRUs and for TFLM
 <!-- - [ ] Run metrics to profile on desktop against onnxruntime inference  -->
 - [x] Make model adjustments to comply with **Torch $\rightarrow$ ONNX $\rightarrow$ TFLite $\rightarrow$ TFLM**
 - [x] Fix parameter replacement for JSON
 - [x] Train new model
 - [ ] ~~Depending on the TCN-based results of the trained model, try a simple LSTM architecture in place of the Dual paths.~~
 - [ ] Setup Streaming converter for chosen architecture
 - [ ] ~~Create QAT training setup to run on HPC~~
 - [ ] Add examples of how to run the model in main README

- - - 

### Export and Quantization

 - [x] Convert PyTorch Adjusted model with TFLite Micro using PTQ 
 - [] Report measure of # of params and MMACs, ensure can fit on-device or adjust
 - [ ] Profile and measure accuracy of quanitzed model
 - [ ] Scope new hardware


- - - 

### Deployment

*NOTE: Subject to change to only MCU dependent on time*

 - [x] ~~Setup build toolchain for ESP32-S3 deployment~~
 - [ ] Setup build toolchain for new hardware...
 - [ ] Setup $I^2S$ for ESP32
 - [ ] Write SE inference program in C for MCU using ```.tflite``` model
 - [ ] Get performance for both models in terms of **Latency, Power Consumption, Accuracy**

- - - 
