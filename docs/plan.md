# Project Plan:
- - -
 * ###### Partially documenting steps for myself, but also for anyone else looking to see what the planning from start to finish looks like.
- - - 
## Goals

#### ESP32
 - [x] Setup Repo and tooling
 - [x] Design an adjusted DNN model based off of *GTCRN* that is supported by TFLite Micro
 - [x] Train new full toolchain-compatible model
 - [ ] Export and Quantize tuned DNN model
 - [ ] Deploy / Test on MCU (*ESP32-S3*)
 - [ ] Report performance and evaluation results

#### STM32
 - [ ] Design an adjusted DNN model based off of *GTCRN* that is supported by Cube AI
 - [ ] Train new full toolchain-compatible model
 - [ ] Export and Quantize tuned DNN model
 - [ ] Deploy / Test on MCU (*STM32H7*)
 - [ ] Report performance and evaluation results
- - - 
### *Notes*

 - Each goal does not warrant the same time/effort allocation, nor with this be completely linear. 
 - Each goal likely is comprised of smaller goals. Example: For both of the hardware deployments, wiring up a circuit for an $I^{2}S$ protocol to set the mic inputs will likely be a challenge in it's own. 
     - For more details, I will have a list of more specific To-Dos in the [TODO.md](./TODO.md) that accompany each larger goal. 
- - - 