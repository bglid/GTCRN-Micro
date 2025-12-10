# Model Training and Evaluation
- - -
## Main adjustments
 - Swapped GRU blocks + DPGRNN for a psuedo "Grouped-TCN"
 - Removed optional SFE and TRA (mainly for TFLite/LiteRT conversion)
 - Dropped channel and group amounts for quantization compliance
 - Adjusted dilation and padding for quantization compliance
- - - 
## Model directory

wip 

- - - 
<!-- **Offline Model**
 - [Full PyTorch checkpoints](./ckpts/best_model_dns3.tar) 
 - [ONNX files](./models/onnx/)
 - [Quantized .tflite files](./models/tflite/)

**Streaming Model**

 - wip...
- - - 

## Evaluation Results

- **Dataset:** Synthetic DNS3 Test Set of 800 10-second instances
- **Model Params:** 18.38 k
- **MMACs:** 46.22
- **Examples:** See [examples](./examples/) - Mixed results were deliberately picked to give an honestly representation

| Model            | Quantization | SDR  | SI-SNR | PESQ | STOI |
|------------------|-------------|------|--------|------|------|
| Noisy Baseline || 3.39 | 3.39 | 1.40 | 0.79 |
| GTCRN-Micro    | None     | 8.72  | 8.39    | 1.79  | 0.82  |
| GTCRN-Micro  (.tflite)   | int8        | ... | ...   | ... | ... |
| GTCRN-Micro Streaming    | None     | ...  | ...    | ...  | ...  |
| GTCRN-Micro Streaming (.tflite) | int8  | ...  | ...    | ...  | ...  |
- - - 

- **Dataset:** DNS3 blind test set
- **Model Params:** 18.38 k
- **MMACs:** 46.22

| Model            | Quantization | SDR  | SI-SNR | PESQ | STOI |
|------------------|-------------|------|--------|------|------|
| GTCRN-Micro    | None     | 8.72  | 8.39    | 1.79  | 0.82  |
| GTCRN-Micro  (.tflite)   | int8        | ... | ...   | ... | ... |
| GTCRN-Micro Streaming    | None     | ...  | ...    | ...  | ...  |
| GTCRN-Micro Streaming (.tflite) | int8  | ...  | ...    | ...  | ...  | -->

## Acknowledgements
The original model this is based off of is [GTCRN](https://github.com/Xiaobin-Rong/gtcrn), as well as a notable amount of the setup to train and change the model was based off of the same authors project [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus).

If you found any of this helpful, please give the original authors repos a star or check out their great work! Without it, none of this project would be possible.
- - -