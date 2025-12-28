# Model Training and Evaluation
- - -
## Main adjustments
 - Swapped GRU blocks + DPGRNN for a psuedo "Grouped-TCN"
 - Removed optional SFE and TRA (mainly for TFLite/LiteRT conversion)
 - Dropped channel and group amounts for quantization compliance
 - Adjusted dilation and padding for quantization compliance
- - - 
## Model directory

- - - 
**Offline Model**
 - [Full PyTorch checkpoints](./ckpts/best_model_dns3.tar) 
 <!-- - [ONNX files](./models/onnx/)
 - [Quantized .tflite files](./models/tflite/) -->

**Streaming Model**

 - wip...
- - - 

## Evaluation Results

- **Dataset:** Synthetic DNS3 Test Set 
- **Model Params:** **19.01 k**
- **MMACs:** **45.92**
- **Examples:** See [examples](./examples/) 

| Model            | Quantization | SDR  | SI-SNR | PESQ | STOI |
|------------------|-------------|------|--------|------|------|
| Noisy Baseline || 3.39 | 3.39 | 1.40 | 0.79 |
| GTCRN-Micro    | - | 10.41  | 9.85    | 1.98  | 0.85  |
| GTCRN-Micro Streaming    | - | ...  | ...    | ...  | ...  |
| GTCRN-Micro Streaming  (ONNX)  | - | ...  | ...    | ...  | ...  |
| GTCRN-Micro Streaming (.tflite) | int8  | ...  | ...    | ...  | ...  |
- - - 

- **Dataset:** DNS3 Blind Test Set
- **Model Params:** **19.01 k**
- **MMACs:** **45.92**

| Model            | Quantization | DNSMOS-P.808  | BAK | SIG | OVRL |
|------------------|-------------|------|--------|------|------|
| Noisy Baseline |      | 2.96  | 2.65    | 3.20  | 2.33  |
|  RNNoise   | -     | 3.15  |  3.45   | 3.00  | 2.53  |
| GTCRN *(Original)*    | -     | 3.44  |  3.90   | 3.00  | 2.70  |
| GTCRN-Micro    | -      | 3.25  |  3.60   | 2.99  | 2.58  |
| GTCRN-Micro Streaming    | - | ...  | ...    | ...  | ...  |
| GTCRN-Micro Streaming  (ONNX)  | - | ...  | ...    | ...  | ...  |
| GTCRN-Micro Streaming (.tflite) | int8  | ...  | ...    | ...  | ...  |

## Acknowledgements
The original model this is based off of is [GTCRN](https://github.com/Xiaobin-Rong/gtcrn), as well as a notable amount of the setup to train and change the model was based off of the same authors project [SEtrain](https://github.com/Xiaobin-Rong/SEtrain/tree/plus).

If you found any of this helpful, please give the original authors repos a star or check out their great work! Without it, none of this project would be possible.
- - -