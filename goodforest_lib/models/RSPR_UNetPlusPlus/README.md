# RSPRUNet++ Model

## Description
The `RSPRUNet++` model is an advanced variant of the U-Net architecture designed for segmentation tasks. It leverages a combination of a ResNeSt encoder for feature extraction and a custom decoder with spatial and channel squeeze and excitation (scSE) blocks. The model is designed to perform well on multi-class segmentation tasks, using advanced normalization techniques (Masked Batch Normalization) to handle irregular input masks.

## Methodology

1. **Masked Batch Normalization**:
   - This custom normalization layer ensures that only the valid (non-zero) parts of the input are normalized, effectively ignoring masked-out regions. Masked pixels in the Sentinel-2 data are set to 0 across the first 11 channels.
   
2. **Encoder**:
   - The encoder is built upon the ResNeSt (https://github.com/zhanghang1989/ResNeSt) architecture (ResNeSt101), which is pre-trained on large datasets (e.g ImageNet) to provide powerful feature extraction capabilities. 
   - The first convolutional layer is customized to handle multi-channel inputs, beyond the standard 3-channel RGB.

3. **Spatial and Channel Squeeze and Excitation (scSE)**:
   - Each decoder block incorporates an scSE module that applies spatial and channel-wise attention to the feature maps. This attention mechanism helps the model focus on important spatial and channel information, enhancing segmentation performance.

4. **Decoder Blocks**:
   - The decoder mirrors the encoder, gradually upsampling the feature maps while applying the scSE blocks. The decoder progressively merges multi-level feature maps from the encoder and the decoder to refine the output segmentation.

5. **Final Convolution**:
   - The model combines outputs from the multiple decoder stages using a final 1x1 convolution to produce the final segmentation map.

## Model Architecture

### Data format

The data to pass to the model must satisfy the following format: `np.ndarray` with shape `(NB_CUBES, NB_CHANNELS, CUBE_HEIGHT, CUBE_WIDTH)`. In training phase, `NB_CHANNELS` contains one additionnal dimension to contain the class of the grand truth to predict.

### Key Components:
- **Initial Convolution**: Converts input to a format suitable for the ResNeSt encoder.
- **ResNeSt Encoder**: Pretrained backbone for powerful feature extraction.
- **Masked BatchNorm**: Normalization tailored for inputs with masked regions. See modelv0.py for example of usage.
- **scSE Modules**: Applied at the decoder level to refine feature maps.
- **Multi-stage Decoding**: Upsampling combined with skip connections to progressively reconstruct the segmentation mask.
- **Final Upsampling and Combination**: Produces the final segmentation map by combining results from multiple stages of the decoder.

### Example Usage

```python
if __name__ == "__main__":
    model = RSPRUNetPlusPlus(num_classes=3, input_channels=24)
    input_tensor = torch.randn(2, 24, 256, 256)  # Example input (batch of 2)
    outputs = model(input_tensor)
    print("Output shape:", outputs.shape)

```

### References 
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9570766/