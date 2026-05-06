# Vision Transformer with Sparse Encoder

<p align="center">
  <img src="figures/encoder.png" />
</p>

## Abstract

Vision Transformers have become a viable alternative to convolutional neural networks for
computer vision and pattern recognition tasks. These machine learning models gradually
transform the hidden vector representation of an image using an attention mechanism, sequentially aggregating information between all its elements. This allows for the detection
of patterns in the input data. However, attention is an algorithm with quadratic complexity. Its processing speed depends on the number of elements that interpret the input image
in the hidden space. There are tasks involving sparse data, where ignoring empty regions
can significantly accelerate the neural network’s performance. This paper proposes a static
Sparse Encoder for the case of a priori known empty image regions. Its key feature is a
ViT-compatible scheme for data processing without introducing new parametrized operations. Through its use in the Vision Transformer, significant acceleration is achieved, which
depends on the sparsity of the input data.

## Prerequisite

### Installation

* Set up conda environment:

    ```bash
    conda env create -n vit-se -f environment.yml
    conda activate vit-se
    ```

* Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

Be aware that preinstalled CUDA is required to run the project on GPU
(instructions [here](https://developer.nvidia.com/cuda-downloads)).

### ImageNet-1K Dataset

Experiments were conducted on the ImageNet-1K dataset. To download it, follow these steps:

* Register at [ImageNet](https://www.image-net.org/).

* Access and download [ILSVRC2012 (ImageNet-1K)](https://www.image-net.org/download.php) dataset.

* Organize the dataset in the following way:

    ```bash
    imagenet-1k
    ├── train
    │   ├── n01440764
    │   │   ├── n01440764_10026.JPEG
    │   │   ├── ...
    │   ├── ...
    ├── val
    │   ├── n01440764
    │   │   ├── ILSVRC2012_val_00000293.JPEG
    │   │   ├── ...
    │   ├── ...
    ```

    Notice that class labels (*n01440764*, etc.) are in subfolders' names instead of images filenames' postfixes. Folder tree structure like this is required for **torchvision** *ImageFolder* generic dataloader.

Taking the fact that the ImageNet-1K dataset is not sparse by default into consideration, a patch-aware "sparsification" augmentation is implemented. At each application, one of three mask types is sampled with equal probability: line masks, symmetric padding-like masks, and irregular blob masks. The target sparsity is controlled on the patch grid, so the number of excluded tokens matches the requested ratio for the chosen ViT patch size. Though not true sparsity, it still helps to study effectiveness of the Sparse Encoder.

<p align="center">
  <img src="figures/augmentation.png">
</p>

## Usage

### Training

ViT/SE (Vision Transformer with Sparse Encoder) uses pure ViT weights for ImageNet-1K classification as checkpoints. A loader from **torchvision** downloads them automatically upon launching training or testing. Bias in the hidden space projection convolution is removed as proposed by the paper "Effective data processing in Vision Transformers". If `--num_classes` differs from ImageNet-1K, the classifier head is reinitialized while compatible pretrained weights are loaded.

If you want to train ViT/SE (for example, on a different dataset) with chosen pretrained ViT variation weights, run `train.py`:

```bash
python3 train.py [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] --experiment_name EXPERIMENT_NAME [--model_name MODEL_NAME] [--encoder_mode {sparse,masked,default}] [--num_classes NUM_CLASSES] [--resize_size RESIZE_SIZE] [--crop_size CROP_SIZE] [--interpolation INTERPOLATION] [--lr LR] [--warmup_epochs WARMUP_EPOCHS] [--weight_decay WEIGHT_DECAY] [--randaug_n RANDAUG_N] [--randaug_m RANDAUG_M] [--mixup_alpha MIXUP_ALPHA] [--erase_ratio_min ERASE_RATIO_MIN] [--erase_ratio_max ERASE_RATIO_MAX] [--background_threshold BACKGROUND_THRESHOLD] [--weights WEIGHTS] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR] [--device DEVICE] [--seed SEED]

options:
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --experiment_name EXPERIMENT_NAME
                        Name of the experiment, used for saving checkpoints and logs
  --model_name MODEL_NAME
                        Name of the ViT/SE model to use
  --encoder_mode {sparse,masked,default}
                        Encoder forward mode: sparse, masked, or default
  --num_classes NUM_CLASSES
                        Number of output classes
  --resize_size RESIZE_SIZE
                        256 for ViT/SE-B, 242 for ViT/SE-L, 518 for ViT/SE-H
  --crop_size CROP_SIZE
                        224 for ViT/SE-B and ViT-L, 518 for ViT/SE-H
  --interpolation INTERPOLATION
                        "bilinear" for ViT/SE-B and ViT/SE-L, "bicubic" for ViT/SE-H
  --lr LR               Base learning rate
  --warmup_epochs WARMUP_EPOCHS
                        Amount of warmup epochs
  --weight_decay WEIGHT_DECAY
                        Weight decay
  --randaug_n RANDAUG_N
                        BigVisionRandAugment number of operations
  --randaug_m RANDAUG_M
                        BigVisionRandAugment magnitude
  --mixup_alpha MIXUP_ALPHA
                        TwoHotMixUp alpha value
  --erase_ratio_min ERASE_RATIO_MIN
                        Lower bound for RandomSelectiveErasing ratio
  --erase_ratio_max ERASE_RATIO_MAX
                        Upper bound for RandomSelectiveErasing ratio
  --background_threshold BACKGROUND_THRESHOLD
                        Apply ThresholdBackgroundZeroing after ToTensor if non-negative
  --weights WEIGHTS     Path to checkpoint to use (optional)
  --data_path DATA_PATH
                        Path to dataset
  --output_dir OUTPUT_DIR Path to save checkpoints
  --device DEVICE       Training device
  --seed SEED
```

Example for launching the training of ViT/SE-H/14 with recommended default settings would look like this:

```bash
python3 train.py --experiment_name custom_vit_se_h_14 --model_name vit_se_h_14 --resize_size 518 --crop_size 518 --interpolation bicubic --data_path /path/to/imagenet-1k --output_dir .
```

During training in the end of each epoch, a checkpoint is saved with the name `last.pth`. A validation loss tracker is being used to also preserve `best.pth` model in case of overfitting. Metrics are being logged into `log.txt`.

### Evaluation

If you want to evaluate ViT/SE with chosen pretrained ViT variation weights or your custom weights that were saved after
running `train.py` accordingly, run `test.py`:

```bash
python3 test.py [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--model_name MODEL_NAME] [--encoder_mode {sparse,masked,default}] [--num_classes NUM_CLASSES] [--resize_size RESIZE_SIZE] [--crop_size CROP_SIZE] [--interpolation INTERPOLATION] [--weights WEIGHTS] [--data_path DATA_PATH] [--device DEVICE] [--erase_ratio ERASE_RATIO] [--background_threshold BACKGROUND_THRESHOLD] [--seed SEED]

options:
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --model_name MODEL_NAME
                        Name of the ViT/SE model to use
  --encoder_mode {sparse,masked,default}
                        Encoder forward mode: sparse, masked, or default
  --num_classes NUM_CLASSES
                        Number of output classes
  --resize_size RESIZE_SIZE
                        256 for ViT/SE-B, 242 for ViT/SE-L, 518 for ViT/SE-H
  --crop_size CROP_SIZE
                        224 for ViT/SE-B and ViT-L, 518 for ViT/SE-H
  --interpolation INTERPOLATION
                        "bilinear" for ViT/SE-B and ViT/SE-L, "bicubic" for ViT/SE-H
  --weights WEIGHTS     Path to weights of trained model (optional)
  --data_path DATA_PATH
                        Path to dataset
  --device DEVICE       Evaluation device
  --erase_ratio ERASE_RATIO
  --background_threshold BACKGROUND_THRESHOLD
                        Apply ThresholdBackgroundZeroing after ToTensor if non-negative
  --seed SEED
```

Example for launching the evaluation of ViT/SE-H/14 with recommended default settings would look like this:

```bash
python3 test.py --model_name vit_se_h_14 --resize_size 518 --crop_size 518 --interpolation bicubic --data_path /path/to/imagenet-1k
```

## Results

Due to lack of new parametrized operations, no new weights are to be distributed. Theoretically, ViT/SE will train differently from that of original model if the input data is sparse due to attention masking. Later on pretrained weights on widely recognized open sparse image benchmarks might be released.

As per the paper, evaluation experiments were conducted on a single NVIDIA GeForce RTX 3070 Ti with batch size being 32 samples unless stated otherwise. Masked ViT is the original ViT with empty patches masked using an *attention mask*. Latency was measured only for the forward pass on GPU – data loading and preprocessing were excluded. Gradients were disabled before measurement, 20 warm-up runs were performed, and CUDA was synchronized. The final latency value was averaged over all subsequent batches of the measurement run on the full validation set.

Comparison of models with fixed sparsity ratio – 50 %:

| Model             | Latency, ms   | Acceleration, ms | Acceleration, % | Acc@1, % | Acc@5, % |
|-------------------|---------------|------------------|-----------------|----------|----------|
| Masked ViT–B/16   | 3.413         | –                | –               | 63.348   | 83.674   |
| Masked ViT–B/32   | 0.925         | –                | –               | 44.356   | 67.286   |
| Masked ViT–L/16   | 11.347        | –                | –               | 64.448   | 84.628   |
| Masked ViT–L/32   | 3.040         | –                | –               | 46.214   | 68.714   |
| Masked ViT–H/14   | 74.307        | –                | –               | 78.022   | 93.176   |
| ViT/SE–B/16       | 2.071         | 1.342            | 39.320          | 63.320   | 83.664   |
| ViT/SE–B/32       | 0.897         | 0.028            | 3.027           | 44.378   | 67.300   |
| ViT/SE–L/16       | 6.282         | 5.065            | 44.637          | 64.410   | 84.596   |
| ViT/SE–L/32       | 2.389         | 0.651            | 21.414          | 46.196   | 68.706   |
| ViT/SE–H/14       | 36.129        | 38.178           | 51.379          | 78.028   | 93.170   |

Latency and accuracy dependence on the sparsity ratio on ViT/SE-H/14:

| Sparsity, %      | Latency, ms   | Acceleration, ms | Acceleration, % | Acc@1, % | Acc@5, % |
|------------------|---------------|------------------|-----------------|----------|----------|
| 0                | 79.387        | –                | –               | 88.096   | 98.524   |
| 20               | 61.497        | 17.890           | 22.535          | 86.276   | 97.630   |
| 40               | 47.073        | 32.314           | 40.704          | 81.888   | 95.494   |
| 60               | 27.535        | 51.852           | 65.315          | 71.740   | 89.602   |
| 80               | 12.910        | 66.477           | 83.738          | 41.618   | 63.226   |

## Citation

```bibtex
@article{makarov2025vit-se,
  title = {Effective sparse data processing in Vision Transformers},
  volume = {},
  url = {},
  doi = {},
  number = {},
  journal = {Vestnik of Saint Petersburg State University. Applied Mathematics. Computer Science. Control Processes},
  author = {Makarov, Georgy V.},
  year = {2025},
  pages = {}
}
```
