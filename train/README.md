## 🧫 Evaluation

1. Download datasets
    - SPair-71k: `bash datasets/download_spair.sh`
    - PF-WILLOW: `bash datasets/download_pfwillow.sh`
    - CUB-200-2011: `bash datasets/download_cub.sh`

2. Setup your `dataset_config.yaml`
    - `path`: absolute path to the dataset
    - `num_samples`: limit the number of samples to use
    - `random_sampling`: whether to shuffle the dataset before sampling

3. Setup your `eval_config.yaml`
    - `image_size`: size of the input images after resizing
    - `batch_size`: overwrite the batch size for evaluation
    - `grad_enabled`: whether to enable gradient calculation
    - `drop_last_batch`: whether to drop the last batch if it is smaller than the batch size
    - `layers`: list of layers to evaluate, only possible together with `--use_cache`
    - Additional options are passed to the model

4. Run the evaluation script: `accelerate launch --multi_gpu --num_processes [n] eval.py [options]`
    - Make sure to set visible GPUs e.g. `export CUDA_VISIBLE_DEVICES=0`
    - Some models need a different diffusers version:
        - hedlin: `diffusers==0.8.0`
        - tang: `diffusers==0.15.0`
        - luo: `diffusers==0.14.0`
    - For all other models we use `diffusers==0.24.0`
    - Use `--use_cache` to speed up evaluation
    - Use `--plot` to plot the results of layerwise evaluation
    - When using cache, remember to use `--reset_cache` if you made changes on the model, else the features will stay the same

    Example: `accelerate launch --multi_gpu --num_processes 4 eval.py distilled_model --use_cache --reset_cache`

## 🔬 Training

- Supervised Training: `accelerate launch --multi_gpu --num_processes 4 train.py distilled_s --dataset_name SPair-71k`
- Weakly Supervised Distillation: `accelerate launch --multi_gpu --num_processes 4 train.py distilled_ws --dataset_name SPair-71k --use_cache`
- Unsupervised Distillation: `accelerate launch --multi_gpu --num_processes 4 train.py distilled_us --dataset_name COCO --use_cache --parallel_cache`

For fully unsupervised distillation, retrieval-based image sampling is needed, therefore you first have to embed the dataset using the following command: `python embed.py --dataset_name COCO`

## ⚗️ Distilled models

| Model | Link |
|----------|----------|
| DistillDIFT (Unsupervised) | [Download (8MB)]() |
| DistillDIFT (Weakly-Supervised) | [Download (8MB)]() |
| DistillDIFT (Supervised) | [Download (8MB)]() |
