<p align="center">
 <h1 align="center">⚗️ DistillDIFT: Distillation of Diffusion Features for Semantic Correspondence</h2>
 <p align="center"> 
    Frank Fundel · Johannes Schusterbauer · Vincent Tao Hu · Björn Ommer
 </p>
 <p align="center"> 
    <b>CompVis @ LMU Munich, MCML</b>
 </p>
 <p align="center">
    WACV 2025
 </p>
 
<p align="center">
    <a href="https://compvis.github.io/distilldift">
        <img src="assets/badge-website.svg" alt="Website">
    </a>
    <a href="https://arxiv.org/abs/XXX">
        <img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" alt="Paper">
    </a>
</p>

## TLDR;
We present DistillDIFT, a highly efficient approach to semantic correspondence that delivers state-of-the-art performance with 
significantly reduced computational cost. Unlike traditional methods that combine multiple large generative models, 
DistillDIFT uses a novel distillation technique to unify the strengths of two vision foundation models into a single, 
streamlined model. By integrating 3D data without requiring human annotations, DistillDIFT further improves accuracy.

Overall, our empirical results demonstrate that our distilled model with 3D data augmentation achieves superior performance
to current state-of-the-art methods while significantly reducing computational load and enhancing practicality for real-world 
applications, such as semantic video correspondence.

## 🛠️ Setup
This setup was tested with `Ubuntu 22.04.4 LTS`, `CUDA Version: 12.2`, and `Python 3.9.20`.

First, clone the github repo...
```bash
git clone git@github.com:CompVis/distilldift.git
cd DistillDIFT
```

## 🔬 Evaluation on SPair-71K
Our evaluation pipeline for SPair-71K is based on [Telling-Left-From-Right](https://github.com/Junyi42/geoaware-sc) for better comparability.

So, first follow their environment setup and data preparation, don't forget to first:
```bash
cd eval
```

Then download the LoRA weights for the desired model via
```bash
wget https://ommer-lab.com/files/distilldift/distilldift_us.ckpt -P checkpoints/
```

And finally, run the evaluation script via
```bash
bash eval_distilldift.sh
```

## 🏋️ Training
First use
```bash
cd train
```

Then you have either the option to setup a virtual environment and install all required packages with `pip` via
```bash
pip install -r requirements.txt
```
or if you prefer to use `conda` create the conda environment via
```bash
conda env create -f environment.yaml
```

Download the COCO dataset and embed the images (for unsupervised training) via
```bash
bash datasets/download_coco.sh
python embed.py --dataset_name COCO
```

And run the training via
- Unsupervised Distillation
    ```bash
    accelerate launch --multi_gpu --num_processes 4 train.py distilled_us --dataset_name COCO --use_cache --parallel_cache
    ```
- Weakly Supervised Distillation
    ```bash
    accelerate launch --multi_gpu --num_processes 4 train.py distilled_ws --dataset_name SPair-71k --use_cache --parallel_cache
    ```
- Supervised Training
    ```bash
    accelerate launch --multi_gpu --num_processes 4 train.py distilled_s --dataset_name SPair-71k --use_cache --parallel_cache
    ```

### Refinement using CO3D
Follow the [official instructions](https://github.com/facebookresearch/co3d) to download the CO3D dataset and then prepare the CO3D dataset via
```bash
python datasets/create_co3d.py
```

And run the training via
```bash
accelerate launch --multi_gpu --num_processes 4 train.py distilled_s --dataset_name CO3D --use_cache --parallel_cache
```

## 🎓 Citation

Please cite our paper:

```bibtex
@article{fundel2025distilldift,
  author    = {Fundel, Frank and Schusterbauer, Johannes and Hu, Vincent Tao and Ommer, Björn},
  title     = {Distillation of Diffusion Features for Semantic Correspondence},
  journal   = {WACV},
  year      = {2025},
}
```
