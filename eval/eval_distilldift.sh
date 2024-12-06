#conda activate geo-aware
export CUDA_VISIBLE_DEVICES=0
python preprocess_map.py --config configs/eval_distilldift_us.yaml
python eval.py --config configs/eval_distilldift_us.yaml
