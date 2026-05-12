#!/usr/bin/env bash

if ! conda env list | grep -qE '^\s*hawor\s'; then
    conda create -n hawor python=3.10 -y
fi
eval "$(conda shell.bash hook)"
conda activate hawor

uv sync

cd thirdparty/DROID-SLAM
uv run setup.py install
cd ../..

[ -f ./weights/external/droid.pth ] || uv run gdown --fuzzy https://drive.google.com/file/d/1XXmgxlDSHTs-dqFAbmPx9uVeBUiTCwV3/view?usp=sharing -O ./weights/external/droid.pth
[ -f ./thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth ] || uv run gdown --fuzzy https://drive.google.com/file/d/1jnqadPT9kCcO-O2puUcrUtsurXG6v6Fo/view?usp=sharing -O ./thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
[ -f ./_DATA/data/mano/MANO_RIGHT.pkl ] || uv run gdown --fuzzy https://drive.google.com/file/d/1QKy4wIo7uXnNU4VKSJvZF8RtjVzeKc4-/view?usp=sharing -O ./_DATA/data/mano/MANO_RIGHT.pkl
[ -f ./_DATA/data_left/mano_left/MANO_LEFT.pkl ] || uv run gdown --fuzzy https://drive.google.com/file/d/13W1J2kdGNRhJE53sfVQZt_JBDu4Tc5eu/view?usp=sharing -O ./_DATA/data_left/mano_left/MANO_LEFT.pkl

[ -f ./weights/external/detector.pt ] || wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/external/
[ -f ./weights/hawor/checkpoints/hawor.ckpt ] || wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
[ -f ./weights/hawor/checkpoints/infiller.pt ] || wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P ./weights/hawor/checkpoints/
[ -f ./weights/hawor/model_config.yaml ] || wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P ./weights/hawor/

PROJECT_ROOT="$(cd .. && pwd)"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" uv run celery -A pipeline.celery_app worker --loglevel=info -Q stage_5_queue -c 1 -n "hawor-$$@%h"
