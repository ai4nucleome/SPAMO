#!/usr/bin/env bash
# ================================================================
#  SpaMO — Best-Only Reproduction
# ================================================================
set -euo pipefail

cd "$(dirname "$0")"

SEED=${1:-2025}
DATA_ROOT=${DATA_ROOT:-./Data}
OUT=./results/

mkdir -p "${OUT}/HLN" "${OUT}/Mouse_Brain" "${OUT}/Simulation"

echo "========================================================"
echo "  Running SpaMO best configurations with seed=${SEED}"
echo "========================================================"

## HLN - RNA + ADT
python main.py \
  --file_fold "${DATA_ROOT}/HLN" --data_type 10x \
  --n_clusters 10 --init_k 10 --KNN_k 20 \
  --RNA_weight 5 --ADT_weight 5 \
  --dgi_weight 0.3 --spatial_weight 0.01 \
  --epochs_override 200 \
  --optimizer_type adam --lr_scheduler_type cosine \
  --random_seed "${SEED}" \
  --vis_out_path "${OUT}/HLN/spamo.png" \
  --txt_out_path "${OUT}/HLN/labels.txt" \
# eval
python cal_matrics.py \
  --GT_path "${DATA_ROOT}/HLN/GT_labels.txt" \
  --our_path "${OUT}/HLN/labels.txt" \
  --save_path "${OUT}/HLN/metrics.txt"

## Mouse Brain - RNA + ATAC
python main.py \
  --file_fold "${DATA_ROOT}/Mouse_Brain" --data_type Spatial-epigenome-transcriptome \
  --n_clusters 14 --init_k 14 --KNN_k 20 \
  --RNA_weight 1 --ADT_weight 10 \
  --dgi_weight 0.1 --spatial_weight 0.01 \
  --epochs_override 400 \
  --optimizer_type adamw --lr_scheduler_type cosine \
  --random_seed "${SEED}" \
  --vis_out_path "${OUT}/Mouse_Brain/spamo.png" \
  --txt_out_path "${OUT}/Mouse_Brain/labels.txt" \
# eval
python cal_matrics.py \
  --GT_path "${DATA_ROOT}/Mouse_Brain/MB_cluster.txt" \
  --our_path "${OUT}/Mouse_Brain/labels.txt" \
  --save_path "${OUT}/Mouse_Brain/metrics.txt"

## Simulation - RNA + ADT + ATAC
python main.py \
  --file_fold "${DATA_ROOT}/Simulation" --data_type Simulation \
  --n_clusters 5 --init_k 5 --KNN_k 20 \
  --RNA_weight 5 --ADT_weight 5 \
  --dgi_weight 0.1 --spatial_weight 0.01 \
  --epochs_override 200 \
  --optimizer_type adam \
  --random_seed "${SEED}" \
  --vis_out_path "${OUT}/Simulation/spamo.png" \
  --txt_out_path "${OUT}/Simulation/labels.txt" \
# eval
python cal_matrics.py \
  --GT_path "${DATA_ROOT}/Simulation/GT_1.txt" \
  --our_path "${OUT}/Simulation/labels.txt" \
  --save_path "${OUT}/Simulation/metrics.txt"

echo "========================================================"
echo "  DONE — results in ${OUT}/"
echo "========================================================"
