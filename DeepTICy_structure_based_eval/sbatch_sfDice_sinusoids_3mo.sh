#!/bin/bash
#SBATCH --job-name=sfdice_sinusoids
#SBATCH --time=0-2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=./out_and_err/step3-sfdice-%j.out
#SBATCH --error=./out_and_err/step3-sfdice-%j.err
#SBATCH --mail-user=yihao.liu@dkfz-heidelberg.de
#SBATCH --mail-type=ALL

set -euo pipefail

mkdir -p out_and_err
mkdir -p results_bone_age

IMS_PATH="data_prediction_metric/190429KK st6 CBMM E1  iCD105_pSinusoids_miCD105.ims"
LEVEL=0

CH_GT=0
CH_PRED=1

CHUNK_Z=20
CHUNK_Y=64
CHUNK_X=64

ROI_LOW=1
CHUNK_ALPHA=0.01

# sfDice params
TAU=2.0
T_COV=0.5
T_MATCH=0.6

# voxel size (zyx)
VZ=2.5
VY=0.7575
VX=0.7575

# tolerant dilation on GT only (set 1 to enable)
DO_TOLERANT=0

# output
OUT_DIR="results_bone_age"
BASE_TAG="sinusoids_3mo"

# Optional flag: require cov_img for TP (set to 1 to enable)
REQ_COV_IMG_FOR_TP=0

# Grid values
T_GT=10
T_PRED=10

# Build optional args
EXTRA_ARGS=()
if [ "$DO_TOLERANT" -eq 1 ]; then
  EXTRA_ARGS+=("--do_tolerant")
fi
if [ "$REQ_COV_IMG_FOR_TP" -eq 1 ]; then
  EXTRA_ARGS+=("--req_cov_img_for_tp")
fi

# Grid search
TAG="${BASE_TAG}_tgt${T_GT}_tpred${T_PRED}"
echo "=== Running: t_gt=${T_GT}, t_pred=${T_PRED}, tag=${TAG} ==="

python structure_based_eval_sfdice.py \
  --ims_path "$IMS_PATH" \
  --level "$LEVEL" \
  --ch_gt "$CH_GT" \
  --ch_pred "$CH_PRED" \
  --t_gt "$T_GT" \
  --t_pred "$T_PRED" \
  --chunk_z "$CHUNK_Z" --chunk_y "$CHUNK_Y" --chunk_x "$CHUNK_X" \
  --roi_low "$ROI_LOW" \
  --chunk_alpha "$CHUNK_ALPHA" \
  --tau "$TAU" \
  --T_cov "$T_COV" \
  --T_match "$T_MATCH" \
  --vz "$VZ" --vy "$VY" --vx "$VX" \
  "${EXTRA_ARGS[@]}" \
  --out_dir "$OUT_DIR" \
  --tag "$TAG"
