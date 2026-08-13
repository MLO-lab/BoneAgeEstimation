#!/bin/bash
#SBATCH --job-name=cldice_capillaries_3mo
#SBATCH --time=0-2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --output=./out_and_err/step3-cldice-%j.out
#SBATCH --error=./out_and_err/step3-cldice-%j.err

#SBATCH --mail-user=yihao.liu@dkfz-heidelberg.de
#SBATCH --mail-type=ALL

# Optional: load modules / activate env

mkdir -p out_and_err
mkdir -p results_bone_age

IMS_PATH="data_prediction_metric/210806KK st11 CBMM iEMCN pCap (for Yihao).ims"
LEVEL=0

CH_GT=0
CH_PRED=1
T_GT=1
T_PRED=1

CHUNK_Z=20
CHUNK_Y=64
CHUNK_X=64

ROI_LOW=1
CHUNK_ALPHA=0.005

# clDice params
TAU=2.0
T_COV=0.5
T_MATCH=0.7

# voxel size (zyx)
VZ=2.5
VY=0.7575
VX=0.7575

# tolerant dilation on GT only (set 1 to enable)
DO_TOLERANT=0

# output
OUT_DIR="results_bone_age"
TAG="capillaries_3mo" # the comparison tag 

# flags
REQ_COV_IMG_FOR_TP=""   # keep or remove
USE_S2S=""                                      # set to "--use_skeleton_to_skeleton" if you want it

python structure_based_eval_cldice.py \
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
  $( [ "$DO_TOLERANT" -eq 1 ] && echo "--do_tolerant" ) \
  $REQ_COV_IMG_FOR_TP \
  $USE_S2S \
  --max_examples_per_class 25 \
  --example_strategy borderline \
  --out_dir "$OUT_DIR" \
  --tag "$TAG"