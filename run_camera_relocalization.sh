# General variables definition. We leave examples as guide.
EXPERIMENT_DESCRIPTION="short description of experiment for saving purposes"  # e.g. "demo"
SAVE_EXPERIMENT_DESCRIPTION="short description of experiment for saving purposes"  # e.g. "demo"
# Note: The latter variables are usually the same. They can differ if you want to save different 
# PnP results over the same set of correspondences (saved originally in EXPERIMENT_DESCRIPTION). 

MODEL_PATH="/path/to/dense/descriptor/model" 
# Note: Should be the feature-providing model defined in run_prerequisites.sh, L.59

SEQUENCE_ROOT="/path/to/preop/data/specific/directory"  # e.g. "./data_root_pat_1_left/preop_01/1/_start_000230_end_005520"
# Note: Should be the same SEQUENCE_ROOT defined in run_prerequisites.sh, L.6

QUERY_ROOT="/path/to/intraop/data/specific/directory"  # e.g. "./data_root_pat_1_left/step_01/1/_start_000200_end_000945"
BASE_DIR="/path/to/base/saving/directory"  # e.g. "./localization_experiments/pat_1_base_preop_01_query_step_01"

mkdir -p $BASE_DIR"/"$SAVE_EXPERIMENT_DESCRIPTION

python3 -W ignore relocalization.py \
--base_dir $BASE_DIR --sequence_root $SEQUENCE_ROOT --query_sequence_root $QUERY_ROOT \
--exp_description $EXPERIMENT_DESCRIPTION --save_exp_description $SAVE_EXPERIMENT_DESCRIPTION
--mode "all" --gpu_id 0 \
--query_subsampling 1 --reprojection_error 12.0 \
--minimum_response_threshold 0.7 --maximum_response_threshold 1.0 --response_threshold_step 0.001 \
--minimum_point_number 10 --maximum_point_number 100 \
--matching_model_description "short description for base matching model (same as Step 7 in prerequisites)" \
--feature_model_description "short description for feature-providing model (same as Step 7 in prerequisites)" \
--extra_visualizations --erosion_iterations 30 \
--trained_model_path $MODEL_PATH \
--image_downsampling 4.0 --feature_length 256 --filter_growth_rate 8