# General variables definition. We leave examples as guide.
LOG_ROOT="/path/to/logging/directory"  # e.g. "./log_root_pat_1_left"
PRECOMPUTE_ROOT="/path/to/precompute/directory"  # e.g. "./precompute_root_pat_1_left"

DATA_ROOT="/path/to/data/root/directory"  # e.g. "./data_root_pat_1_left/preop_01"
SEQUENCE_ROOT="/path/to/data/specific/directory"  # e.g. "./data_root_pat_1_left/preop_01/1/_start_000230_end_005520"

SEQUENCE_DESCRIPTION="description for saving purposes"  # e.g. "pat_1_preop"
PAT_ID="patient id"  # e.g. 1

GPU_ID="gpu id"  # e.g. 0


# 1) Dense feature matching. Assumes a trained dense descriptor model as presented in
# https://github.com/lppllppl920/DenseDescriptorLearning-Pytorch.
/path/to/python feature_correspondence_generation.py \
--image_downsampling 4.0 --network_downsampling 64 --input_size 256 384 --batch_size 1 --num_workers 1 \
--data_root $DATA_ROOT --precompute_root $PRECOMPUTE_ROOT \
--sequence_root $SEQUENCE_ROOT \
--matching_trained_model_path "/path/to/dense/descriptor/model" \
--matching_model_description "short description for trained model (one word recommended)" \
--feature_length 256 --filter_growth_rate 8 --max_feature_detection 3000 --cross_check_distance 3.0 --patient_id $PAT_ID \
--gpu_id $GPU_ID --temporal_range 30 --test_keypoint_num 200 --residual_threshold 5.0 --octave_layers 8 --contrast_threshold 5e-5 \
--edge_threshold 100 --sigma 1.1 --skip_interval 5 --min_inlier_ratio 0.2 --hysterisis_factor 0.7

# 2) New COLMAP database creation. As presented in
# https://github.com/lppllppl920/DenseReconstruction-Pytorch.
/path/to/python colmap_database_creation.py \
--sequence_root $SEQUENCE_ROOT --overwrite_database

# 3) Sparse reconstruction with dense feature matches. As presented in
# https://github.com/lppllppl920/DenseReconstruction-Pytorch.
/path/to/python colmap_sparse_reconstruction.py \
--colmap_exe_path "colmap" \
--sequence_root $SEQUENCE_ROOT \
--overwrite_reconstruction

# 4) COLMAP model format conversion with dense feature matches. As presented in
# https://github.com/lppllppl920/DenseReconstruction-Pytorch.
/path/to/python colmap_model_converter.py \
--colmap_exe_path "colmap" \
--sequence_root $SEQUENCE_ROOT \
--overwrite_output

# 5) Selection of best COLMAP result (if multiple computed in previous steps).
/path/to/python select_best_colmap_folder.py \
--sequence_root $SEQUENCE_ROOT

cp "$SEQUENCE_ROOT/colmap/0/camera_intrinsics_per_view" "$SEQUENCE_ROOT"

# 6) Save reconstruction data in database.
/path/to/python database_utils.py \
--sequence_root $SEQUENCE_ROOT \
--descriptor_type "dense" \
--matching_model_description "short description for trained model (same as Step 1)"

# 7) Feature database filling
# Note that base matching and feature-providing model are equivalent in this case.
FILL_MODEL="/path/to/dense/descriptor/model"

/path/to/python feature_population.py \
--image_downsampling 4.0 --network_downsampling 64 --sequence_root $SEQUENCE_ROOT \
--feature_trained_model_path $FILL_MODEL \
--feature_length 256 --filter_growth_rate 8 --gpu_id $GPU_ID \
--matching_model_description "short description for trained model (same as Step 1)" \
--feature_model_description "short description for feature-providing model"

# 8) Train depth estimation. As presented in 
# https://github.com/lppllppl920/EndoscopyDepthEstimation-Pytorch.
/path/to/python train_depth_estimation.py \
--adjacent_range 5 30 --image_downsampling 4.0 --network_downsampling 64 --input_size 256 320 --batch_size 2 --num_workers 1 --slp_weight 1.0 \
--dcl_weight 0.5 --sfl_weight 2.0 --dl_weight 0.05 --lr_range 1.0e-4 1.0e-3 --inlier_percentage 0.9 --display_interval 20 --visible_interval 5 \
--save_interval 5 --training_patient_id $PAT_ID --num_epoch 150 --num_iter 1000 --display_architecture \
--data_root $DATA_ROOT \
--load_trained_model --trained_model_path "/path/to/previous/trained/depth/estimation/model" \
--log_root $LOG_ROOT --precompute_root $PRECOMPUTE_ROOT \
--descriptor_model_path "/path/to/dense/descriptor/model" \
--sequence_description $SEQUENCE_DESCRIPTION

# 9) Fusion data generation. As presented in 
# https://github.com/lppllppl920/DenseReconstruction-Pytorch.
/path/to/python fusion_data_generation.py \
--image_downsampling 4.0 --network_downsampling 64 --input_size 256 384 --batch_size 1 --num_workers 1 \
--visible_interval 5 --inlier_percentage 0.9 \
--trained_model_path "/path/to/previous/trained/depth/estimation/model (from Step 9)" \
--data_root $DATA_ROOT \
--sequence_root $SEQUENCE_ROOT \
--patient_id $PAT_ID --precompute_root $PRECOMPUTE_ROOT

# 10) Surface reconstruction. As presented in
# https://github.com/lppllppl920/DenseReconstruction-Pytorch.
/path/to/python surface_reconstruction.py \
--data_root ${DATA_ROOTS[$i]} --visualize_fused_model \
--trunc_margin_multiplier 10.0 --patient_id $PAT_ID --max_voxel_count 64e6 \
--sequence_root ${SEQUENCE_ROOTS[$i]}