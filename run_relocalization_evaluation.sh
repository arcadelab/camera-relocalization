# General variables definition. We leave examples as guide.
SEQUENCE_ROOT="/path/to/preop/data/specific/directory"  # e.g. "./data_root_pat_1_left/preop_01/1/_start_000230_end_005520"
# Note: Corresponds to the preoperative root. Should be the same SEQUENCE_ROOT defined in run_camera_relocalization.sh, L.10 and run_prerequisites.sh, L.6

QUERY_ROOT="/path/to/intraop/data/specific/directory"  # e.g. "./data_root_pat_1_left/step_01/1/_start_000200_end_000945"
# Note: Corresponds to the intraoperative root. Should be the same QUERY_ROOT defined in run_camera_relocalization.sh, L.13

SAVE_DIR="/path/to/base/saving/directory"  # e.g. "./localization_predictions/pat_1_base_preop_01_query_step_01/demo_results"

python3 -W ignore relocalization_postprocessing_evaluation.py \
--sequence_root $SEQUENCE_ROOT --query_sequence_root $QUERY_ROOT \
--save_dir $SAVE_DIR --filter_size 7 \
--preop_tracking_poses_path "/path/to/preoperative/tracked/poses/csv" \
--intraop_tracking_poses_path "/path/to/intraoperative/tracked/poses/csv" \
--preop_colmap_poses_path $SEQUENCE_ROOT"/colmap/0/motion.yaml" \
--start_preop_img_seq_idx 230 --end_preop_img_seq_idx 5520 --preop_img_seq_interval 5 \
--start_intraop_img_seq_idx 200 --end_intraop_img_seq_idx 945 --intraop_img_seq_interval 3