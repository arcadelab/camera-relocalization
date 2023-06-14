import os
import copy
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import pose_utils as pu

from general_utils import quaternion_matrix

# Helper functions
def read_csv_data_single(path_to_csv, error_description):
    """
    Reads CSV files to print mean errors.
    """
    error_data = pd.read_csv(path_to_csv)
    error_residual = error_data[error_description]

    img_idx_list, error_list = [], []
    for ind in error_residual.index:
        try:
            error_list.append(error_residual[ind])
            img_idx_list.append(int(error_data["name"][ind].split(".")[0]))
        except:
            continue

    return img_idx_list, error_list

def read_polaris_poses(trajectory_path):
    """
    Reads ground-truth pose data as formatted in CSV.
    """
    trajectory = np.genfromtxt(trajectory_path, delimiter=",")
    num_poses = int(len(trajectory)/16)
    trajectory = trajectory.reshape((num_poses,4,4))

    return trajectory

def select_images_from_data_source(poses, interval, start_idx, end_idx):
    """
    Subsamples image information as defined in relocalization stage or for evaluation.
    """
    selected_poses = []
    poses_dict = {}
    for i, pose in enumerate(poses):  
        if i % interval == 0 and i >= start_idx and i <= end_idx:
            selected_poses.append(pose)
            poses_dict["%08i.jpg" % i] = [pose[:3,:3], pose[:3, 3]]

    return np.asarray(selected_poses), poses_dict

def subsample_poses(poses, indexes, intervals):
    return select_images_from_data_source(poses, interval=intervals, start_idx=indexes[0], end_idx=indexes[1])

def subsample_preop_poses(poses, visible_indexes):
    """
    Subsamples preoperative image pose information as defined by visible images in reconstruction.
    """
    selected_poses = []
    poses_dict = {}
    for i, pose in enumerate(poses):  
        if i in visible_indexes:
            selected_poses.append(pose)
            poses_dict["%08i.jpg" % i] = [pose[:3,:3], pose[:3, 3]]

    return np.asarray(selected_poses), poses_dict

def get_extrinsic_matrix(poses):
    """
    Reformats poses as matrices.
    """
    extrinsic_matrices = []
    visible_view_count = len(poses)
    for i in range(visible_view_count):
        rigid_transform = quaternion_matrix(
            [poses["poses[" + str(i) + "]"]['orientation']['w'], poses["poses[" + str(i) + "]"]['orientation']['x'],
             poses["poses[" + str(i) + "]"]['orientation']['y'],
             poses["poses[" + str(i) + "]"]['orientation']['z']])
        rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
        rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
        rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']
        transform = np.asmatrix(rigid_transform)
        extrinsic_matrices.append(transform)   

    return np.asarray(extrinsic_matrices)

def read_colmap_trajectory(trajectory_path):
    """
    Reads and formats preoperative trajectory as estimated by COLMAP.
    """
    stream = open(trajectory_path, 'r')
    doc = yaml.safe_load(stream)
    _, values = doc.items()
    poses = values[1]    
    mat_poses = get_extrinsic_matrix(poses)

    return mat_poses

def invert_poses(poses):
    """
    Invert reference coordinate system of COLMAP estimated poses.
    """
    copy_poses = copy.deepcopy(poses)
    for i, pose in enumerate(copy_poses):
        pose[:3, :3] = np.transpose(poses[i, :3, :3])
        pose[:3, 3] = np.matmul(-np.transpose(poses[i, :3, :3]), poses[i, :3, 3])

    return copy_poses

def match_poses(reference_names, source_names, source_poses):
    """
    Finds the poses indicated by reference_names inside source_names. 
    Returns the name and poses of the matching camera instances. 
    """
    matching_names = []
    matching_poses = []
    for name in reference_names:
        idx = source_names.index(name)
        matching_names.append(name)
        matching_poses.append(source_poses[idx])

    return matching_names, np.asarray(matching_poses)

def read_visible_indexes(visible_indexes_path):
    """
    Reads indexes corresponding to images contributing to reconstruction.
    As defined in https://github.com/lppllppl920/DenseReconstruction-Pytorch.
    """
    selected_indexes = []
    with open(visible_indexes_path) as fp:
        for line in fp:
            selected_indexes.append(int(line))

    return selected_indexes


# Pre-processing functions
def load_preop_polaris_csv_in_mm(poses_path, visible_indexes):
    """
    Pre-processing of tracking data for preoperative sequence.
    Reads, samples and scales translation component to millimeter scale.
    """
    pol_poses = read_polaris_poses(poses_path)
    pol_poses, camera_names = subsample_preop_poses(pol_poses, visible_indexes=visible_indexes)
    pol_poses_mm = pu.PoseRegistration.scale_poses(pol_poses, 1000)

    return pol_poses_mm, list(camera_names.keys())

def load_polaris_csv_in_mm(poses_path, indexes, intervals):
    """
    Pre-processing of tracking data for intraoperative sequence.
    Reads, samples and scales translation component to millimeter scale.
    """
    pol_poses = read_polaris_poses(poses_path)
    pol_poses, camera_names = subsample_poses(pol_poses, indexes=indexes, intervals=intervals)
    pol_poses_mm = pu.PoseRegistration.scale_poses(pol_poses, 1000)

    return pol_poses_mm, list(camera_names.keys())

def load_colmap_yaml(poses_path):
    """
    Pre-processing of COLMAP preoperative trajectory data.
    """
    colmap_poses = read_colmap_trajectory(poses_path)
    colmap_poses = invert_poses(colmap_poses)

    return colmap_poses

def load_poses_from_pth(poses_path):
    """
    Reads and formats estimated poses from relocalization stage.
    """
    poses_dict = torch.load(poses_path)
    poses_names = []
    poses = []
    for name in poses_dict:
        r, t = poses_dict[name]
        poses_names.append(name)
        pose = np.zeros((4, 4))
        pose[:3, :3] = r
        pose[:3, 3] = t
        pose[3, 3] = 1
        poses.append(pose)

    return poses_names, np.asarray(poses)
    

# Main function
def main(args):
    """
    Performs registration between COLMAP and tracking preoperative trajectories.
    Obtains transformation parameters between coordinate systems, enabling evaluation.
    Performs sequential filtering stages, visualizes trajectories and computes error metrics.
    """
    # Data loading
    COLMAP_POINT_CLOUD = os.path.join(args.sequence_root, "colmap", "0", "structure.ply")
    ANATOMY_POINT_CLOUD = os.path.join(args.sequence_root, "fused_mesh.ply")

    PRE_POLARIS_POSES_PATH = args.preop_tracking_poses_path
    PREOP_VIS_INDEXES_PATH = os.path.join(args.sequence_root, "colmap", "0", "visible_view_indexes")
    preop_visible_indexes = read_visible_indexes(visible_indexes_path=PREOP_VIS_INDEXES_PATH)
    pre_pol_poses_mm, _ = load_preop_polaris_csv_in_mm(poses_path=PRE_POLARIS_POSES_PATH, 
                                                       visible_indexes=preop_visible_indexes)

    PRE_COLMAP_TRAJECTORY_PATH = os.path.join(args.sequence_root, "colmap", "0", "motion.yaml")    
    pre_colmap_poses = load_colmap_yaml(poses_path=PRE_COLMAP_TRAJECTORY_PATH)

    INTRA_POLARIS_POSES_PATH = args.intraop_tracking_poses_path
    intra_pol_poses_mm, intra_names = load_polaris_csv_in_mm(poses_path=INTRA_POLARIS_POSES_PATH, 
                                                             indexes=(args.start_intraop_img_seq_idx, 
                                                                      args.end_intraop_img_seq_idx), 
                                                             intervals=args.intraop_img_seq_interval)

    PREDICTIONS = \
        os.path.join(args.base_dir, args.exp_description, "raw_query_pose_estimations.pth")

    predicted_names, predicted_poses = load_poses_from_pth(PREDICTIONS)
    original_indexes = list(range(args.start_intraop_img_seq_idx, args.end_intraop_img_seq_idx, args.intraop_img_seq_interval))
    original_indexes = original_indexes[::1]
    ss_predicted_names, ss_predicted_poses = [], []

    for p_name, p_pose in zip(predicted_names, predicted_poses):
        p_idx = int(p_name.split(".")[0])
        if p_idx in original_indexes:
            ss_predicted_names.append(p_name)
            ss_predicted_poses.append(p_pose)

    # Registration stage
    preop_registration = pu.PoseRegistration()
    preop_registration.register_poses(source_poses=pre_colmap_poses, target_poses=pre_pol_poses_mm)

    pre_colmap_in_polaris_frame = preop_registration.transform_to_internal_target_frame(poses=pre_colmap_poses)

    # Filtering stage
    filtered_predictions = []
         
    # Anatomical prior filter
    pc_scale = pu.compute_pc_scale(pre_colmap_poses, COLMAP_POINT_CLOUD)
    anatomy_filter = pu.MeshPoseFilter(sampled_mesh_path=ANATOMY_POINT_CLOUD)
    anatomy_filtered = []
    filtered_names = []
    for i, pose in enumerate(ss_predicted_poses):
        include_pose = anatomy_filter.apply(pose=pose, scale_to_unit=pc_scale)
        if include_pose:
            anatomy_filtered.append(pose)
            filtered_names.append(ss_predicted_names[i])

    filtered_predictions = np.asarray(anatomy_filtered)

    # Median filter
    median_filtered = []
    filter_size = args.filter_size
    pose_filter = pu.MedianPoseFilter(filter_size)
    buffer = pu.TemporalPoseBuffer(size=filter_size)

    for i, pose in enumerate(filtered_predictions):        
        buffer.put_pose(pose)
        median_filtered.append(pose)
        pose_index =  i - filter_size // 2       
        if not buffer.is_full():
            continue
        filtered_pose = pose_filter.apply(pose_sequence=buffer.get_buffer_array())
        median_filtered[pose_index] = filtered_pose

    filtered_predictions = np.asarray(median_filtered)

    # Distance-based filter    
    distance_filter = pu.DistanceBasedFilter(preop_poses=pre_colmap_poses)
    distance_filtered = []
    distance_filtered_names = []
    for i, pose in enumerate(filtered_predictions):
        include_pose = distance_filter.apply(pose=pose)
        if include_pose:
            distance_filtered.append(pose)
            distance_filtered_names.append(filtered_names[i])

    filtered_predictions = np.asarray(distance_filtered)

    # Transforms filtered and original estimated poses in tracking system coordinate frame
    predictions_in_polaris_frame = preop_registration.transform_to_internal_target_frame(poses=filtered_predictions)

    raw_predictions_in_polaris_frame = preop_registration.transform_to_internal_target_frame(poses=ss_predicted_poses)
    
    # Obtains ground-truth poses for localized instances.
    localized_names, localized_intra_polaris_poses_in_mm = \
        match_poses(reference_names=distance_filtered_names, source_names=intra_names, source_poses=intra_pol_poses_mm)

    if args.extra_visualizations:
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        raw_save_dir = args.save_dir + "_raw"
        if not os.path.exists(raw_save_dir):
            os.makedirs(raw_save_dir)

        pu.PoseRegistration.plot_and_save_trajectory(pu.PoseRegistration.scale_poses(pre_colmap_poses, pc_scale), 
                                                     axlen=0.1 * pc_scale, subsampling_factor=1, 
                                                     save_name=os.path.join(save_dir, "preop_colmap_in_mesh_scale.ply"))

        pu.PoseRegistration.plot_and_save_trajectory(pu.PoseRegistration.scale_poses(filtered_predictions, pc_scale),  
                                                     axlen=0.1 * pc_scale, subsampling_factor=1, 
                                                     save_name=os.path.join(save_dir, "predictions_in_mesh_scale.ply"))

        pu.PoseRegistration.plot_and_save_trajectory(pre_colmap_in_polaris_frame,  
                                                     axlen=1, subsampling_factor=1, 
                                                     save_name=os.path.join(save_dir, "preop_colmap_in_tracking_frame.ply"))

        pu.PoseRegistration.plot_and_save_trajectory(pre_pol_poses_mm,  
                                                     axlen=1, subsampling_factor=1, 
                                                     save_name=os.path.join(save_dir, "preop_tracked_poses_in_mm.ply"))

        pu.PoseRegistration.plot_and_save_trajectory(intra_pol_poses_mm,  
                                                     axlen=1, subsampling_factor=4, 
                                                     save_name=os.path.join(save_dir, "intraop_tracked_poses_in_mm.ply"))

        pu.PoseRegistration.plot_and_save_trajectory(predictions_in_polaris_frame,  
                                                     axlen=1, subsampling_factor=1, 
                                                     save_name=os.path.join(save_dir, "predictions_in_tracking_frame.ply"))

        pu.PoseRegistration.plot_and_save_trajectory(raw_predictions_in_polaris_frame,  
                                                     axlen=1, subsampling_factor=1, 
                                                     save_name=os.path.join(raw_save_dir, "predictions_in_tracking_frame.ply"))    
        
        pu.PoseRegistration.plot_and_save_trajectory(localized_intra_polaris_poses_in_mm,  
                                                     axlen=1, subsampling_factor=1, 
                                                     save_name=os.path.join(save_dir, "localized_intraop_tracked_poses_in_mm.ply"))

    # Evaluation
    error_metric = pu.RelocErrorMetric()
    error_names = ["x", "y", "z", "diff_trans", "diff_rot", "diff_rad", "euler_est", "euler_gt", "euler_residual"]
    analizer = pu.PoseAnalysis(error_funct=error_metric, metrics_names=error_names)
    analizer.wokring_dir = save_dir
    analizer.set_poses(camera_names=localized_names, gt_poses=localized_intra_polaris_poses_in_mm, predicted_poses=predictions_in_polaris_frame)
    analizer.save_poses()
    analizer.save_errors()

    _, t_error_list = read_csv_data_single(path_to_csv=os.path.join(save_dir, "errors.csv"), error_description="diff_trans")
    _, rot_error_list = read_csv_data_single(path_to_csv=os.path.join(save_dir, "errors.csv"), error_description="diff_rad")

    print("-"*50)
    print("Translation error: {} +/- {}".format(np.mean(t_error_list), np.std(t_error_list)))
    print("Rotation error: {} +/- {}".format(np.mean(rot_error_list), np.std(rot_error_list)))
    print("Localization %: {}".format((len(t_error_list)/249)*100))
    print("-"*50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trajectory filtering and evaluation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--sequence_root", type=str, required=True,
                        help="Directory root of the preoperative video sequence")
    parser.add_argument("--query_sequence_root", type=str, required=True,
                        help="Directory root of the intraoperative video sequence")
    parser.add_argument("--save_dir", type=str, required=True, 
                        help="Save directory") 
    parser.add_argument("--filter_size", type=int, default=7, 
                        help="Size of Median filter")
    parser.add_argument("--preop_tracking_poses_path", type=str, required=True,
                        help="Path to preoperative tracked poses (CSV format)")
    parser.add_argument("--intraop_tracking_poses_path", type=str, required=True,
                        help="Path to intraoperative tracked poses (CSV format)")
    parser.add_argument("--preop_colmap_poses_path", type=str, required=True,
                        help="Path to preoperative COLMAP poses (YAML format)")
    parser.add_argument("--start_preop_img_seq_idx", type=int, required=True,
                        help="Start index of preoperative image sequence (including)")
    parser.add_argument("--end_preop_img_seq_idx", type=int, required=True,
                        help="End index of preoperative image sequence (including)")
    parser.add_argument("--preop_img_seq_interval", type=int, required=True,
                        help="Subsampling interval of preoperative image sequence")
    parser.add_argument("--start_intraop_img_seq_idx", type=int, required=True,
                        help="Start index of intraoperative image sequence (including)")
    parser.add_argument("--end_intraop_img_seq_idx", type=int, required=True,
                        help="End index of intraoperative image sequence (including)")
    parser.add_argument("--intraop_img_seq_interval", type=int, required=True,
                        help="Subsampling interval of intraoperative image sequence")
    parser.add_argument('--extra_visualizations', action='store_true', 
                        help='Save additional visualizations (not strictly necessary for filtering or evaluation)')

    args = parser.parse_args()

    main(args=args)