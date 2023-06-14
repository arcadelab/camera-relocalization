import os
import cv2
import json
import time
import torch
from tqdm import tqdm

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import general_utils
import database_utils as db
import visualization_utils
import localization_database_utils

# Helper functions
def invert_extrinsic_matrix(extrinsic_rot, extrinsic_t, global_scale=1):
    """
    Convert the extrinsic matrix to T^(world)_(camera)
    """
    temp_extrinsic = np.zeros((4, 4))
    temp_extrinsic[:3, :3] = extrinsic_rot
    temp_extrinsic[:3, 3] = extrinsic_t
    temp_extrinsic[3, 3] = 1.0

    training_extrinsic_1 = np.zeros_like(temp_extrinsic)
    training_extrinsic_1[:3, :3] = np.transpose(temp_extrinsic[:3, :3])
    training_extrinsic_1[:3, 3] = np.matmul(-np.transpose(temp_extrinsic[:3, :3]), temp_extrinsic[:3, 3]) / global_scale
    training_extrinsic_1[3, 3] = 1.0
    training_extrinsic_1 = general_utils.type_float_and_reshape(training_extrinsic_1, (4, 4))

    return training_extrinsic_1


def adaptative_pose_estimation_PnP(image_name, point_responses, points_3d, points_2d, 
                                   response_threshold_interval, threshold_step, point_number_interval, 
                                   reprojection_error, point_info_dict, pose_save_dict,
                                   camera_mtx=None, dist_coeffs=None):
    """
    Solves PnP problem from a set of corresponding 2D and 3D points.
    """
    if points_2d.shape[0] != points_3d.shape[0]:
        print("Equal number of initial object and image points needed")
        return [], [], []

    minimum_condition = False
    eval_response_threshold = response_threshold_interval[1]

    while minimum_condition is False:
        projection_indexes = np.where(point_responses.astype(np.float64) >= eval_response_threshold)

        if (len(projection_indexes[0]) >= point_number_interval[0] and len(projection_indexes[0]) <= point_number_interval[1]) or eval_response_threshold <= response_threshold_interval[0]:
            minimum_condition = True
        else:
            eval_response_threshold -= threshold_step
    
    if (len(projection_indexes[0]) >= point_number_interval[0] and len(projection_indexes[0]) <= point_number_interval[1]):
        points_2d = points_2d[projection_indexes]
        points_3d = points_3d[projection_indexes]

        if dist_coeffs is None:
            dist_coeffs = np.zeros((5,1))

        points_2d = points_2d.reshape((points_2d.shape[0], 1, 2)).astype(np.float32)
        points_3d = points_3d.reshape((points_3d.shape[0], 1, 3)).astype(np.float64)

        ret, rot, t, inliers = cv2.solvePnPRansac(objectPoints=points_3d, imagePoints=points_2d, 
                                                  cameraMatrix=camera_mtx, distCoeffs=dist_coeffs,
                                                  reprojectionError=reprojection_error, flags=cv2.SOLVEPNP_EPNP)
        
        if ret:
            orientation = R.from_rotvec(rot.reshape((3)))
            orientation = orientation.as_matrix()
            t = t.reshape((3))
            
            inverted_extrinsic = invert_extrinsic_matrix(extrinsic_rot=orientation, extrinsic_t=t)
            point_info_dict[image_name] = (points_2d.shape[0], len(inliers), eval_response_threshold)
            pose_save_dict[image_name] = (inverted_extrinsic[:3, :3], inverted_extrinsic[:3, 3])

            return ret, point_info_dict, pose_save_dict

        else:
            return None, point_info_dict, pose_save_dict

    else:
        return None, point_info_dict, pose_save_dict


# Main functions
def image_based_localize_and_compute_correspondences(args, query_image_path, model,
                                                     preop_feature_array, preop_world_coordinates,
                                                     preop_image_names, preop_image_coordinates,
                                                     undistorted_mask_boundary, undistorted_mask_boundary_path,
                                                     localization_db_path):
    """
    Performs Keypoint Relocalization on selected intraoperative image.
    """
    query_feature_map, query_starts_h, query_starts_w = \
        general_utils.image_dataloader(args=args, model=model, query_image_path=query_image_path,
                                       undistorted_mask_boundary_path=undistorted_mask_boundary_path)

    response_array = general_utils.feature_localization(query_feature_map=query_feature_map,
                                                        preop_feature_array=torch.from_numpy(preop_feature_array),
                                                        gpu_id=args.gpu_id)
    
    del query_feature_map

    response_array[:,1] = args.image_downsampling * (response_array[:,1] + query_starts_w)
    response_array[:,2] = args.image_downsampling * (response_array[:,2] + query_starts_h)

    # Disregard low convolution response correspondences
    projection_indexes = np.where(response_array[:, 0] > args.minimum_response_threshold)
    points2d = response_array[:,1:3][projection_indexes]
    points3d = preop_world_coordinates[projection_indexes]
    point_responses = response_array[:, 0][projection_indexes]

    filtered_preop_image_names = preop_image_names[projection_indexes]
    filtered_preop_image_coordinates = preop_image_coordinates[projection_indexes]

    # Mask correspondences outside endoscope's FoV
    valid_indexes = []
    points2d = points2d.astype(np.int)
    for j, point2d in enumerate(points2d):
        if undistorted_mask_boundary[point2d[0], point2d[1]] > 0:
            valid_indexes.append(j)

    points2d_in = points2d[np.asarray(valid_indexes).astype(np.int)]
    points3d_in = points3d[np.asarray(valid_indexes).astype(np.int)]

    point_responses_in = point_responses[np.asarray(valid_indexes).astype(np.int)]

    filtered_preop_image_names = filtered_preop_image_names[np.asarray(valid_indexes).astype(np.int)]
    filtered_preop_image_coordinates = filtered_preop_image_coordinates[np.asarray(valid_indexes).astype(np.int)]

    localization_database_utils.insert_image_correspondences(db_path=localization_db_path,
                                                             image_name=os.path.basename(query_image_path),
                                                             responses=point_responses_in,
                                                             points_2d=points2d_in,
                                                             points_3d=points3d_in,
                                                             preop_image_names=filtered_preop_image_names,
                                                             preop_image_coords=filtered_preop_image_coordinates)

def localize_correspondences(args):
    """
    Performs Keypoint Relocalization over all intraoperative images and obtains 2D-3D correspondences.
    """
    start = time.process_time()

    # Definition of query image list
    query_image_names = general_utils.get_all_color_image_names_in_sequence(Path(args.query_sequence_root))
    query_image_names = [str(j) for j in query_image_names]
    query_image_names = query_image_names[::args.query_subsampling]
    
    # Reads base and query sequence database file
    dreco_db_path = \
        os.path.join(args.sequence_root, "dreco_base_{}_fill_{}.db".format(args.matching_model_description, 
                                                                           args.feature_model_description))
    rdb = db.ReconstrunctionDB(db_path=dreco_db_path)

    # Correspondence saving directory handling
    SAVE_DIR = os.path.join(args.base_dir, args.exp_description)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Correspondence database creation and setup
    localization_db_path = os.path.join(SAVE_DIR, "localization.db")
    localization_database_utils.create_localization_database(db_file_path=localization_db_path)
    conn = localization_database_utils.create_connection(db_file=localization_db_path)
    localization_database_utils.create_table(conn=conn)

    # Saving commandline args for future lookup
    with open(os.path.join(SAVE_DIR, "commandline_args"), 'w') as f:
        f.write("script: {}".format(str(os.path.realpath(__file__))))
        json.dump(args.__dict__, f, indent=2)

    # Retrieval of feature database
    print("Retrieving pre-operative feature and 3D point database...")
    preop_image_names, preop_image_coordinates, preop_feature_array, preop_world_coordinates = rdb.get_all_point3d_features()
    original_feat_number = preop_feature_array.shape[0]
    print("Retrieved {} features in database!".format(original_feat_number))

    # Imports mask for point correspondence rejection outside endoscope's FoV
    undistorted_mask_boundary_path = os.path.join(args.sequence_root, "undistorted_mask.bmp")
    undistorted_mask_boundary = cv2.imread(undistorted_mask_boundary_path, cv2.IMREAD_GRAYSCALE)
    undistorted_mask_boundary = undistorted_mask_boundary.T

    # Erode the boundary to remove near-boundary matches
    kernel = np.ones((5, 5), np.uint8)
    undistorted_mask_boundary = cv2.erode(undistorted_mask_boundary, kernel, iterations=args.erosion_iterations)

    # Loads dense descriptor network for correspondence compute
    model = general_utils.load_pretrained_model(feature_descriptor_model_path=Path(args.trained_model_path), 
                                                filter_growth_rate=args.filter_growth_rate, 
                                                feature_length=args.feature_length, 
                                                gpu_id=args.gpu_id)

    print("Feature localization and correspondence computing...")
    tq = tqdm(total=len(query_image_names))
    for query_image_path in query_image_names:
        
        query_image_index = os.path.basename(query_image_path).split(".")[0]
        tq.set_description("Query image index: {}".format(int(query_image_index)))

        image_based_localize_and_compute_correspondences(args=args, query_image_path=query_image_path, model=model,
                                                         preop_feature_array=preop_feature_array, 
                                                         preop_world_coordinates=preop_world_coordinates,
                                                         preop_image_names=preop_image_names, 
                                                         preop_image_coordinates=preop_image_coordinates,
                                                         undistorted_mask_boundary=undistorted_mask_boundary,
                                                         undistorted_mask_boundary_path=undistorted_mask_boundary_path,
                                                         localization_db_path=localization_db_path)

        tq.update(1)
    tq.close()
    
    print("Feature localization for {} images in {} seconds".format(len(query_image_names), time.process_time() - start))

def correct_points_and_solve_PnP(args):
    """
    Solves PnP problem with 2D-3D correspondences and performs pose estimation.
    """
    start = time.process_time()

    # PnP saving directory handling (branch from correspondence saving directory)
    SAVE_DIR = os.path.join(args.base_dir, args.save_exp_description)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Saving commandline args for future lookup
    with open(os.path.join(SAVE_DIR, "commandline_args"), 'w') as f:
        f.write("script: {}".format(str(os.path.realpath(__file__))))
        json.dump(args.__dict__, f, indent=2)
    
    # Reads database files
    dreco_db_path = \
        os.path.join(args.sequence_root, "dreco_base_{}_fill_{}.db".format(args.matching_model_description, args.feature_model_description))
    rdb = db.ReconstrunctionDB(db_path=dreco_db_path)

    localization_db_path = os.path.join(args.base_dir, args.exp_description, "localization.db")
    query_image_names = localization_database_utils.retrieve_query_image_names(db_path=localization_db_path)
    query_image_names = [i[0] for i in query_image_names]

    # Extracts camera intrinsic parameters from base reconstruction (given by COLMAP)
    camera_params = rdb.get_camera_params()
    camera_params = camera_params[0]
    camera_mtx = np.array([[camera_params[0], 0, camera_params[2]],
                           [0, camera_params[1], camera_params[3]],
                           [0, 0, 1]])

    pose_save_dict, point_info_dict = {}, {}
    
    print("Accumulating correspondences and solving PnP...")
    tq = tqdm(total=len(query_image_names))
    for j, query_image_path in enumerate(query_image_names):

        query_img_idx = int(query_image_path.split(".")[0])
        tq.set_description("query_img_idx: {}".format(query_img_idx))

        point_responses, points2d, points3d, preop_image_names, preop_points2d = \
            localization_database_utils.retrieve_image_correspondences(db_path=localization_db_path, 
                                                                       query_image_name=query_image_path)

        ret, point_info_dict, pose_save_dict = \
            adaptative_pose_estimation_PnP(image_name=query_image_path, 
                                           point_responses=point_responses,
                                           points_3d=points3d, points_2d=points2d,
                                           response_threshold_interval=[args.minimum_response_threshold, args.maximum_response_threshold],
                                           threshold_step=args.response_threshold_step, 
                                           point_number_interval=[args.minimum_point_number, args.maximum_point_number],
                                           reprojection_error=args.reprojection_error,
                                           point_info_dict=point_info_dict, pose_save_dict=pose_save_dict,
                                           camera_mtx=camera_mtx)

        if ret is None:
            tq.update(1)
            continue

        # Saving purposes
        if j % 2 == 0 or j == (len(query_image_names) - 1):
            torch.save(pose_save_dict, os.path.join(SAVE_DIR, "raw_query_pose_estimations.pth"))
            torch.save(point_info_dict, os.path.join(SAVE_DIR, "point_number_dict.pth"))

        # Visualization of point correspondences
        if args.extra_visualizations:
            # Keep responses over selected threshold for visualization
            selected_indexes = np.where(point_responses.astype(np.float64) > point_info_dict[query_image_path][2])
            points2d_in = points2d[selected_indexes]

            preop_image_names_in = preop_image_names[selected_indexes]
            preop_points2d_in = preop_points2d[selected_indexes]
            query_image_path_complete = os.path.join(args.query_sequence_root, "images", query_image_path)
            visualization_utils.display_point_feature_mapping(graph_preop_image_names=preop_image_names_in, 
                                                              graph_preop_image_coordinates=preop_points2d_in.astype(np.float32),
                                                              query_image_path=query_image_path_complete, 
                                                              graph_query_image_coordinates=points2d_in.astype(np.float32),
                                                              args=args)

            visualization_utils.draw_correspondences(img_path=query_image_path_complete, 
                                                     img_pts=points2d_in.astype(np.float32),
                                                     base_dir=os.path.join(args.base_dir, args.save_exp_description, "images_keypoints"))
        tq.update(1)
    tq.close()

    print("Pose estimation for {} images in {} seconds ".format(len(pose_save_dict.keys()), round(time.process_time() - start, 3)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Main camera relocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path settings
    parser.add_argument("--sequence_root", type=str, required=True,
                        help="Directory root of the preoperative video sequence")
    parser.add_argument("--query_sequence_root", type=str, required=True,
                        help="Directory root of the intraoperative video sequence")
    parser.add_argument("--exp_description", type=str, required=True,
                        help="Experiment description for file reading purposes")
    parser.add_argument("--save_exp_description", type=str, required=True,
                        help="Experiment description for saving purposes")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Save base directory")

    # Dense descriptor model arguments
    parser.add_argument("--image_downsampling", type=float, default=4.0,
                        help="Input image downsampling rate")
    parser.add_argument("--trained_model_path", type=str, required=True, 
                        help="Path to the trained descriptor model")
    parser.add_argument("--feature_length", type=int, default=128, 
                        help="Length of point descriptors (depends on output channel dimension of network)")
    parser.add_argument("--filter_growth_rate", type=int, default=10, 
                        help="Filter growth rate of network")

    # General relocalization arguments
    parser.add_argument("--mode", type=str, choices=["localize", "pnp", "all"], required=True,
                        help="Process step(s) to perform")
    parser.add_argument("--gpu_id", type=int, default=0, 
                        help="GPU ID for descriptor matching stage")
    parser.add_argument("--query_subsampling", type=int, default=3, required=True,
                        help="Subsampling rate for query sequence in Keypoint Relocalization stage")
    parser.add_argument("--reprojection_error", type=float, default=8.0,
                        help="Maximum allowed distance between observed/computed point projections to consider inliers")
    parser.add_argument("--minimum_response_threshold", type=float, default=0.7, required=True,
                        help="Lower bound in response threshold dynamic method")
    parser.add_argument("--maximum_response_threshold", type=float, default=1.0, required=True,
                        help="Upper bound in response threshold dynamic method")
    parser.add_argument("--response_threshold_step", type=float, default=0.001, required=True,
                        help="Step size in response threshold dynamic method")
    parser.add_argument("--minimum_point_number", type=int, default=10, required=True,
                        help="Minimum number of correspondences")
    parser.add_argument("--maximum_point_number", type=int, default=100, required=True,
                        help="Maximum number of correspondences")
    parser.add_argument('--matching_model_description', type=str, required=True, 
                        help='Description of model providing base for matching')
    parser.add_argument('--feature_model_description', type=str, required=True, 
                        help='Description of model providing features to fill')
    parser.add_argument('--extra_visualizations', action='store_true', 
                        help='Save additional visualizations (not strictly necessary for reloc. process)')
    parser.add_argument("--erosion_iterations", type=int, default=10, 
                        help="Number of erosion iterations to avoid near-boundary matches")
 
    args = parser.parse_args()

    if args.mode == "pnp_eval":
        correct_points_and_solve_PnP(args=args)
    elif args.mode == "localize":
        localize_correspondences(args=args)
    else:
        localize_correspondences(args=args)
        correct_points_and_solve_PnP(args=args)