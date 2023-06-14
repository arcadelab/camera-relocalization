import os
import cv2
import copy
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt

import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
from pytransform3d.transform_manager import TransformManager

import general_utils
import database_utils as db


def graph_points(point_array, color, save_path):
    """
    Exports 3D point set to PLY format.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)
    pcd.paint_uniform_color(color)
    o3d.io.write_point_cloud(save_path, pcd)


def read_and_visualize_tracker_trajectory(args):
    """
    Exports tracker trajectory in PLY format.
    OJO: The query segment's number should correspond to the read csv files.
         The query image numbers are multiplied because of original data subsampling.
    """
    query_matches = pd.read_csv(os.path.join("PolarisandIndexMatches", "P3_1_matches.csv"),
                                names=["img_idx", "polaris_idx"], header=None)
    query_positions = pd.read_csv(os.path.join("PolarisandIndexMatches", "P3_1.csv"),
                                  names=["x", "y", "z"], header=None)

    query_image_names = general_utils.get_all_color_image_names_in_sequence(Path(args.query_sequence_root))
    query_image_names = [str(j) for j in query_image_names]
    query_image_numbers = [int(int(os.path.basename(i).split(".")[0])*4) for i in query_image_names]

    polaris_trajectory = np.zeros((len(query_image_numbers), 3))
    
    for j, query_num in enumerate(query_image_numbers):
        query_match_row = query_matches[query_matches.img_idx == query_num]
        query_position_idx = int(query_match_row.polaris_idx.iloc[0])
        query_position_row = query_positions.iloc[query_position_idx]
        query_position_vector = np.array([query_position_row.x, query_position_row.y, query_position_row.z])

        polaris_trajectory[j,:] = query_position_vector

    graph_points(point_array=polaris_trajectory, color=(1,1,1),
                 save_path=os.path.join(args.base_dir, "polaris_trajectory.ply"))


def visualize_reconstruction(args):
    """
    Visualizes sparse reconstruction from database file.
    OJO: Visualizes 3D points with at least one 2D correspondence.
    """
    dreco_db_path = os.path.join(args.query_sequence_root, "dreco.db")
    rdb = db.ReconstrunctionDB(db_path=dreco_db_path)

    print("Retrieving pre-operative feature and 3D point database...")
    _, _, preop_world_coordinates = rdb.get_all_point3d_features()

    graph_points(point_array=preop_world_coordinates, color=(0, 1, 0),
                 save_path=os.path.join(args.base_dir, "sparse_reconstruction_COLMAP_query.ply"))


def visualize_transformation_between_3D_matches(args, src_points, tgt_points, transformation_params, evaluate_transform=True):
    """
    Visualize original and transformed 3D point clouds (composed of 3D matches in query 2D space).
    OJO: Source > Intra-operative model (query sequence root)
         Target > Pre-operative model (sequence root)
         Transformation type is hard-coded (line 94) - change accordingly!
    """
    transformation_matrix = \
        general_utils.transformation_matrix_from_params(transformation_params=transformation_params, use_scale=True)

    transformed_points3d = np.zeros((src_points.shape))
    trans_error_list = []

    for p, point3d in enumerate(src_points):
        point = np.zeros((4))
        point[0:3] = point3d
        point[-1] = 1.0
        
        transformed = transformation_matrix @ point
        transformed_points3d[p,:] = transformed[0:3].T

        if evaluate_transform:
            trans_error = ((transformed[0:3] - tgt_points[p,:]) ** 2).mean(axis=None)
            trans_error_list.append(trans_error)

    graph_points(point_array=transformed_points3d, color=(0, 1, 0),
                 save_path=os.path.join(args.base_dir, args.save_exp_description, "transformed_src.ply"))

    graph_points(point_array=tgt_points, color=(1, 0, 0),
                 save_path=os.path.join(args.base_dir, args.save_exp_description, "tgt.ply"))
    
    graph_points(point_array=src_points, color=(0, 0, 1),
                 save_path=os.path.join(args.base_dir, args.save_exp_description, "src.ply"))
                 
    return np.mean(trans_error_list), np.std(trans_error_list)



def draw_correspondences(img_path, img_pts, base_dir, color=(0, 255, 0)):
    """
    Draws a set of 2D points over an image.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    img = cv2.imread(img_path)
    image_name = os.path.basename(img_path)
    for i in range(len(img_pts)):
        x1, y1 = img_pts[i, 0], img_pts[i, 1]        
        cv2.circle(img, (int(x1), int(y1)), 1, color, 5)

    cv2.imwrite(os.path.join(base_dir, image_name), img)


def draw_projections_preop(preop_image_names, preop_image_coordinates, args):

    for preop_img, preop_coords in zip(preop_image_names, preop_image_coordinates):
        breakpoint()
        preop_img_path = os.path.join(args.sequence_root, "images", preop_img)
        draw_correspondences(img_path=preop_img_path, img_pts=preop_coords, base_dir=os.path.join(args.storage_dir, "preop_images"))


def draw_pairwise_correspondences(source_img_path, target_img_path, source_keypoint_locations, target_keypoint_locations):
    """
    Plots keypoint locations from DFM process between two images.
    """
    assert len(source_keypoint_locations) == len(target_keypoint_locations), "Source and target keypoint number should be equal"

    src_img = cv2.imread(source_img_path)
    tgt_img = cv2.imread(target_img_path)

    offset = 1
    joint_img = np.zeros(shape=(src_img.shape[0], offset + src_img.shape[1] + tgt_img.shape[1], 3), dtype=np.uint8)
    joint_img[0:src_img.shape[0], 0:src_img.shape[1], :] = src_img
    joint_img[0:src_img.shape[0], (offset + src_img.shape[1]):, :] = tgt_img

    num_draw = 30
    num_corresp = len(source_keypoint_locations)

    source_keypoint_locations = source_keypoint_locations[::num_corresp//num_draw]
    target_keypoint_locations = target_keypoint_locations[::num_corresp//num_draw]

    for i in range(len(source_keypoint_locations)):
        x1, y1 = int(source_keypoint_locations[i, 0]), int(source_keypoint_locations[i, 1])
        x2, y2 = int(target_keypoint_locations[i, 0]), int(target_keypoint_locations[i, 1])
        
        x2 += offset + src_img.shape[1]

        cv2.line(joint_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(joint_img, (x1, y1), 1, (0, 0, 255), 5)
        cv2.circle(joint_img, (x2, y2), 1, (0, 0, 255), 5)

    SAVE_PATH = os.path.join("./test_inference", "correspondence_check")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    src_idx = int(os.path.basename(source_img_path).split(".")[0])
    tgt_idx = int(os.path.basename(target_img_path).split(".")[0])
    image_name = os.path.join(SAVE_PATH, "keypoint_locations_{}_{}.jpg".format(src_idx, tgt_idx))

    cv2.imwrite(image_name, joint_img)


def visualize_image_3D_points(args):
    """
    Overlay images with 2D points having 3D correspondences.
    """
    query_image_names = general_utils.get_all_color_image_names_in_sequence(Path(args.query_sequence_root))
    query_image_names = [str(j) for j in query_image_names]
    pose_save_dict = torch.load(os.path.join(args.base_dir, args.exp_description, "computed_points_dict.pth"))

    indexes = [53, 131, 59, 78, 120, 121]

    for query_image_path in query_image_names:
        if query_image_path not in pose_save_dict.keys():
            continue
        
        points2d_in = pose_save_dict[query_image_path]["computed_points2d"]
        points3d_in = pose_save_dict[query_image_path]["computed_points3d"]
        img_idx = int(os.path.basename(query_image_path).split(".")[0])

        if img_idx in indexes:
            graph_points(point_array=points3d_in, color=(0, 0, 1),
                         save_path=os.path.join(args.base_dir, args.exp_description, "3D_points_{}.ply".format(img_idx)))

            draw_correspondences(img_path=query_image_path, img_pts=points2d_in,
                                 base_dir=args.base_dir)


def plot_trajectories_o3d(pose_dict, axlen, subsampling_factor, camera_mtx, point_cloud=None):
    """
    Visualize camera trajectory saved in pose_dict using Open3D.
    """
    tm = []
    transformation_matrices = np.empty((len(pose_dict.values()), 4, 4))
    sensor_size = (float(1920), float(1080))

    for j, cam_pose in enumerate(pose_dict.values()):
        if j % subsampling_factor != 0:
            continue
        tm.append(pt.transform_from(R=cam_pose[0], p=cam_pose[1]))   
    transformation_matrices = np.asarray(tm)

    fig = pv.figure()
    if point_cloud is not None:
        pc_color = np.zeros((len(point_cloud), 3)) 
        pc_color[:, 0], pc_color[:, 1], pc_color[:, 2] = 0.5, 0.5, 0.5
        fig.scatter(point_cloud, s=0.04, c=pc_color)

    for pose in transformation_matrices:
        fig.plot_transform(A2B=pose, s=axlen)
        fig.plot_camera(M=camera_mtx, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)

    fig.show()


def plot_trajectories_o3d_plt(pose_dict, axlen, subsampling_factor):
    """
    Visualize camera trajectory saved in pose_dict using matplotlib.
    """
    tm = TransformManager()
    eval_cases = ["00000590.jpg", "00000605.jpg", "00000575.jpg"]

    for j, (query_im, cam_pose) in enumerate(pose_dict.items()):
        if j % subsampling_factor != 0:
            continue
        pose = pt.transform_from(R=cam_pose[0], p=cam_pose[1])
        tm.add_transform(os.path.basename(query_im), "robot", pose)

    ax = tm.plot_frames_in("robot", s=axlen)
    ax.set_xlim((-1.0, 1.0))
    ax.set_ylim((-1.0, 1.0))
    ax.set_zlim((-1.0, 1.0))
    plt.show()


def display_point_feature_mapping(graph_preop_image_names, graph_preop_image_coordinates,
                                  query_image_path, graph_query_image_coordinates, point_responses, args):
    """
    Draws a set of corresponding 2D points between two images (base and query).
    """
    unique_preop_image_list = np.unique(graph_preop_image_names)

    plot_save_path = os.path.join(args.storage_dir, "complete_mapping")
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    for preop_image_path in unique_preop_image_list:

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.grid(False)
        ax.axis("off")
        ax.set_title("Pre-operative image ({})".format(preop_image_path))
        
        preop_image = plt.imread(os.path.join(args.sequence_root, "images", preop_image_path))
        preop_coords = graph_preop_image_coordinates[graph_preop_image_names == preop_image_path]
        responses = point_responses[graph_preop_image_names == preop_image_path]
        
        point_num = len(preop_coords)
        times = np.fromiter(range(point_num), dtype=float)
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(times))[np.newaxis, :, :3]

        for i in range(point_num):
            x, y = preop_coords[i, 0], preop_coords[i, 1]
            u_col = colors[0][i]*255      
            cv2.circle(preop_image, (int(x), int(y)), 8, (int(u_col[0]), int(u_col[1]), int(u_col[2])), 5)

        ax.imshow(preop_image)

        ax1 = fig.add_subplot(1, 2, 2)
        ax1.grid(False)
        ax1.axis("off")
        ax1.set_title("Query image ({})".format(os.path.basename(query_image_path)))

        query_image = plt.imread(query_image_path)
        query_coords = graph_query_image_coordinates[graph_preop_image_names == preop_image_path]

        for i in range(point_num):
            x1, y1 = query_coords[i, 0], query_coords[i, 1]       
            u_col = colors[0][i]*255
            cv2.circle(query_image, (int(x1), int(y1)), 8, (int(u_col[0]), int(u_col[1]), int(u_col[2])), 5)
        
        ax1.imshow(query_image)
        query_num = int(os.path.basename(query_image_path).split(".")[0])
        preop_num = int(preop_image_path.split(".")[0])

        fig.tight_layout()
        fig.suptitle("{}".format(responses))
        fig.savefig(os.path.join(plot_save_path, "p-{}-q-{}.jpg".format(preop_num, query_num)))


def display_point_feature_mapping_only_query(graph_preop_image_names, graph_preop_image_coordinates,
                                  query_image_path, graph_query_image_coordinates, point_responses, args):
    """
    Draws a set of corresponding 2D points between two images (base and query).
    """
    unique_preop_image_list = np.unique(graph_preop_image_names)

    plot_save_path = os.path.join(args.storage_dir, "query_mapping")
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    for preop_image_path in unique_preop_image_list:

        preop_coords = graph_preop_image_coordinates[graph_preop_image_names == preop_image_path]
        responses = point_responses[graph_preop_image_names == preop_image_path]
        
        point_num = len(preop_coords)
        times = np.fromiter(range(point_num), dtype=float)
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(times))[np.newaxis, :, :3]

        query_image = plt.imread(query_image_path)
        query_coords = graph_query_image_coordinates[graph_preop_image_names == preop_image_path]

        for i in range(point_num):
            x1, y1 = query_coords[i, 0], query_coords[i, 1]       
            u_col = colors[0][i]*255
            cv2.circle(query_image, (int(x1), int(y1)), 8, (int(u_col[0]), int(u_col[1]), int(u_col[2])), 5)
        
        query_num = int(os.path.basename(query_image_path).split(".")[0])
        preop_num = int(preop_image_path.split(".")[0])

        plt.imsave(os.path.join(plot_save_path, "p-{}-q-{}-r-{}.jpg".format(preop_num, query_num, responses)), query_image)


def graph_point_correspondences(args, src_points2d, tgt_points2d, image_pair_paths, eval_threshold, shared):
    """
    Draws a set of corresponding 2D points between two images (base and query).
    Similar to display_point_feature_mapping function, but changes the logic for formal evaluation.
    """
    plot_save_path = os.path.join("./localization_evaluation", args.exp_description, "point_correspondences")
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.grid(False)
    ax.axis("off")
    ax.set_title("Source image ({})".format(os.path.basename(image_pair_paths[0])))
    
    src_image = plt.imread(image_pair_paths[0])
    
    point_num = len(src_points2d)
    times = np.fromiter(range(point_num), dtype=float)
    norm = plt.Normalize()
    colors = plt.cm.hsv(norm(times))[np.newaxis, :, :3]

    for i in range(point_num):
        x, y = src_points2d[i, 0], src_points2d[i, 1]
        u_col = colors[0][i]*255      
        cv2.circle(src_image, (int(x), int(y)), 8, (int(u_col[0]), int(u_col[1]), int(u_col[2])), 8)

    ax.imshow(src_image)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.grid(False)
    ax1.axis("off")
    ax1.set_title("Target image ({})".format(os.path.basename(image_pair_paths[1])))

    tgt_image = plt.imread(image_pair_paths[1])

    for i in range(point_num):
        x1, y1 = tgt_points2d[i, 0], tgt_points2d[i, 1]       
        u_col = colors[0][i]*255
        cv2.circle(tgt_image, (int(x1), int(y1)), 8, (int(u_col[0]), int(u_col[1]), int(u_col[2])), 8)
    
    ax1.imshow(tgt_image)

    fig.tight_layout()
    title = "shared" if shared else "non-shared"
    src_num = int(os.path.basename(image_pair_paths[0]).split(".")[0])
    tgt_num = int(os.path.basename(image_pair_paths[1]).split(".")[0])
    fig.savefig(os.path.join(plot_save_path, "{}-src-{}-tgt-{}-thr-{}.jpg".format(title, src_num, tgt_num, np.round(eval_threshold,2))))


def graph_metrics_vs_points(args, total_point_num, inlier_num, eval_metrics):
    """
    Simple graph plotting number of point correspondences and evaluation metrics.
    """
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(inlier_num, eval_metrics, color="lightsalmon", marker="o")
    plt.xlabel("No. of inliers")
    plt.ylabel("Metric")

    plt.subplot(1, 3, 2)
    plt.plot(total_point_num, eval_metrics, color="teal", marker="o")
    plt.xlabel("Total no. of point correspondences")
    plt.ylabel("Metric")

    inlier_ratio_list = [i / j for i, j in zip(inlier_num, total_point_num)]

    plt.subplot(1, 3, 3)
    plt.plot(inlier_ratio_list, eval_metrics, color="plum", marker="o")
    plt.xlabel("Inlier ratio")
    plt.ylabel("Metric")

    plt.tight_layout()
    plt.savefig(os.path.join(args.base_dir, args.save_exp_description, "point_metrics_graph.png"))


def graph_metrics_vs_images(args, query_image_names, trans_metrics, rot_metrics):
    """
    Simple graph plotting query image indexes and evaluation metrics.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(query_image_names, trans_metrics, color="lightsalmon", marker="o")
    plt.axhline(y = np.mean(trans_metrics), color = 'b', linestyle = 'dashed')   
    plt.xlabel("Query image indexes")
    plt.ylabel("Translation error")

    plt.subplot(1, 2, 2)
    plt.plot(query_image_names, rot_metrics, color="darkred", marker="o")
    plt.axhline(y = np.mean(rot_metrics), color = 'b', linestyle = 'dashed') 
    plt.xlabel("Query image indexes")
    plt.ylabel("Orientation error")

    plt.tight_layout()
    plt.savefig(os.path.join(args.base_dir, args.save_exp_description, "image_metrics_graph.png"))


def plot_and_save_trajectory(pose_dict, axlen, save_name, subsampling_factor=1, draw_connections=True, connection_color=[1, 0, 0], show=False):
    """
    Plots trajectory and saves it in PLY format.
    """
    ignore_idxs = [410, 1330, 1430, 1530]
    # ignore_idxs = []
    tm = []  # Temporal transformations
    transformation_matrices = np.empty((len(pose_dict), 4, 4))
    points = []
    for j, (img_name,cam_pose) in enumerate(pose_dict.items()):
        if j % subsampling_factor != 0:
            continue
        if np.all(np.isfinite(cam_pose[1])) == False or int(img_name.split(".")[0]) in ignore_idxs:
            continue
        points.append(cam_pose[1])
        tm.append(pt.transform_from(R=cam_pose[0], p=cam_pose[1]))

    transformation_matrices = np.asarray(tm)
    trajectory = None
    for i, pose in enumerate(transformation_matrices):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axlen)
        mesh_frame.transform(pose)
        if i == 0:
            trajectory = copy.deepcopy(mesh_frame)
        else:
            trajectory += copy.deepcopy(mesh_frame)

    if draw_connections:
        lines = []
        for i in range(len(points) - 1):
            lines.append([i, i+1])
        colors = [connection_color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

    if show:
        o3d.visualization.draw_geometries([trajectory, line_set])

    o3d.io.write_triangle_mesh(save_name, trajectory)


def plot_camera_at_index(pose_dict, axlen, index, connection_color=[1, 0, 0]):
    """
    Plots a single camera (own coord. system) within linear trajectory.
    Only visualization.
    """
    tm = []  # Temporal transformations
    transformation_matrices = np.empty((len(pose_dict.values()), 4, 4))
    points = []
    for j, cam_pose in enumerate(pose_dict):
        points.append(cam_pose[0:3, 3])
        tm.append(pt.transform_from(R=cam_pose[0:3, 0:3], p=cam_pose[0:3, 3]))

    transformation_matrices = np.asarray(tm)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pc_colors = [[0, 0, 0] for i in range(len(points))]
    pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axlen)
    mesh_frame.transform(transformation_matrices[index])
    lines = []
    for i in range(len(points) - 1):
        lines.append([i, i+1])
    colors = [connection_color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([mesh_frame, line_set, pcd])


def visualize_2d_point_differences(args, query_image_name, colmap_points2d, reprojected_points2d, localized_points2d):

    plot_save_path = os.path.join(args.base_dir, args.save_exp_description, "point_differences_2")
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    query_image = plt.imread(os.path.join(args.query_sequence_root, "images", query_image_name))

    for i in range(colmap_points2d.shape[0]):
        x, y = colmap_points2d[i, 0], colmap_points2d[i, 1]    
        cv2.circle(query_image, (int(x), int(y)), 8, (0,255,0), 8)
    
    for i in range(localized_points2d.shape[0]):
        x, y = localized_points2d[i, 0], localized_points2d[i, 1]    
        cv2.circle(query_image, (int(x), int(y)), 8, (255,0,0), 8)

    plt.imshow(query_image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_path, query_image_name))
