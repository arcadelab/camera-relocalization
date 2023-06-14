'''
Author: Xingtong Liu, Maia Stiber, Jindan Huang, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''
import cv2
import torch
import pickle
import numpy as np
from multiprocessing import Process, Queue

import albumentations as albu
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor

import utils

"""
Modified version of DescriptorDataset to include negative matches samples (see lines 809-883).
"""

def pre_processing_data(process_id, folder_list, image_downsampling, network_downsampling, inlier_percentage,
                        visible_interval, suggested_h, suggested_w,
                        queue_clean_point_list, queue_intrinsic_matrix, queue_point_cloud,
                        queue_mask_boundary, queue_view_indexes_per_point, queue_selected_indexes,
                        queue_visible_view_indexes, queue_extrinsics, queue_projection, 
                        queue_crop_positions, queue_estimated_scale, queue_shared_point_views=None):
                        
    for folder in folder_list:
        # For now, we only use the results in the subfolder named "0" produced by COLMAP
        colmap_result_folder = folder / "colmap" / "0"
        images_folder = folder / "images"

        # We use folder path as the key for dictionaries
        folder_str = str(folder)

        # Read visible view indexes
        visible_view_indexes = utils.read_visible_view_indexes(colmap_result_folder)
        if len(visible_view_indexes) == 0:
            print("Sequence {} does not have relevant files".format(folder_str))
            continue
        queue_visible_view_indexes.put([folder_str, visible_view_indexes])

        # Read undistorted mask image
        undistorted_mask_boundary = cv2.imread(str(colmap_result_folder / "undistorted_mask.bmp"), cv2.IMREAD_GRAYSCALE)

        # Downsample and crop the undistorted mask image
        cropped_downsampled_undistorted_mask_boundary, start_h, end_h, start_w, end_w = \
            utils.downsample_and_crop_mask(undistorted_mask_boundary, downsampling_factor=image_downsampling,
                                           divide=network_downsampling, suggested_h=suggested_h,
                                           suggested_w=suggested_w)

        queue_mask_boundary.put([folder_str, cropped_downsampled_undistorted_mask_boundary])
        queue_crop_positions.put([folder_str, [start_h, end_h, start_w, end_w]])

        # Read selected image indexes
        selected_indexes = utils.read_selected_indexes(colmap_result_folder)
        queue_selected_indexes.put([folder_str, selected_indexes])

        # Read undistorted camera intrinsics
        undistorted_camera_intrinsic_per_view = utils.read_camera_intrinsic_per_view(colmap_result_folder)
        
        # Downsample and crop the undistorted camera intrinsics
        # Assuming that camera intrinsics within one video sequence remains the same
        cropped_downsampled_undistorted_intrinsic_matrix = utils.modify_camera_intrinsic_matrix(
            undistorted_camera_intrinsic_per_view[0], start_h=start_h,
            start_w=start_w, downsampling_factor=image_downsampling)
        queue_intrinsic_matrix.put([folder_str, cropped_downsampled_undistorted_intrinsic_matrix])

        # Read sparse point cloud from SfM
        point_cloud = utils.read_point_cloud(str(colmap_result_folder / "structure.ply"))
        queue_point_cloud.put([folder_str, point_cloud])

        # Read visible view indexes per point
        view_indexes_per_point = \
            utils.read_view_indexes_per_point(colmap_result_folder, visible_view_indexes=visible_view_indexes, 
                                              point_cloud_count=len(point_cloud))

        # Generate matrix with information about shared 3D points between views
        if queue_shared_point_views is not None:
            shared_point_views_matrix = \
                utils.generate_shared_point_views_matrix(view_indexes_per_point=view_indexes_per_point, 
                                                         visible_view_indexes=visible_view_indexes)
            queue_shared_point_views.put([folder_str, shared_point_views_matrix])

        # Update view_indexes_per_point with neighborhood frames to increase point correspondences and
        # avoid as much occlusion problem as possible
        view_indexes_per_point = utils.overlapping_visible_view_indexes_per_point(view_indexes_per_point,
                                                                                  visible_interval)
        queue_view_indexes_per_point.put([folder_str, view_indexes_per_point])

        # Read pose data for all visible views
        poses = utils.read_pose_data(colmap_result_folder)

        # Calculate extrinsic and projection matrices
        visible_extrinsic_matrices, visible_cropped_downsampled_undistorted_projection_matrices = \
            utils.get_extrinsic_matrix_and_projection_matrix(poses,
                                                             intrinsic_matrix=
                                                             cropped_downsampled_undistorted_intrinsic_matrix,
                                                             visible_view_count=len(visible_view_indexes))
        queue_extrinsics.put([folder_str, visible_extrinsic_matrices])
        queue_projection.put([folder_str, visible_cropped_downsampled_undistorted_projection_matrices])

        # Get approximate data global scale to reduce training data imbalance
        global_scale = utils.global_scale_estimation(visible_extrinsic_matrices, point_cloud)
        queue_estimated_scale.put([folder_str, global_scale])
        visible_cropped_downsampled_imgs = utils.get_color_imgs(images_folder,
                                                                visible_view_indexes=visible_view_indexes,
                                                                start_h=start_h, start_w=start_w,
                                                                end_h=end_h, end_w=end_w,
                                                                downsampling_factor=image_downsampling)
        # Calculate contaminated point list
        clean_point_indicator_array = utils.get_clean_point_list(imgs=visible_cropped_downsampled_imgs,
                                                                 point_cloud=point_cloud,
                                                                 mask_boundary=
                                                                 cropped_downsampled_undistorted_mask_boundary,
                                                                 inlier_percentage=inlier_percentage,
                                                                 projection_matrices=
                                                                 visible_cropped_downsampled_undistorted_projection_matrices,
                                                                 extrinsic_matrices=visible_extrinsic_matrices,
                                                                 view_indexes_per_point=view_indexes_per_point)
        queue_clean_point_list.put([folder_str, clean_point_indicator_array])
        print("Sequence {} finished by process {}".format(folder_str, process_id))

    print("Process {} finished".format(process_id))


def find_common_valid_size(folder_list, image_downsampling, network_downsampling, queue_size):
    for folder in folder_list:
        # Read mask image
        undistorted_mask_boundary = cv2.imread(str(folder / "undistorted_mask.bmp"), cv2.IMREAD_GRAYSCALE)
        # Downsample and crop the undistorted mask image
        _, start_h, end_h, start_w, end_w = \
            utils.downsample_and_crop_mask(undistorted_mask_boundary, downsampling_factor=image_downsampling,
                                           divide=network_downsampling)
        queue_size.put([end_h - start_h, end_w - start_w])


class DepthDataset(Dataset):
    def __init__(self, image_file_names, folder_list, adjacent_range,
                 image_downsampling, network_downsampling, inlier_percentage, visible_interval,
                 load_intermediate_data, intermediate_data_root, num_pre_workers, num_iter=None, phase="Train"):
        self.image_file_names = image_file_names
        self.folder_list = folder_list
        self.adjacent_range = adjacent_range
        self.inlier_percentage = inlier_percentage
        self.image_downsampling = image_downsampling
        self.network_downsampling = network_downsampling
        self.visible_interval = visible_interval
        self.num_pre_workers = min(len(folder_list), num_pre_workers)
        self.num_iter = num_iter
        self.num_sample = len(self.image_file_names)
        self.phase = phase

        self.clean_point_list_per_seq = {}
        self.intrinsic_matrix_per_seq = {}
        self.point_cloud_per_seq = {}
        self.mask_boundary_per_seq = {}
        self.view_indexes_per_point_per_seq = {}
        self.selected_indexes_per_seq = {}
        self.visible_view_indexes_per_seq = {}
        self.extrinsics_per_seq = {}
        self.projection_per_seq = {}
        self.crop_positions_per_seq = {}
        self.estimated_scale_per_seq = {}
        self.normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)

        precompute_path = intermediate_data_root / ("precompute_" + str(
            self.image_downsampling) + "_" + str(self.network_downsampling) + ".pkl")

        # Save all intermediate results to hard disk for quick access later on
        if not load_intermediate_data or not precompute_path.exists():
            queue_size = Queue()
            queue_clean_point_list = Queue()
            queue_intrinsic_matrix = Queue()
            queue_point_cloud = Queue()
            queue_mask_boundary = Queue()
            queue_view_indexes_per_point = Queue()
            queue_selected_indexes = Queue()
            queue_visible_view_indexes = Queue()
            queue_extrinsics = Queue()
            queue_projection = Queue()
            queue_crop_positions = Queue()
            queue_estimated_scale = Queue()

            process_pool = []

            interval = len(self.folder_list) / self.num_pre_workers

            # Go through the entire image list to find the largest required h and w
            for i in range(self.num_pre_workers):
                process_pool.append(Process(target=find_common_valid_size, args=(
                    self.folder_list[
                    int(np.round(i * interval)): min(int(np.round((i + 1) * interval)), len(self.folder_list))],
                    self.image_downsampling,
                    self.network_downsampling,
                    queue_size)))

            for t in process_pool:
                t.start()

            largest_h = 0
            largest_w = 0

            for t in process_pool:
                while t.is_alive():
                    while not queue_size.empty():
                        h, w = queue_size.get()
                        if h > largest_h:
                            largest_h = h
                        if w > largest_w:
                            largest_w = w
                    t.join(timeout=1)

            while not queue_size.empty():
                h, w = queue_size.get()
                if h > largest_h:
                    largest_h = h
                if w > largest_w:
                    largest_w = w

            if largest_h == 0 or largest_w == 0:
                print("image size calculation failed.")
                raise IOError

            print("Largest image size is: ", largest_h, largest_w)
            print("Start pre-processing dataset...")
            process_pool = []
            for i in range(self.num_pre_workers):
                process_pool.append(Process(target=pre_processing_data,
                                            args=(i, self.folder_list[
                                                     int(np.round(i * interval)): min(int(np.round((i + 1) * interval)),
                                                                                      len(self.folder_list))],
                                                  self.image_downsampling, self.network_downsampling,
                                                  self.inlier_percentage, self.visible_interval, largest_h, largest_w,
                                                  queue_clean_point_list,
                                                  queue_intrinsic_matrix, queue_point_cloud,
                                                  queue_mask_boundary, queue_view_indexes_per_point,
                                                  queue_selected_indexes,
                                                  queue_visible_view_indexes,
                                                  queue_extrinsics, queue_projection,
                                                  queue_crop_positions,
                                                  queue_estimated_scale)))

            for t in process_pool:
                t.start()

            count = 0
            for t in process_pool:
                print("Waiting for process {} to complete".format(count))
                count += 1
                while t.is_alive():
                    while not queue_selected_indexes.empty():
                        folder, selected_indexes = queue_selected_indexes.get()
                        self.selected_indexes_per_seq[folder] = selected_indexes
                    while not queue_visible_view_indexes.empty():
                        folder, visible_view_indexes = queue_visible_view_indexes.get()
                        self.visible_view_indexes_per_seq[folder] = visible_view_indexes
                    while not queue_view_indexes_per_point.empty():
                        folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                        self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
                    while not queue_clean_point_list.empty():
                        folder, clean_point_list = queue_clean_point_list.get()
                        self.clean_point_list_per_seq[folder] = clean_point_list
                    while not queue_intrinsic_matrix.empty():
                        folder, intrinsic_matrix = queue_intrinsic_matrix.get()
                        self.intrinsic_matrix_per_seq[folder] = intrinsic_matrix
                    while not queue_extrinsics.empty():
                        folder, extrinsics = queue_extrinsics.get()
                        self.extrinsics_per_seq[folder] = extrinsics
                    while not queue_projection.empty():
                        folder, projection = queue_projection.get()
                        self.projection_per_seq[folder] = projection
                    while not queue_crop_positions.empty():
                        folder, crop_positions = queue_crop_positions.get()
                        self.crop_positions_per_seq[folder] = crop_positions
                    while not queue_point_cloud.empty():
                        folder, point_cloud = queue_point_cloud.get()
                        self.point_cloud_per_seq[folder] = point_cloud
                    while not queue_mask_boundary.empty():
                        folder, mask_boundary = queue_mask_boundary.get()
                        self.mask_boundary_per_seq[folder] = mask_boundary
                    while not queue_estimated_scale.empty():
                        folder, estiamted_scale = queue_estimated_scale.get()
                        self.estimated_scale_per_seq[folder] = estiamted_scale
                    t.join(timeout=1)

            while not queue_selected_indexes.empty():
                folder, selected_indexes = queue_selected_indexes.get()
                self.selected_indexes_per_seq[folder] = selected_indexes
            while not queue_visible_view_indexes.empty():
                folder, visible_view_indexes = queue_visible_view_indexes.get()
                self.visible_view_indexes_per_seq[folder] = visible_view_indexes
            while not queue_view_indexes_per_point.empty():
                folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
            while not queue_clean_point_list.empty():
                folder, clean_point_list = queue_clean_point_list.get()
                self.clean_point_list_per_seq[folder] = clean_point_list
            while not queue_intrinsic_matrix.empty():
                folder, intrinsic_matrix = queue_intrinsic_matrix.get()
                self.intrinsic_matrix_per_seq[folder] = intrinsic_matrix
            while not queue_extrinsics.empty():
                folder, extrinsics = queue_extrinsics.get()
                self.extrinsics_per_seq[folder] = extrinsics
            while not queue_projection.empty():
                folder, projection = queue_projection.get()
                self.projection_per_seq[folder] = projection
            while not queue_crop_positions.empty():
                folder, crop_positions = queue_crop_positions.get()
                self.crop_positions_per_seq[folder] = crop_positions
            while not queue_point_cloud.empty():
                folder, point_cloud = queue_point_cloud.get()
                self.point_cloud_per_seq[folder] = point_cloud
            while not queue_mask_boundary.empty():
                folder, mask_boundary = queue_mask_boundary.get()
                self.mask_boundary_per_seq[folder] = mask_boundary
            while not queue_estimated_scale.empty():
                folder, estimated_scale = queue_estimated_scale.get()
                self.estimated_scale_per_seq[folder] = estimated_scale
            print("Pre-processing complete.")

            # Store all intermediate information to a single data file
            with open(str(precompute_path), "wb") as f:
                pickle.dump(
                    [self.crop_positions_per_seq, self.selected_indexes_per_seq,
                     self.visible_view_indexes_per_seq,
                     self.point_cloud_per_seq, self.intrinsic_matrix_per_seq,
                     self.mask_boundary_per_seq, self.view_indexes_per_point_per_seq, self.extrinsics_per_seq,
                     self.projection_per_seq, self.clean_point_list_per_seq,
                     self.image_downsampling, self.network_downsampling, self.inlier_percentage,
                     self.estimated_scale_per_seq],
                    f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(str(precompute_path), "rb") as f:
                [self.crop_positions_per_seq, self.selected_indexes_per_seq,
                 self.visible_view_indexes_per_seq,
                 self.point_cloud_per_seq, self.intrinsic_matrix_per_seq,
                 self.mask_boundary_per_seq, self.view_indexes_per_point_per_seq, self.extrinsics_per_seq,
                 self.projection_per_seq, self.clean_point_list_per_seq,
                 self.image_downsampling, self.network_downsampling,
                 self.inlier_percentage, self.estimated_scale_per_seq] = pickle.load(f)

    def __len__(self):
        if self.num_iter is None:
            return len(self.image_file_names)
        else:
            return self.num_iter

    def __getitem__(self, idx):
        if self.phase == "Train" or self.phase == "Validation":
            while True:
                img_file_name = self.image_file_names[idx % len(self.image_file_names)]
                # Retrieve the folder path
                folder = img_file_name.parents[1]
                images_folder = folder / "images"
                folder_str = str(folder)
                # Randomly pick one adjacent frame
                # We assume the filename has 8 logits followed by ".jpg"
                if folder_str not in self.crop_positions_per_seq:
                    print("{} not in pre-compute data".format(folder_str))
                    idx = np.random.randint(0, len(self.image_file_names))
                    continue

                # Randomly pick one adjacent frame
                # We assume the filename of color image has 8 logits with ".jpg" as suffix
                start_h, end_h, start_w, end_w = self.crop_positions_per_seq[folder_str]
                pos, increment = utils.generating_pos_and_increment(idx=idx,
                                                                    visible_view_indexes=
                                                                    self.visible_view_indexes_per_seq[folder_str],
                                                                    adjacent_range=self.adjacent_range)
                                                                    
                img_file_name = self.visible_view_indexes_per_seq[folder_str][
                    idx % len(self.visible_view_indexes_per_seq[folder_str])]

                # Get pair visible view indexes and pair extrinsic and projection matrices
                pair_indexes = [self.visible_view_indexes_per_seq[folder_str][pos],
                                self.visible_view_indexes_per_seq[folder_str][pos + increment]]
                pair_extrinsic_matrices = [self.extrinsics_per_seq[folder_str][pos],
                                           self.extrinsics_per_seq[folder_str][pos + increment]]
                pair_projection_matrices = [self.projection_per_seq[folder_str][pos],
                                            self.projection_per_seq[folder_str][pos + increment]]

                pair_mask_imgs, pair_sparse_depth_imgs, pair_flow_mask_imgs, pair_flow_imgs = \
                    utils.get_torch_training_data(pair_extrinsics=pair_extrinsic_matrices,
                                                  pair_projections=
                                                  pair_projection_matrices, pair_indexes=pair_indexes,
                                                  point_cloud=self.point_cloud_per_seq[folder_str],
                                                  mask_boundary=self.mask_boundary_per_seq[folder_str],
                                                  view_indexes_per_point=self.view_indexes_per_point_per_seq[
                                                      folder_str],
                                                  visible_view_indexes=self.visible_view_indexes_per_seq[folder_str],
                                                  clean_point_list=self.clean_point_list_per_seq[
                                                      folder_str])

                if np.sum(pair_mask_imgs[0]) != 0 and np.sum(pair_mask_imgs[1]) != 0:
                    break
                else:
                    idx = np.random.randint(0, len(self.image_file_names))

            # Read pair images with downsampling and cropping
            pair_imgs = utils.get_pair_color_imgs(prefix_seq=images_folder, pair_indexes=pair_indexes, start_h=start_h,
                                                  start_w=start_w,
                                                  end_h=end_h, end_w=end_w, downsampling_factor=self.image_downsampling)

            # Calculate relative motion between two frames
            relative_motion = np.matmul(pair_extrinsic_matrices[0], np.linalg.inv(pair_extrinsic_matrices[1]))
            rotation_1_wrt_2 = np.reshape(relative_motion[:3, :3], (3, 3)).astype(np.float32)
            translation_1_wrt_2 = (
                    np.reshape(relative_motion[:3, 3], (3, 1)) / self.estimated_scale_per_seq[folder_str]).astype(
                    np.float32)

            # Scale the sparse depth map
            pair_sparse_depth_imgs[0] /= self.estimated_scale_per_seq[folder_str]
            pair_sparse_depth_imgs[1] /= self.estimated_scale_per_seq[folder_str]

            # Format training data
            color_img_1 = pair_imgs[0]
            color_img_2 = pair_imgs[1]

            rotation_2_wrt_1 = np.transpose(rotation_1_wrt_2).astype(np.float32)
            translation_2_wrt_1 = np.matmul(-np.transpose(rotation_1_wrt_2), translation_1_wrt_2).astype(np.float32)

            rotation_1_wrt_2 = rotation_1_wrt_2.reshape((3, 3))
            rotation_2_wrt_1 = rotation_2_wrt_1.reshape((3, 3))
            translation_1_wrt_2 = translation_1_wrt_2.reshape((3, 1))
            translation_2_wrt_1 = translation_2_wrt_1.reshape((3, 1))

            sparse_depth_img_1 = pair_sparse_depth_imgs[0].astype(np.float32)
            sparse_depth_img_2 = pair_sparse_depth_imgs[1].astype(np.float32)
            mask_img_1 = pair_mask_imgs[0].astype(np.float32)
            mask_img_2 = pair_mask_imgs[1].astype(np.float32)
            sparse_depth_img_1 = sparse_depth_img_1.reshape((sparse_depth_img_1.shape[0],
                                                             sparse_depth_img_1.shape[1], 1))
            sparse_depth_img_2 = sparse_depth_img_2.reshape((sparse_depth_img_2.shape[0],
                                                             sparse_depth_img_2.shape[1], 1))
            mask_img_1 = mask_img_1.reshape(
                (mask_img_1.shape[0], mask_img_1.shape[1], 1))
            mask_img_2 = mask_img_2.reshape(
                (mask_img_2.shape[0], mask_img_2.shape[1], 1))
            flow_mask_img_1 = pair_flow_mask_imgs[0].astype(np.float32)
            flow_mask_img_2 = pair_flow_mask_imgs[1].astype(np.float32)
            flow_img_1 = pair_flow_imgs[0].astype(np.float32)
            flow_img_2 = pair_flow_imgs[1].astype(np.float32)

            intrinsic_matrix = self.intrinsic_matrix_per_seq[folder_str][:3, :3]
            intrinsic_matrix = intrinsic_matrix.astype(np.float32)
            intrinsic_matrix = intrinsic_matrix.reshape((3, 3))

            mask_boundary = self.mask_boundary_per_seq[folder_str].astype(np.float32) / 255.0
            mask_boundary[mask_boundary > 0.9] = 1.0
            mask_boundary[mask_boundary <= 0.9] = 0.0
            mask_boundary = mask_boundary.reshape((mask_boundary.shape[0], mask_boundary.shape[1], 1))

            kernel = np.ones((10, 10), np.uint8)
            shrink_boundary = cv2.erode(mask_boundary.astype(np.uint8), kernel, iterations=3)
            shrink_boundary = shrink_boundary.astype(np.float32).reshape(
                (mask_boundary.shape[0], mask_boundary.shape[1], 1))

            # Normalize
            color_img_1 = self.normalize(image=color_img_1)['image']
            color_img_2 = self.normalize(image=color_img_2)['image']

            return [img_to_tensor(color_img_1), img_to_tensor(color_img_2),
                    img_to_tensor(sparse_depth_img_1), img_to_tensor(sparse_depth_img_2),
                    img_to_tensor(mask_img_1), img_to_tensor(mask_img_2),
                    img_to_tensor(flow_img_1), img_to_tensor(flow_img_2),
                    img_to_tensor(flow_mask_img_1), img_to_tensor(flow_mask_img_2),
                    img_to_tensor(mask_boundary), img_to_tensor(shrink_boundary),
                    torch.from_numpy(rotation_1_wrt_2),
                    torch.from_numpy(rotation_2_wrt_1), torch.from_numpy(translation_1_wrt_2),
                    torch.from_numpy(translation_2_wrt_1), torch.from_numpy(intrinsic_matrix),
                    folder_str, img_file_name]

        elif self.phase == "Loading":
            img_file_name = self.image_file_names[idx]
            # Retrieve the folder path
            img_index = int(img_file_name.name[-12:-4])
            folder = img_file_name.parents[1]
            images_folder = folder / "images"
            folder_str = str(folder)

            # Naive solution to errors in data structure from args
            if "Pytorch" in folder_str:
                folder_str = folder_str.split("Pytorch/")[-1]

            start_h, end_h, start_w, end_w = self.crop_positions_per_seq[folder_str]
            pos = self.visible_view_indexes_per_seq[folder_str].index(img_index)

            if pos < len(self.visible_view_indexes_per_seq[folder_str]) - 1:
                increment = 1
            else:
                increment = -1

            # Get pair visible view indexes and pair extrinsic and projection matrices
            pair_indexes = [self.visible_view_indexes_per_seq[folder_str][pos],
                            self.visible_view_indexes_per_seq[folder_str][pos + increment]]
            pair_extrinsic_matrices = [self.extrinsics_per_seq[folder_str][pos],
                                       self.extrinsics_per_seq[folder_str][pos + increment]]
            pair_projection_matrices = [self.projection_per_seq[folder_str][pos],
                                        self.projection_per_seq[folder_str][pos + increment]]
            # Read pair images with downsampling and cropping
            pair_imgs = utils.get_pair_color_imgs(prefix_seq=images_folder, pair_indexes=pair_indexes, start_h=start_h,
                                                  start_w=start_w,
                                                  end_h=end_h, end_w=end_w, downsampling_factor=self.image_downsampling)

            pair_mask_imgs, pair_sparse_depth_imgs, _, _ = \
                utils.get_torch_training_data(pair_extrinsics=pair_extrinsic_matrices,
                                              pair_projections=
                                              pair_projection_matrices, pair_indexes=pair_indexes,
                                              point_cloud=self.point_cloud_per_seq[folder_str],
                                              mask_boundary=self.mask_boundary_per_seq[folder_str],
                                              view_indexes_per_point=self.view_indexes_per_point_per_seq[folder_str],
                                              visible_view_indexes=self.visible_view_indexes_per_seq[folder_str],
                                              clean_point_list=self.clean_point_list_per_seq[
                                                  folder_str])

            # Scale the sparse depth map
            pair_sparse_depth_imgs[0] /= self.estimated_scale_per_seq[folder_str]
            pair_sparse_depth_imgs[1] /= self.estimated_scale_per_seq[folder_str]

            # Format training data
            training_color_img_1 = pair_imgs[0]
            height, width, _ = training_color_img_1.shape

            # Full indexes
            training_sparse_depth_img_1 = utils.type_float_and_reshape(pair_sparse_depth_imgs[0],
                                                                       (height, width, 1))
            training_mask_img_1 = utils.type_float_and_reshape(pair_mask_imgs[0],
                                                               (height, width, 1))
            training_intrinsic_matrix = utils.type_float_and_reshape(
                self.intrinsic_matrix_per_seq[folder_str][:3, :3],
                (3, 3))
            training_mask_boundary = utils.type_float_and_reshape(
                self.mask_boundary_per_seq[folder_str].astype(np.float32) / 255.0,
                (height, width, 1))
            training_mask_boundary[training_mask_boundary > 0.9] = 1.0
            training_mask_boundary[training_mask_boundary <= 0.9] = 0.0

            # Convert the extrinsic matrix to T^(world)_(camera)
            temp_extrinsic = pair_extrinsic_matrices[0]
            training_extrinsic_1 = np.zeros_like(pair_extrinsic_matrices[0])
            training_extrinsic_1[:3, :3] = np.transpose(temp_extrinsic[:3, :3])
            training_extrinsic_1[:3, 3] = np.matmul(-np.transpose(temp_extrinsic[:3, :3]), temp_extrinsic[:3, 3]) \
                                          / self.estimated_scale_per_seq[folder_str]
            training_extrinsic_1[3, 3] = 1.0
            training_extrinsic_1 = utils.type_float_and_reshape(training_extrinsic_1, (4, 4))

            # Normalize
            training_color_img_1 = self.normalize(image=training_color_img_1)['image']

            return [img_to_tensor(training_color_img_1),
                    img_to_tensor(training_sparse_depth_img_1),
                    img_to_tensor(training_mask_img_1),
                    img_to_tensor(training_mask_boundary),
                    torch.from_numpy(training_extrinsic_1),
                    torch.from_numpy(training_intrinsic_matrix),
                    img_index, folder_str, start_h, start_w]


class DescriptorDataset(Dataset):
    def __init__(self, image_file_names, folder_list,
                 image_downsampling, network_downsampling, load_intermediate_data,
                 intermediate_data_root, visible_interval=30, num_pre_workers=12, inlier_percentage=0.998,
                 adjacent_range=(1, 1), num_iter=None,
                 sampling_size=10, heatmap_sigma=5.0, phase='Train', shared_views=False):

        self.image_file_names = sorted(image_file_names)
        self.folder_list = folder_list
        self.parent_sequence = folder_list[0]
        assert (len(adjacent_range) == 2)
        self.adjacent_range = adjacent_range
        self.inlier_percentage = inlier_percentage
        self.image_downsampling = image_downsampling
        self.network_downsampling = network_downsampling
        self.visible_interval = visible_interval
        self.sampling_size = sampling_size
        self.num_iter = num_iter
        self.heatmap_sigma = heatmap_sigma
        self.num_pre_workers = min(len(folder_list), num_pre_workers)
        self.normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)
        self.phase = phase
        self.shared_views = shared_views

        self.clean_point_list_per_seq = {}
        self.intrinsic_matrix_per_seq = {}
        self.point_cloud_per_seq = {}
        self.mask_boundary_per_seq = {}
        self.view_indexes_per_point_per_seq = {}
        self.selected_indexes_per_seq = {}
        self.visible_view_indexes_per_seq = {}
        self.extrinsics_per_seq = {}
        self.projection_per_seq = {}
        self.crop_positions_per_seq = {}
        self.estimated_scale_per_seq = {}
        self.shared_point_views_per_seq = {}

        precompute_path = intermediate_data_root / ("precompute_" + str(
            self.image_downsampling) + "_" + str(self.network_downsampling) + ".pkl")

        # Save all intermediate results to hard disk for quick access later on
        if not load_intermediate_data or not precompute_path.exists():
            queue_size = Queue()
            queue_clean_point_list = Queue()
            queue_intrinsic_matrix = Queue()
            queue_point_cloud = Queue()
            queue_mask_boundary = Queue()
            queue_view_indexes_per_point = Queue()
            queue_selected_indexes = Queue()
            queue_visible_view_indexes = Queue()
            queue_extrinsics = Queue()
            queue_projection = Queue()
            queue_crop_positions = Queue()
            queue_estimated_scale = Queue()
            queue_shared_point_views = Queue()

            process_pool = []

            interval = len(self.folder_list) / self.num_pre_workers

            # Go through the entire image list to find the largest required h and w
            for i in range(self.num_pre_workers):
                process_pool.append(Process(target=find_common_valid_size, args=(
                    self.folder_list[
                    int(np.round(i * interval)): min(int(np.round((i + 1) * interval)), len(self.folder_list))],
                    self.image_downsampling,
                    self.network_downsampling,
                    queue_size)))

            for t in process_pool:
                t.start()

            largest_h = 0
            largest_w = 0

            for t in process_pool:
                while t.is_alive():
                    while not queue_size.empty():
                        h, w = queue_size.get()
                        if h > largest_h:
                            largest_h = h
                        if w > largest_w:
                            largest_w = w
                    t.join(timeout=1)

            while not queue_size.empty():
                h, w = queue_size.get()
                if h > largest_h:
                    largest_h = h
                if w > largest_w:
                    largest_w = w

            if largest_h == 0 or largest_w == 0:
                print("image size calculation failed.")
                raise IOError
            print("Largest image size is: ", largest_h, largest_w)

            print("Start pre-processing dataset...")
            process_pool = []
            for i in range(self.num_pre_workers):
                process_pool.append(Process(target=pre_processing_data,
                                            args=(i, self.folder_list[int(np.round(i * interval)):
                                                                      min(int(np.round((i + 1) * interval)),
                                                                          len(self.folder_list))],
                                                  self.image_downsampling, self.network_downsampling,
                                                  self.inlier_percentage, self.visible_interval, largest_h, largest_w,
                                                  queue_clean_point_list,
                                                  queue_intrinsic_matrix, queue_point_cloud,
                                                  queue_mask_boundary, queue_view_indexes_per_point,
                                                  queue_selected_indexes,
                                                  queue_visible_view_indexes,
                                                  queue_extrinsics, queue_projection,
                                                  queue_crop_positions,
                                                  queue_estimated_scale,
                                                  queue_shared_point_views)))

            for t in process_pool:
                t.start()

            count = 0
            for t in process_pool:
                print("Waiting for {:d}th process to complete".format(count))
                count += 1
                while t.is_alive():
                    while not queue_selected_indexes.empty():
                        folder, selected_indexes = queue_selected_indexes.get()
                        self.selected_indexes_per_seq[folder] = selected_indexes
                    while not queue_visible_view_indexes.empty():
                        folder, visible_view_indexes = queue_visible_view_indexes.get()
                        self.visible_view_indexes_per_seq[folder] = visible_view_indexes
                    while not queue_view_indexes_per_point.empty():
                        folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                        self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
                    while not queue_clean_point_list.empty():
                        folder, clean_point_list = queue_clean_point_list.get()
                        self.clean_point_list_per_seq[folder] = clean_point_list
                    while not queue_intrinsic_matrix.empty():
                        folder, intrinsic_matrix = queue_intrinsic_matrix.get()
                        self.intrinsic_matrix_per_seq[folder] = intrinsic_matrix
                    while not queue_extrinsics.empty():
                        folder, extrinsics = queue_extrinsics.get()
                        self.extrinsics_per_seq[folder] = extrinsics
                    while not queue_projection.empty():
                        folder, projection = queue_projection.get()
                        self.projection_per_seq[folder] = projection
                    while not queue_crop_positions.empty():
                        folder, crop_positions = queue_crop_positions.get()
                        self.crop_positions_per_seq[folder] = crop_positions
                    while not queue_point_cloud.empty():
                        folder, point_cloud = queue_point_cloud.get()
                        self.point_cloud_per_seq[folder] = point_cloud
                    while not queue_mask_boundary.empty():
                        folder, mask_boundary = queue_mask_boundary.get()
                        self.mask_boundary_per_seq[folder] = mask_boundary
                    while not queue_estimated_scale.empty():
                        folder, estiamted_scale = queue_estimated_scale.get()
                        self.estimated_scale_per_seq[folder] = estiamted_scale
                    while not queue_shared_point_views.empty():
                        folder, shared_point_views = queue_shared_point_views.get()
                        self.shared_point_views_per_seq[folder] = shared_point_views
                    t.join(timeout=1)

            while not queue_selected_indexes.empty():
                folder, selected_indexes = queue_selected_indexes.get()
                self.selected_indexes_per_seq[folder] = selected_indexes
            while not queue_visible_view_indexes.empty():
                folder, visible_view_indexes = queue_visible_view_indexes.get()
                self.visible_view_indexes_per_seq[folder] = visible_view_indexes
            while not queue_view_indexes_per_point.empty():
                folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
            while not queue_clean_point_list.empty():
                folder, clean_point_list = queue_clean_point_list.get()
                self.clean_point_list_per_seq[folder] = clean_point_list
            while not queue_intrinsic_matrix.empty():
                folder, intrinsic_matrix = queue_intrinsic_matrix.get()
                self.intrinsic_matrix_per_seq[folder] = intrinsic_matrix
            while not queue_extrinsics.empty():
                folder, extrinsics = queue_extrinsics.get()
                self.extrinsics_per_seq[folder] = extrinsics
            while not queue_projection.empty():
                folder, projection = queue_projection.get()
                self.projection_per_seq[folder] = projection
            while not queue_crop_positions.empty():
                folder, crop_positions = queue_crop_positions.get()
                self.crop_positions_per_seq[folder] = crop_positions
            while not queue_point_cloud.empty():
                folder, point_cloud = queue_point_cloud.get()
                self.point_cloud_per_seq[folder] = point_cloud
            while not queue_mask_boundary.empty():
                folder, mask_boundary = queue_mask_boundary.get()
                self.mask_boundary_per_seq[folder] = mask_boundary
            while not queue_estimated_scale.empty():
                folder, estimated_scale = queue_estimated_scale.get()
                self.estimated_scale_per_seq[folder] = estimated_scale
            while not queue_shared_point_views.empty():
                folder, shared_point_views = queue_shared_point_views.get()
                self.shared_point_views_per_seq[folder] = shared_point_views
            print("Pre-processing complete.")

            # Store all intermediate information to a single data file
            with open(str(precompute_path), "wb") as f:
                pickle.dump(
                    [self.crop_positions_per_seq, self.selected_indexes_per_seq,
                     self.visible_view_indexes_per_seq,
                     self.point_cloud_per_seq, self.intrinsic_matrix_per_seq,
                     self.mask_boundary_per_seq, self.view_indexes_per_point_per_seq, self.extrinsics_per_seq,
                     self.projection_per_seq, self.clean_point_list_per_seq,
                     self.image_downsampling, self.network_downsampling, self.inlier_percentage,
                     self.estimated_scale_per_seq, self.shared_point_views_per_seq],
                    f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(str(precompute_path), "rb") as f:
                [self.crop_positions_per_seq, self.selected_indexes_per_seq,
                 self.visible_view_indexes_per_seq,
                 self.point_cloud_per_seq, self.intrinsic_matrix_per_seq,
                 self.mask_boundary_per_seq, self.view_indexes_per_point_per_seq, self.extrinsics_per_seq,
                 self.projection_per_seq, self.clean_point_list_per_seq,
                 self.image_downsampling, self.network_downsampling,
                 self.inlier_percentage, self.estimated_scale_per_seq, self.shared_point_views_per_seq] = pickle.load(f)

    def __len__(self):
        if self.num_iter is not None:
            return self.num_iter
        else:
            return len(self.image_file_names)

    def __getitem__(self, idx):
        if self.phase == "Train" or self.phase == "Validation":
            while True:
                img_file_name = self.image_file_names[idx % len(self.image_file_names)]
                # Retrieve the folder path
                folder = img_file_name.parents[1]
                images_folder = folder / "images"
                folder_str = str(folder)
                # Randomly pick one adjacent frame
                # We assume the filename has 8 logits followed by ".jpg"
                if folder_str not in self.crop_positions_per_seq:
                    print("{} not in stored data".format(folder_str))
                    idx = np.random.randint(0, len(self.image_file_names))
                    continue
                
                # Here somewhere
                start_h, end_h, start_w, end_w = self.crop_positions_per_seq[folder_str]

                if self.shared_views:
                    pos, increment, shared = \
                        utils.generating_pos_and_increment_localization(idx=idx,
                                                                        visible_view_indexes=
                                                                        self.visible_view_indexes_per_seq[folder_str],
                                                                        shared_point_views_matrix=
                                                                        self.shared_point_views_per_seq[folder_str])
                else:
                    pos, increment = utils.generating_pos_and_increment(idx=idx,
                                                                        visible_view_indexes=
                                                                        self.visible_view_indexes_per_seq[folder_str],
                                                                        adjacent_range=self.adjacent_range)
                    shared = None

                # Get pair visible view indexes and pair extrinsic and projection matrices
                pair_indexes = [self.visible_view_indexes_per_seq[folder_str][pos],
                                self.visible_view_indexes_per_seq[folder_str][pos + increment]]
                pair_projection_matrices = [self.projection_per_seq[folder_str][pos],
                                            self.projection_per_seq[folder_str][pos + increment]]

                pair_imgs = utils.get_pair_color_imgs(prefix_seq=images_folder, pair_indexes=pair_indexes,
                                                      start_h=start_h,
                                                      start_w=start_w,
                                                      end_h=end_h, end_w=end_w,
                                                      downsampling_factor=self.image_downsampling)
                height, width = pair_imgs[0].shape[:2]
                feature_matches = \
                    utils.get_torch_training_data_feature_matching(height=height, width=width,
                                                                   pair_projections=
                                                                   pair_projection_matrices,
                                                                   pair_indexes=pair_indexes,
                                                                   point_cloud=self.point_cloud_per_seq[
                                                                       folder_str],
                                                                   mask_boundary=self.mask_boundary_per_seq[folder_str],
                                                                   view_indexes_per_point=
                                                                   self.view_indexes_per_point_per_seq[folder_str],
                                                                   visible_view_indexes=
                                                                   self.visible_view_indexes_per_seq[
                                                                       folder_str],
                                                                   clean_point_list=
                                                                   self.clean_point_list_per_seq[
                                                                       folder_str])

                if shared and feature_matches.shape[0] > 0:
                    sampled_feature_matches_indexes = \
                        np.asarray(np.random.choice(np.arange(feature_matches.shape[0]), 
                                                    size=self.sampling_size), dtype=np.int32).reshape((-1,))
                    sampled_feature_matches = np.asarray(feature_matches[sampled_feature_matches_indexes, :],
                                                         dtype=np.float32).reshape((self.sampling_size, 4))

                    training_heatmaps_1, training_heatmaps_2 = \
                        utils.generate_heatmap_from_locations(sampled_feature_matches, height, 
                                                              width, self.heatmap_sigma)
                    positive_example = True
                    break

                elif shared and feature_matches.shape[0] == 0:
                    idx = np.random.randint(0, 100)
                    continue

                else:
                    sampled_feature_matches = \
                        utils.generate_negative_feature_matches(mask_boundary=
                                                                self.mask_boundary_per_seq[folder_str],
                                                                sampling_size=self.sampling_size)

                    training_heatmaps_1 = np.zeros((self.sampling_size, height, width), dtype=np.float32)
                    training_heatmaps_2 = np.zeros((self.sampling_size, height, width), dtype=np.float32)
                    positive_example = False
                    break

            # Format training data
            training_color_img_1 = pair_imgs[0]
            training_color_img_2 = pair_imgs[1]

            training_mask_boundary = utils.type_float_and_reshape(
                self.mask_boundary_per_seq[folder_str].astype(np.float32) / 255.0,
                (height, width, 1))
            training_mask_boundary[training_mask_boundary > 0.9] = 1.0
            training_mask_boundary[training_mask_boundary <= 0.9] = 0.0

            source_feature_2D_locations = sampled_feature_matches[:, :2]
            target_feature_2D_locations = sampled_feature_matches[:, 2:]

            source_feature_1D_locations = np.zeros(
                (sampled_feature_matches.shape[0], 1), dtype=np.int32)
            target_feature_1D_locations = np.zeros(
                (sampled_feature_matches.shape[0], 1), dtype=np.int32)

            clipped_source_feature_2D_locations = source_feature_2D_locations
            clipped_source_feature_2D_locations[:, 0] = np.clip(clipped_source_feature_2D_locations[:, 0], a_min=0,
                                                                a_max=width - 1)
            clipped_source_feature_2D_locations[:, 1] = np.clip(clipped_source_feature_2D_locations[:, 1], a_min=0,
                                                                a_max=height - 1)

            clipped_target_feature_2D_locations = target_feature_2D_locations
            clipped_target_feature_2D_locations[:, 0] = np.clip(clipped_target_feature_2D_locations[:, 0], a_min=0,
                                                                a_max=width - 1)
            clipped_target_feature_2D_locations[:, 1] = np.clip(clipped_target_feature_2D_locations[:, 1], a_min=0,
                                                                a_max=height - 1)

            source_feature_1D_locations[:, 0] = np.round(clipped_source_feature_2D_locations[:, 0]) + \
                                                np.round(clipped_source_feature_2D_locations[:, 1]) * width
            target_feature_1D_locations[:, 0] = np.round(clipped_target_feature_2D_locations[:, 0]) + \
                                                np.round(clipped_target_feature_2D_locations[:, 1]) * width

            # Normalize
            training_color_img_1 = self.normalize(image=training_color_img_1)['image']
            training_color_img_2 = self.normalize(image=training_color_img_2)['image']
    
            return [img_to_tensor(training_color_img_1), img_to_tensor(training_color_img_2),
                    torch.from_numpy(source_feature_1D_locations),
                    torch.from_numpy(target_feature_1D_locations),
                    torch.from_numpy(source_feature_2D_locations),
                    torch.from_numpy(target_feature_2D_locations),
                    torch.from_numpy(training_heatmaps_1),
                    torch.from_numpy(training_heatmaps_2),
                    img_to_tensor(training_mask_boundary),
                    folder_str, str(img_file_name), positive_example]

        elif self.phase == "Loading":
            img_file_name = self.image_file_names[idx]
            # Retrieve the folder path
            folder_str = str(self.parent_sequence)
            start_h, end_h, start_w, end_w = self.crop_positions_per_seq[folder_str]
            color_img = utils.read_color_img(img_file_name, start_h, end_h, start_w, end_w,
                                             self.image_downsampling)
            training_color_img_1 = color_img
            height, width, _ = training_color_img_1.shape

            training_mask_boundary = utils.type_float_and_reshape(
                self.mask_boundary_per_seq[folder_str].astype(np.float32) / 255.0,
                (height, width, 1))
            training_mask_boundary[training_mask_boundary > 0.9] = 1.0
            training_mask_boundary[training_mask_boundary <= 0.9] = 0.0

            # Normalize
            training_color_img_1 = self.normalize(image=training_color_img_1)['image']

            return [img_to_tensor(training_color_img_1),
                    img_to_tensor(training_mask_boundary),
                    str(img_file_name), folder_str, start_h, start_w]