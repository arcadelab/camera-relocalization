'''
This script has been modified from the original pipeline published in the work titled: 
"Reconstructing Sinus Anatomy from Endoscopic Video - Towards a Radiation-free Approach for Quantitative Longitudinal Assessment".

Code source: https://github.com/lppllppl920/DenseReconstruction-Pytorch

Author: Xingtong Liu, Maia Stiber, Jindan Huang, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''
import os
import cv2
import h5py
import tqdm
import argparse
import numpy as np
from pathlib import Path
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

import general_utils
import database_utils as idb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense feature matching',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--image_downsampling', type=float, default=4.0,
                        help='input image downsampling rate')
    parser.add_argument('--network_downsampling', type=int, default=64, 
                        help='network bottom layer downsampling')
    parser.add_argument('--input_size', nargs='+', type=int, required=True,
                        help='input size')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='number of workers for input data loader')
    parser.add_argument('--load_intermediate_data', action='store_true', 
                        help='whether to load intermediate data')
    parser.add_argument('--data_root', type=str, required=True, 
                        help='path to the data for feature matching')
    parser.add_argument('--sequence_root', type=str, required=True,
                        help='root of the specific video sequence')
    parser.add_argument('--matching_trained_model_path', type=str, required=True, 
                        help='path to the trained model')
    parser.add_argument('--matching_model_description', type=str, required=True, 
                        help='description of model providing base for matching')
    parser.add_argument('--precompute_root', type=str, required=True, 
                        help='path to the precompute data')
    parser.add_argument('--feature_length', type=int, default=128, 
                        help='output channel dimension of network')
    parser.add_argument('--filter_growth_rate', type=int, default=10, 
                        help='filter growth rate of network')
    parser.add_argument('--max_feature_detection', type=int, default=3000,
                        help='max allowed number of detected features per frame')
    parser.add_argument('--cross_check_distance', type=float, default=3.0,
                        help='max cross check distance for valid matches')
    parser.add_argument('--patient_id', nargs='+', type=int,
                        help='list patient ids')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='gpu id for matching generation')
    parser.add_argument('--temporal_range', type=int, default=30, 
                        help='range for temporal sampling')
    parser.add_argument('--test_keypoint_num', type=int, default=200, 
                        help='number of keypoints used for quick spatial testing')
    parser.add_argument('--residual_threshold', type=float, default=5.0, 
                        help='pixel threshold for ransac estimation')
    parser.add_argument('--octave_layers', type=int, default=8)
    parser.add_argument('--contrast_threshold', type=float, default=0.00005)
    parser.add_argument('--edge_threshold', type=float, default=100)
    parser.add_argument('--sigma', type=float, default=1.1)
    parser.add_argument('--skip_interval', type=int, default=5,
                        help="number of skipping frames in searching state")
    parser.add_argument('--min_inlier_ratio', type=float, default=0.2,
                        help="minimum inlier ratio of ransac")
    parser.add_argument('--hysterisis_factor', type=float, default=0.7,
                        help="factor of the inlier ratio in the spatial_range state")

    args = parser.parse_args()
    
    PREOPERATIVE_DATABASE = os.path.join(args.sequence_root, "dreco_base_{}.db".format(args.matching_model_description))
    idb.create_db_tables(db_path=PREOPERATIVE_DATABASE)
    intra_db = idb.ReconstrunctionDB(PREOPERATIVE_DATABASE)

    SIFT_PREOPERATIVE_DATABASE = os.path.join(args.sequence_root, "sift_dreco.db")
    idb.create_db_tables(db_path=SIFT_PREOPERATIVE_DATABASE)
    sift_db = idb.ReconstrunctionDB(SIFT_PREOPERATIVE_DATABASE)

    if not Path(args.precompute_root).exists():
        Path(args.precompute_root).mkdir(parents=True)

    print("SIFT detector creating...")
    sift = cv2.SIFT_create(nfeatures=args.max_feature_detection, nOctaveLayers=args.octave_layers,
                           contrastThreshold=args.contrast_threshold,
                           edgeThreshold=args.edge_threshold, sigma=args.sigma)

    colors_list, boundary, feature_maps_list, start_h, start_w, image_names_list = \
        general_utils.gather_feature_matching_data(feature_descriptor_model_path=Path(args.matching_trained_model_path),
                                                   sub_folder=Path(args.sequence_root),
                                                   data_root=Path(args.data_root), image_downsampling=args.image_downsampling,
                                                   network_downsampling=args.network_downsampling,
                                                   load_intermediate_data=args.load_intermediate_data,
                                                   precompute_root=Path(args.precompute_root),
                                                   batch_size=args.batch_size, id_list=args.patient_id,
                                                   filter_growth_rate=args.filter_growth_rate,
                                                   feature_length=args.feature_length, gpu_id=args.gpu_id)

    # Erode the boundary to remove near-boundary matches
    kernel = np.ones((5, 5), np.uint8)
    boundary = cv2.erode(boundary, kernel, iterations=3)
    
    f_matches = None
    print("Extracting keypoint locations...")
    sift_keypoints_list, sift_keypoint_location_list_1D, sift_keypoint_location_list_2D, sift_descriptions_list = \
        general_utils.extract_keypoints(sift, colors_list, boundary, args.input_size[0], args.input_size[1])

    f_matches = h5py.File(str(Path(args.sequence_root) / "feature_matches.hdf5"), 'w')
    dataset_matches = f_matches.create_dataset('matches', (0, 4, 1),
                                               maxshape=(None, 4, 1), chunks=(40960, 4, 1),
                                               compression="gzip", compression_opts=9, dtype='int16')

    feature_length, height, width = feature_maps_list[0].shape
    frame_count_in_total = len(colors_list)

    tq = tqdm.tqdm(total=frame_count_in_total * (frame_count_in_total - 1) // 2)
    for i in range(frame_count_in_total - 1):
        color_1 = colors_list[i]
        feature_map_1 = feature_maps_list[i].cuda(args.gpu_id)

        sift_keypoint_1 = sift_keypoints_list[i]
        np.random.seed(10086)
        random_indexes = list(np.random.choice(range(0, len(sift_keypoint_1)), args.test_keypoint_num,
                                               replace=False))
        sift_keypoint_locations_1D_1 = sift_keypoint_location_list_1D[i]
        sift_keypoint_locations_2D_1 = sift_keypoint_location_list_2D[i]
        sift_descriptors_1 = sift_descriptions_list[i]

        cur_state = "temporal_range"
        for j in range(1, len(colors_list) - i):
            tq.set_description(cur_state)

            if cur_state == "temporal_range":
                if j > args.temporal_range:
                    cur_state = "searching"

            if cur_state == "searching":
                if j % args.skip_interval != 0:
                    tq.update(1)
                    continue

            if cur_state == "spatial_range":
                pass

            color_2 = colors_list[i + j]
            feature_map_2 = feature_maps_list[i + j].cuda(args.gpu_id)

            if cur_state == "temporal_range" or cur_state == "spatial_range":
                x = general_utils.feature_matching_single_generation(
                        feature_map_1=feature_map_1, 
                        feature_map_2=feature_map_2,
                        kps_1D_1=sift_keypoint_locations_1D_1,
                        cross_check_distance=args.cross_check_distance,
                        gpu_id=args.gpu_id)
            elif cur_state == "searching":
                x = general_utils.feature_matching_single_generation(
                        feature_map_1=feature_map_1,
                        feature_map_2=feature_map_2,
                        kps_1D_1=sift_keypoint_locations_1D_1[random_indexes],
                        cross_check_distance=args.cross_check_distance,
                        gpu_id=args.gpu_id)

            if x is None:
                tq.update(1)
                continue

            source_keypoint_indexes, target_keypoint_locations = x

            if cur_state == "searching":
                source_keypoint_indexes = [random_indexes[source_keypoint_index] for
                                           source_keypoint_index in source_keypoint_indexes]

            source_keypoint_locations = sift_keypoint_locations_2D_1[source_keypoint_indexes,:].reshape((-1, 2))
            source_sift_descriptors = sift_descriptors_1[source_keypoint_indexes,:]
     
            general_utils.save_descriptor_locations(source_feature_locations=source_keypoint_locations,
                                                    source_image_name=image_names_list[i],
                                                    start_h=start_h, start_w=start_w, 
                                                    image_downsampling=args.image_downsampling, 
                                                    save_db=intra_db)

            general_utils.save_sift_descriptors(source_feature_locations=source_keypoint_locations,
                                                source_sift_descriptors=source_sift_descriptors,
                                                source_image_name=image_names_list[i],
                                                start_h=start_h, start_w=start_w, 
                                                image_downsampling=args.image_downsampling, 
                                                save_db=sift_db)

            source_keypoint_locations[:, 0] = args.image_downsampling * (
                    source_keypoint_locations[:, 0] + start_w)
            source_keypoint_locations[:, 1] = args.image_downsampling * (
                    source_keypoint_locations[:, 1] + start_h)

            target_keypoint_locations[:, 0] = args.image_downsampling * (
                    target_keypoint_locations[:, 0] + start_w)
            target_keypoint_locations[:, 1] = args.image_downsampling * (
                    target_keypoint_locations[:, 1] + start_h)
                    
            try:
                model, inliers = ransac((source_keypoint_locations,
                                         target_keypoint_locations),
                                         FundamentalMatrixTransform, min_samples=8,
                                         residual_threshold=args.residual_threshold, max_trials=5, 
                                         random_state=1111)
            except ValueError:
                tq.set_postfix(
                    source_frame_index='{:d}'.format(i),
                    target_frame_index='{:d}'.format(i + j),
                    point_num='{:d}'.format(target_keypoint_locations.shape[0]))
                tq.update(1)
                continue

            try:
                inlier_ratio = np.sum(inliers) / source_keypoint_locations.shape[0]
            except Exception as e:
                tq.set_postfix(
                    source_frame_index='{:d}'.format(i),
                    target_frame_index='{:d}'.format(i + j),
                    point_num='{:d}'.format(target_keypoint_locations.shape[0]))
                print("Warning except: {}".format(e))
                tq.update(1)
                continue       

            if j == 1:
                mean_inlier_ratio = inlier_ratio
            elif j <= args.temporal_range:
                mean_inlier_ratio = mean_inlier_ratio * ((j - 1) / j) + inlier_ratio * (1 / j)
            elif j == args.temporal_range + 1:
                mean_inlier_ratio = max(args.min_inlier_ratio, mean_inlier_ratio)

            if cur_state == "temporal_range":
                start_index = dataset_matches.shape[0]
                dataset_matches.resize(
                    (dataset_matches.shape[0] + target_keypoint_locations.shape[0] + 1, 4, 1))
                dataset_matches[start_index, :, :] = np.asarray(
                    [target_keypoint_locations.shape[0], i, i + j, -1]).reshape((4, 1))

                dataset_matches[start_index + 1:start_index + 1 + target_keypoint_locations.shape[0],
                :] = \
                    np.concatenate([source_keypoint_locations.reshape((-1, 2)),
                                    target_keypoint_locations.reshape((-1, 2))], axis=1).reshape(
                        (-1, 4, 1)).astype(np.int16)
                tq.set_description(cur_state)
                tq.set_postfix(
                    source_frame_index='{:d}'.format(i),
                    target_frame_index='{:d}'.format(i + j),
                    mean_inlier_ratio='{:.3f}'.format(mean_inlier_ratio))

            elif cur_state == "searching":
                if inlier_ratio >= mean_inlier_ratio:
                    cur_state = "spatial_range"
                    # Redo the feature matching with full set of keypoints
                    x = general_utils.feature_matching_single_generation(
                            feature_map_1=feature_map_1,
                            feature_map_2=feature_map_2,
                            kps_1D_1=sift_keypoint_locations_1D_1,
                            cross_check_distance=args.cross_check_distance,
                            gpu_id=args.gpu_id)

                    if x is None:
                        tq.update(1)
                        continue

                    source_keypoint_indexes, target_keypoint_locations = x
                    source_keypoint_locations = sift_keypoint_locations_2D_1[source_keypoint_indexes, :].reshape((-1, 2))
                    source_sift_descriptors = sift_descriptors_1[source_keypoint_indexes,:]
                    
                    general_utils.save_descriptor_locations(source_feature_locations=source_keypoint_locations,
                                                            source_image_name=image_names_list[i],
                                                            start_h=start_h, start_w=start_w, 
                                                            image_downsampling=args.image_downsampling, 
                                                            save_db=intra_db)

                    general_utils.save_sift_descriptors(source_feature_locations=source_keypoint_locations,
                                                        source_sift_descriptors=source_sift_descriptors,
                                                        source_image_name=image_names_list[i],
                                                        start_h=start_h, start_w=start_w, 
                                                        image_downsampling=args.image_downsampling, 
                                                        save_db=sift_db)

                    source_keypoint_locations[:, 0] = args.image_downsampling * (source_keypoint_locations[:, 0] + start_w)
                    source_keypoint_locations[:, 1] = args.image_downsampling * (source_keypoint_locations[:, 1] + start_h)

                    target_keypoint_locations[:, 0] = args.image_downsampling * (
                            target_keypoint_locations[:, 0] + start_w)
                    target_keypoint_locations[:, 1] = args.image_downsampling * (
                            target_keypoint_locations[:, 1] + start_h)

                    # Save locations and now descriptros
                    start_index = dataset_matches.shape[0]
                    dataset_matches.resize(
                        (dataset_matches.shape[0] + target_keypoint_locations.shape[0] + 1, 4, 1))
                    dataset_matches[start_index, :, :] = np.asarray(
                        [target_keypoint_locations.shape[0], i, i + j, -1]).reshape((4, 1))

                    dataset_matches[start_index + 1:start_index + 1 + target_keypoint_locations.shape[0],
                    :] = \
                        np.concatenate([source_keypoint_locations.reshape((-1, 2)),
                                        target_keypoint_locations.reshape((-1, 2))], axis=1).reshape(
                            (-1, 4, 1)).astype(np.int16)
                    tq.set_description(cur_state)
                    tq.set_postfix(
                        source_frame_index='{:d}'.format(i),
                        target_frame_index='{:d}'.format(i + j),
                        inlier_ratio='{:.3f}'.format(inlier_ratio))
                else:
                    tq.set_description(cur_state)
                    tq.set_postfix(
                        source_frame_index='{:d}'.format(i),
                        target_frame_index='{:d}'.format(i + j))

            elif cur_state == "spatial_range":
                # Leave a bit of hyterisis space for spatial_range state to allow for more frame matches
                if inlier_ratio >= args.hysterisis_factor * mean_inlier_ratio:
                    # Save locations and now descriptros
                    start_index = dataset_matches.shape[0]
                    dataset_matches.resize(
                        (dataset_matches.shape[0] + target_keypoint_locations.shape[0] + 1, 4, 1))
                    dataset_matches[start_index, :, :] = np.asarray(
                        [target_keypoint_locations.shape[0], i, i + j, -1]).reshape((4, 1))

                    dataset_matches[start_index + 1:start_index + 1 + target_keypoint_locations.shape[0],
                    :] = \
                        np.concatenate([source_keypoint_locations.reshape((-1, 2)),
                                        target_keypoint_locations.reshape((-1, 2))], axis=1).reshape(
                            (-1, 4, 1)).astype(np.int16)
                    tq.set_description(cur_state)
                    tq.set_postfix(
                        source_frame_index='{:d}'.format(i),
                        target_frame_index='{:d}'.format(i + j),
                        inlier_ratio='{:.3f}'.format(inlier_ratio))
                else:
                    cur_state = "searching"
                    tq.set_description(cur_state)
                    tq.set_postfix(
                        source_frame_index='{:d}'.format(i),
                        target_frame_index='{:d}'.format(i + j))
                    tq.update(1)
                    continue

            tq.update(1)

    tq.close()

    if f_matches is not None:
        f_matches.close()
        
    intra_db.closeDB()
    sift_db.closeDB()