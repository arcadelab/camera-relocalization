import os
import cv2
import torch
import tqdm
import numpy as np
from glob import glob
from plyfile import PlyData
import albumentations as albu
from albumentations.pytorch.functional import img_to_tensor

import utils
import models
import dataset

def read_visible_view_indexes(prefix_seq):
    path = prefix_seq / 'visible_view_indexes'
    if not path.exists():
        return []

    visible_view_indexes = []
    with open(str(path)) as fp:
        for line in fp:
            visible_view_indexes.append(int(line))

    return visible_view_indexes

def extract_keypoints(descriptor, colors_list, boundary, height, width):
    keypoints_list = []
    descriptions_list = []
    keypoints_list_1D = []
    boundary = np.uint8(255 * boundary.reshape((height, width)))
    for i in range(len(colors_list)):
        color_1 = colors_list[i]
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
        kps, des = descriptor.detectAndCompute(color_1, mask=boundary)
        keypoints_list.append(kps)
        descriptions_list.append(des)
        temp = []
        for point in kps:
            temp.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
        keypoints_list_1D.append(temp)

    return keypoints_list, keypoints_list_1D, descriptions_list

def type_float_and_reshape(array, shape):
    array = array.astype(np.float32)
    return array.reshape(shape)

def read_selected_indexes(prefix_seq):
    selected_indexes = []
    with open(str(prefix_seq / 'selected_indexes')) as fp:
        for line in fp:
            selected_indexes.append(int(line))
    return selected_indexes

def get_all_subfolder_names(root, id_range):
    folder_list = []
    for i in id_range:
        folder_list += list(root.glob('{}/*/'.format(i)))
    folder_list.sort()
    return folder_list

def get_all_color_image_names_in_sequence(sequence_root):
    filenames = glob(os.path.join(str(sequence_root), "images", "*.jpg"))
    filenames.sort()

    return filenames

def gather_feature_matching_data(feature_descriptor_model_path, sub_folder, data_root, image_downsampling,
                                 network_downsampling, load_intermediate_data, precompute_root,
                                 batch_size, id_list, filter_growth_rate, feature_length, gpu_id):

    feature_descriptor_model = models.FCDenseNetFeature(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length)

    # Multi-GPU running
    feature_descriptor_model = torch.nn.DataParallel(feature_descriptor_model, device_ids=[gpu_id])
    feature_descriptor_model.eval()

    if feature_descriptor_model_path.exists():
        print("Loading {:s} ...".format(str(feature_descriptor_model_path)))
        state = torch.load(str(feature_descriptor_model_path), map_location='cuda:{}'.format(gpu_id))
        feature_descriptor_model.load_state_dict(state["model"])
    else:
        print("No pre-trained model detected")
        raise OSError
    del state

    video_frame_filenames = get_all_color_image_names_in_sequence(sub_folder)

    print("Gathering feature matching data for {}".format(str(sub_folder)))
    folder_list = get_all_subfolder_names(data_root, id_list)
    video_dataset = dataset.DescriptorDataset(image_file_names=video_frame_filenames,
                                              folder_list=folder_list,
                                              image_downsampling=image_downsampling,
                                              network_downsampling=network_downsampling,
                                              load_intermediate_data=load_intermediate_data,
                                              intermediate_data_root=precompute_root,
                                              phase="Loading")

    video_loader = torch.utils.data.DataLoader(dataset=video_dataset, batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=batch_size)

    colors_list = []
    feature_maps_list = []
    image_names_list = []

    with torch.no_grad():
        # Update progress bar
        tq = tqdm.tqdm(total=len(video_loader) * batch_size)
        for batch, (colors_1, boundaries, image_names,
                    folders, starts_h, starts_w) in enumerate(video_loader):
            tq.update(batch_size)
            colors_1 = colors_1.cuda(gpu_id)
            if batch == 0:
                boundary = boundaries[0].data.numpy()
                start_h = starts_h[0].item()
                start_w = starts_w[0].item()

            feature_maps_1 = feature_descriptor_model(colors_1)
            for idx in range(colors_1.shape[0]):
                colors_list.append(colors_1[idx].data.cpu().numpy())
                feature_maps_list.append(feature_maps_1[idx].data.cpu())
                image_names_list.append(image_names)
  
    tq.close()
    return colors_list, boundary, feature_maps_list, start_h, start_w, image_names_list

def feature_matching_single_generation(feature_map_1, feature_map_2,
                                       kps_1D_1, cross_check_distance, gpu_id):
    with torch.no_grad():
        # Feature map C x H x W
        feature_length, height, width = feature_map_1.shape
        # Extend 1D locations to B x C x Sampling_size
        keypoint_number = len(kps_1D_1)
        source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
            1, 1,
            keypoint_number).expand(
            -1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors = torch.gather(
            feature_map_1.view(1, feature_length, height * width), 2,
            source_feature_1d_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
                                                               keypoint_number,
                                                               1,
                                                               1).permute(0, 2, 1, 3,  # Original input is channel last, this changes it to channel first for convolution?
                                                                          4).view(1,
                                                                                  keypoint_number,
                                                                                  feature_length,
                                                                                  1, 1)

        # 1 x Sampling_size x H x W
        filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_2.view(1, feature_length, height, width),
            weight=sampled_feature_vectors.view(keypoint_number,
                                                feature_length,
                                                1, 1), padding=0)

        max_reponses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)
        del sampled_feature_vectors, filter_response_map, source_feature_1d_locations
        # query is 1 and train is 2 here
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_reponses.view(-1)
        # Do cross check
        feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors_2 = torch.gather(
            feature_map_2.view(1, feature_length, height * width), 2,
            feature_1d_locations_2.long())
        sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
                                                                   keypoint_number,
                                                                   1,
                                                                   1).permute(0, 2, 1, 3,
                                                                              4).view(1,
                                                                                      keypoint_number,
                                                                                      feature_length,
                                                                                      1, 1)

        # 1 x Sampling_size x H x W
        source_filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_1.view(1, feature_length, height, width),
            weight=sampled_feature_vectors_2.view(keypoint_number,
                                                  feature_length,
                                                  1, 1), padding=0)

        max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)
        del sampled_feature_vectors_2, source_filter_response_map, feature_1d_locations_2

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()

        # We will accept the feature matches if the max indexes here is
        # not far away from the original key point location from descriptor
        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())

        valid_detected_target_2d_locations = torch.cat(
            [torch.fmod(valid_detected_1d_locations_2.float(), width).view(-1, 1),
             torch.floor(valid_detected_1d_locations_2.float() / width).view(-1, 1)],
            dim=1).view(-1, 2).float()
        valid_source_keypoint_indexes = valid_correspondence_indexes.view(-1).data.cpu().numpy()
        valid_detected_target_2d_locations = valid_detected_target_2d_locations.view(-1, 2).data.cpu().numpy()

        return valid_source_keypoint_indexes, valid_detected_target_2d_locations



def save_descriptors(source_feature_locations, source_feature_map, source_image_name, 
                     start_h, start_w, image_downsampling, save_db):

    for loc in np.unique(source_feature_locations, axis=0):
        location_descriptor = source_feature_map[:, int(loc[1]), int(loc[0])]
        img_x = (loc[0] + start_w) * image_downsampling
        img_y = (loc[1] + start_h) * image_downsampling
        img_name = os.path.basename(source_image_name[0])
        img_id = "{},{},{}".format(int(img_name.split(".")[0]), int(img_x), int(img_y))

        save_db.add_dense_descriptor_with_params(desc_id=img_id, 
                                                 coor_x=img_x, 
                                                 coor_y=img_y, 
                                                 descriptor=location_descriptor.cpu().numpy(), 
                                                 img_name=img_name)

def save_descriptor_locations(source_feature_locations, source_image_name, 
                              start_h, start_w, image_downsampling, save_db):

    for loc in np.unique(source_feature_locations, axis=0):
        img_x = (loc[0] + start_w) * image_downsampling
        img_y = (loc[1] + start_h) * image_downsampling
        img_name = os.path.basename(source_image_name[0])
        img_id = "{},{},{}".format(int(img_name.split(".")[0]), int(img_x), int(img_y))

        save_db.add_dense_descriptor_with_params(desc_id=img_id, 
                                                 coor_x=img_x, 
                                                 coor_y=img_y, 
                                                 descriptor=np.array([-1]), 
                                                 img_name=img_name)

def feature_localization(query_feature_map, preop_feature_array, gpu_id, max_keypoint3D_number=5000):
    with torch.no_grad():
        feature_length, height, width = query_feature_map.shape #Feature map C x H x W
        response_array = np.zeros((preop_feature_array.shape[0], 3)) # Number of 3D points, times 3        
        total_keypoint_number = preop_feature_array.shape[0]
        num_steps = total_keypoint_number // max_keypoint3D_number + int(total_keypoint_number % max_keypoint3D_number > 0)
        initial = 0
        final = 0
        for i in range(num_steps):
            initial = i * max_keypoint3D_number
            final = (i + 1) * max_keypoint3D_number            
            if final > total_keypoint_number:
                final = total_keypoint_number
            valid_keypoint3d_num = final - initial
            feature_vector = preop_feature_array[initial:final, :]
            feature_vector = feature_vector.type(torch.FloatTensor).cuda(gpu_id)
            multi_filter_response_map = torch.nn.functional.conv2d(input=query_feature_map.view(1, feature_length, height, width),
                                                                weight=feature_vector.view(valid_keypoint3d_num, feature_length, 1, 1), 
                                                                padding=0)
            max_response_columns, max_index_columns = torch.max(multi_filter_response_map, dim=2, keepdim=False)
            _, max_index_x = torch.max(max_response_columns, dim=2, keepdim=False)
            keypoint_indexes = list(range(0, valid_keypoint3d_num))
            max_index_y = max_index_columns.view(valid_keypoint3d_num, -1)[keypoint_indexes, max_index_x]
            max_index_x, max_index_y = max_index_x.cpu().numpy(), max_index_y.cpu().numpy()
            
            max_response = multi_filter_response_map.view(valid_keypoint3d_num, height, width)[keypoint_indexes, max_index_y, max_index_x].cpu().numpy()
            response_array[initial:final, 0] = max_response[0]
            response_array[initial:final, 1] = max_index_x[0]
            response_array[initial:final, 2] = max_index_y[0]            
    
    return response_array

def load_pretrained_model(feature_descriptor_model_path, filter_growth_rate, feature_length, gpu_id):
    feature_descriptor_model = models.FCDenseNetFeature(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length)

    feature_descriptor_model = torch.nn.DataParallel(feature_descriptor_model, device_ids=[gpu_id])
    feature_descriptor_model.eval()

    if feature_descriptor_model_path.exists():
        print("Loading {:s} ...".format(str(feature_descriptor_model_path)))
        state = torch.load(str(feature_descriptor_model_path), map_location='cuda:{}'.format(gpu_id))
        feature_descriptor_model.load_state_dict(state["model"])
    else:
        print("No pre-trained model detected")
        raise OSError
    del state

    return feature_descriptor_model

def load_hardnet_pretrained_model(feature_descriptor_model_path, gpu_id):
    feature_descriptor_model = models.HardNet()

    if os.path.exists(feature_descriptor_model_path):
        print("Loading {:s} ...".format(feature_descriptor_model_path))
        state = torch.load(feature_descriptor_model_path, map_location='cuda:{}'.format(gpu_id))
        feature_descriptor_model.load_state_dict(state["state_dict"])
        feature_descriptor_model.eval()
    else:
        print("No pre-trained model detected")
        raise OSError
    del state

    return feature_descriptor_model.cuda()

def downsample_and_crop_mask(mask, downsampling_factor, divide, suggested_h=None, suggested_w=None):
    downsampled_mask = cv2.resize(mask, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    end_h_index = downsampled_mask.shape[0]
    end_w_index = downsampled_mask.shape[1]
    # divide is related to the pooling times of the teacher model
    indexes = np.where(downsampled_mask >= 200)
    h = indexes[0].max() - indexes[0].min()
    w = indexes[1].max() - indexes[1].min()

    remainder_h = h % divide
    remainder_w = w % divide

    increment_h = divide - remainder_h
    increment_w = divide - remainder_w

    target_h = h + increment_h
    target_w = w + increment_w

    start_h = max(indexes[0].min() - increment_h // 2, 0)
    end_h = start_h + target_h

    start_w = max(indexes[1].min() - increment_w // 2, 0)
    end_w = start_w + target_w

    if suggested_h is not None:
        if suggested_h != h:
            remain_h = suggested_h - target_h
            start_h = max(start_h - remain_h // 2, 0)
            end_h = min(suggested_h + start_h, end_h_index)
            start_h = end_h - suggested_h

    if suggested_w is not None:
        if suggested_w != w:
            remain_w = suggested_w - target_w
            start_w = max(start_w - remain_w // 2, 0)
            end_w = min(suggested_w + start_w, end_w_index)
            start_w = end_w - suggested_w

    kernel = np.ones((5, 5), np.uint8)
    downsampled_mask_erode = cv2.erode(downsampled_mask, kernel, iterations=1)
    cropped_mask = downsampled_mask_erode[start_h:end_h, start_w:end_w]
    return cropped_mask, start_h, end_h, start_w, end_w

def find_common_valid_size(undistorted_mask_path, image_downsampling, network_downsampling):
    undistorted_mask_boundary = cv2.imread(undistorted_mask_path, cv2.IMREAD_GRAYSCALE)
    _, start_h, end_h, start_w, end_w = \
        downsample_and_crop_mask(undistorted_mask_boundary, downsampling_factor=image_downsampling,
                                 divide=network_downsampling)
    return end_h - start_h, end_w - start_w

def read_color_img(image_path, start_h, end_h, start_w, end_w, downsampling_factor):
    img = cv2.imread(str(image_path))
    downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
    downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
    downsampled_img = downsampled_img.astype(np.float32)
    return downsampled_img


def image_dataloader(args, model, query_image_path, undistorted_mask_boundary_path):
    
    largest_h, largest_w = \
        find_common_valid_size(undistorted_mask_path=undistorted_mask_boundary_path, 
                               image_downsampling=args.image_downsampling, 
                               network_downsampling=args.network_downsampling)

    undistorted_mask_boundary = cv2.imread(undistorted_mask_boundary_path, cv2.IMREAD_GRAYSCALE)

    cropped_downsampled_undistorted_mask_boundary, start_h, end_h, start_w, end_w = \
            downsample_and_crop_mask(mask=undistorted_mask_boundary,
                                     downsampling_factor=args.image_downsampling,
                                     divide=args.network_downsampling, 
                                     suggested_h=largest_h,
                                     suggested_w=largest_w)

    color_img = read_color_img(image_path=query_image_path, 
                               start_h=start_h,
                               end_h=end_h, 
                               start_w=start_w, 
                               end_w=end_w,
                               downsampling_factor=args.image_downsampling)

    height, width, _ = color_img.shape

    testing_mask_boundary = type_float_and_reshape(
                cropped_downsampled_undistorted_mask_boundary.astype(np.float32) / 255.0,
                (height, width, 1))
    testing_mask_boundary[testing_mask_boundary > 0.9] = 1.0
    testing_mask_boundary[testing_mask_boundary <= 0.9] = 0.0

    normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)
    testing_color_img = normalize(image=color_img)['image']
    
    img_tensor = img_to_tensor(testing_color_img)
    img_tensor = img_tensor[None, :]

    with torch.no_grad():
        feature_map = model(img_tensor)
        feature_map = feature_map[0] 

    return feature_map, start_h, start_w


def hardnet_image_dataloader(crop_positions_per_seq, mask_boundary_per_seq, normalize,
                             image_downsampling, descriptor, query_image_path, query_sequence_root):
    
    start_h, end_h, start_w, end_w = crop_positions_per_seq[query_sequence_root]

    color_img = utils.get_single_color_img(image_path=os.path.join(query_sequence_root, "images"),
                                           index=int(query_image_path.split(".")[0]),
                                           start_h=start_h,
                                           end_h=end_h, 
                                           start_w=start_w, 
                                           end_w=end_w,
                                           downsampling_factor=image_downsampling,
                                           is_hsv=False,
                                           rgb_mode="rgb")

    training_color_img_1 = color_img
    height, width, _ = training_color_img_1.shape

    training_mask_boundary = utils.type_float_and_reshape(
        mask_boundary_per_seq[query_sequence_root].astype(np.float32) / 255.0,
        (height, width, 1))
    training_mask_boundary[training_mask_boundary > 0.9] = 1.0
    training_mask_boundary[training_mask_boundary <= 0.9] = 0.0

    training_color_img_1 = normalize(image=training_color_img_1)['image']

    color_1 = img_to_tensor(training_color_img_1),
    boundary = img_to_tensor(training_mask_boundary)

    boundary = np.uint8(255 * boundary.reshape((height, width)))
    
    kps, des = descriptor.detectAndCompute(color_1, mask=boundary)
    
    hn_feature_map = torch.zeros(128, color_1.shape[0], color_1.shape[1], dtype=torch.float32) #Feature map C x H x W
    for point, point_desc in zip(kps, des):
        norm_point_desc = (point_desc / np.linalg.norm(point_desc)).astype(np.float32)
        x_coord, y_coord = np.round(point.pt[0]), np.round(point.pt[1])
        hn_feature_map[:, int(y_coord), int(x_coord)] = torch.from_numpy(norm_point_desc)
        
    return hn_feature_map, start_h, start_w


def sift_image_dataloader(args, sift, query_image_path, undistorted_mask_boundary_path):
    
    largest_h, largest_w = \
        find_common_valid_size(undistorted_mask_path=undistorted_mask_boundary_path, 
                               image_downsampling=args.image_downsampling, 
                               network_downsampling=args.network_downsampling)

    undistorted_mask_boundary = cv2.imread(undistorted_mask_boundary_path, cv2.IMREAD_GRAYSCALE)

    cropped_downsampled_undistorted_mask_boundary, start_h, end_h, start_w, end_w = \
            downsample_and_crop_mask(mask=undistorted_mask_boundary,
                                     downsampling_factor=args.image_downsampling,
                                     divide=args.network_downsampling, 
                                     suggested_h=largest_h,
                                     suggested_w=largest_w)

    color_img = read_color_img(image_path=query_image_path, 
                               start_h=start_h,
                               end_h=end_h, 
                               start_w=start_w, 
                               end_w=end_w,
                               downsampling_factor=args.image_downsampling)
    normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)
    testing_color_img = normalize(image=color_img)['image']
    color_1 = cv2.cvtColor(np.uint8(255 * (testing_color_img * 0.5 + 0.5)), cv2.COLOR_RGB2BGR)

    height, width, _ = color_img.shape

    testing_mask_boundary = type_float_and_reshape(
                cropped_downsampled_undistorted_mask_boundary.astype(np.float32) / 255.0,
                (height, width, 1))
    testing_mask_boundary[testing_mask_boundary > 0.9] = 1.0
    testing_mask_boundary[testing_mask_boundary <= 0.9] = 0.0
    
    kernel = np.ones((5, 5), np.uint8)
    boundary = cv2.erode(testing_mask_boundary, kernel, iterations=3)
    boundary = np.uint8(255 * boundary.reshape((height, width)))
    
    kps, des = sift.detectAndCompute(color_1, mask=boundary)
    
    sift_feature_map = torch.zeros(128, color_1.shape[0], color_1.shape[1], dtype=torch.float32) #Feature map C x H x W
    for point, point_desc in zip(kps, des):
        norm_point_desc = (point_desc / np.linalg.norm(point_desc)).astype(np.float32)
        x_coord, y_coord = np.round(point.pt[0]), np.round(point.pt[1])
        sift_feature_map[:, int(y_coord), int(x_coord)] = torch.from_numpy(norm_point_desc)
        
    return sift_feature_map, start_h, start_w


def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def transformation_matrix_from_params(transformation_params, use_scale):
    rotation_matrix = np.zeros((4,4))
    rotation_matrix[:3, :3] = transformation_params[0]
    rotation_matrix[3, 3] = 1.0

    translation_matrix = np.zeros((4,4))
    translation_matrix[:3, :3] = np.identity(3)
    translation_matrix[:3, 3] = (transformation_params[1] * -1).reshape((3))
    translation_matrix[3, 3] = 1.0

    if use_scale and len(transformation_params) == 3:
        scaling_factor = transformation_params[2]
    else:
        scaling_factor = 1

    scaling_matrix = np.zeros((4,4))
    scaling_matrix[:3, :3] = np.identity(3) * (1/scaling_factor)
    scaling_matrix[3, 3] = 1.0

    transformation_matrix = rotation_matrix @ scaling_matrix @ translation_matrix

    return transformation_matrix


def save_sift_descriptors(source_feature_locations, source_sift_descriptors, source_image_name, 
                          start_h, start_w, image_downsampling, save_db):
    
    unique_locations, unique_indexes = np.unique(source_feature_locations, axis=0, return_index=True)
    for loc, idx in zip(unique_locations, unique_indexes):
        location_descriptor = source_sift_descriptors[idx,:]
        img_x = (loc[0] + start_w) * image_downsampling
        img_y = (loc[1] + start_h) * image_downsampling
        img_name = os.path.basename(source_image_name[0])
        img_id = "{},{},{}".format(int(img_name.split(".")[0]), int(img_x), int(img_y))
        
        save_db.add_dense_descriptor_with_params(desc_id=img_id, 
                                                 coor_x=img_x, 
                                                 coor_y=img_y, 
                                                 descriptor=location_descriptor,
                                                 img_name=img_name)

def read_point_cloud(path):
    lists_3D_points = []
    plydata = PlyData.read(path)
    for n in range(plydata["vertex"].count):
        temp = list(plydata["vertex"][n])
        lists_3D_points.append([temp[0], temp[1], temp[2], 1.0])

    return lists_3D_points

def generate_shared_point_views_matrix(view_indexes_per_point):
    shared_point_views_matrix = np.transpose(view_indexes_per_point) @ view_indexes_per_point
    shared_point_views_matrix[shared_point_views_matrix > 0] = 1

    return shared_point_views_matrix