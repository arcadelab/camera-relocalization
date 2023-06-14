import os
import tqdm
import argparse
import numpy as np
from pathlib import Path

import general_utils
import database_utils as idb

"""
Couples extracted keypoints with dense descriptors from a trained descriptor model.
The keypoint list is provided by dense feature matching process.
"""

def main(args):

    TEMPLATE_PREOPERATIVE_DATABASE = os.path.join(args.sequence_root, "dreco_base_{}.db".format(args.matching_model_description))
    PREOPERATIVE_DATABASE = \
        TEMPLATE_PREOPERATIVE_DATABASE.replace("base_{}".format(args.matching_model_description),
                                               "base_{}_fill_{}".format(args.matching_model_description, args.feature_model_description))
    os.system("cp {} {}".format(str(TEMPLATE_PREOPERATIVE_DATABASE), str(PREOPERATIVE_DATABASE)))

    db = idb.ReconstrunctionDB(PREOPERATIVE_DATABASE)

    model = general_utils.load_pretrained_model(feature_descriptor_model_path=Path(args.feature_trained_model_path), 
                                                filter_growth_rate=args.filter_growth_rate, 
                                                feature_length=args.feature_length, 
                                                gpu_id=args.gpu_id)

    undistorted_mask_boundary_path = os.path.join(args.sequence_root, "undistorted_mask.bmp")

    preop_image_list = db.get_image_names()
    preop_image_list = [j[0] for j in preop_image_list]
    tq = tqdm.tqdm(total=len(preop_image_list))

    for preop_img in preop_image_list:
        preop_img_path = os.path.join(args.sequence_root, "images", preop_img)

        preop_feature_map, start_h, start_w = \
        general_utils.image_dataloader(args=args, model=model, query_image_path=preop_img_path,
                                       undistorted_mask_boundary_path=undistorted_mask_boundary_path)

        keypoint_coordinates = db.get_keypoint_locations(image_name=preop_img)
        try:
            downsampled_keypoint_coordinates = np.zeros((keypoint_coordinates.shape))
            downsampled_keypoint_coordinates[:, 0] = (keypoint_coordinates[:, 0] / args.image_downsampling) - start_w
            downsampled_keypoint_coordinates[:, 1] = (keypoint_coordinates[:, 1] / args.image_downsampling) - start_h

            for kpt, ds_kpt in zip(keypoint_coordinates, downsampled_keypoint_coordinates):
                kpt_descriptor = preop_feature_map[:, int(ds_kpt[1]), int(ds_kpt[0])]
                descriptor_id = "{},{},{}".format(int(preop_img.split(".")[0]), int(kpt[0]), int(kpt[1]))

                db.replace_dense_descriptor(desc_id=descriptor_id, img_name=preop_img,
                                            coor_x=kpt[0], coor_y=kpt[1],
                                            descriptor=kpt_descriptor.cpu().numpy())
        except:
            continue
        
        tq.update(1)
    tq.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense feature database filling",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--image_downsampling", type=float, default=4.0,
                        required=True, help="input image downsampling rate")
    parser.add_argument("--network_downsampling", type=int, default=64, required=True,
                        help="network bottom layer downsampling")
    parser.add_argument("--sequence_root", type=str, required=True,
                        help="root of the specific video sequence")
    parser.add_argument("--feature_trained_model_path", type=str, required=True, 
                        help="path to the trained model for feature filling")
    parser.add_argument("--feature_length", type=int, default=128, required=True,
                        help="output channel dimension of network")
    parser.add_argument("--filter_growth_rate", type=int, default=10, required=True,
                        help="filter growth rate of network")
    parser.add_argument("--gpu_id", type=int, default=0, required=True,
                        help="gpu id for matching generation")
    parser.add_argument('--matching_model_description', type=str, required=True, 
                        help='description of model providing base for matching')
    parser.add_argument('--feature_model_description', type=str, required=True, 
                        help='description of model providing features to fill')
    
    args = parser.parse_args()

    main(args=args)