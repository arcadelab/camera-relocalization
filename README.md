# Investigating Keypoint Descriptors for Camera Relocalization in Endoscopy Surgery

This repository contains the source code for the method described in the paper:

***Investigating Keypoint Descriptors for Camera Relocalization in Endoscopy Surgery***

Isabela Hernández*, Roger Soberanis-Mukul*, Jan E.
Mangulabnan, Manish Sahu, Jonas Winter, Swaroop
Vedula, Masaru Ishii, Gregory Hager, Russell H. Taylor and Mathias Unberath.

In **the 14th International Conference on Information Processing in
Computer-Assisted Interventions (IPCAI 2023)**

Please contact Isabela Hernández (iherna12@jhu.edu) or Roger Soberanis-Mukul (rsobera1@jhu.edu) if you have any questions.

We kindly ask you to cite [this paper](https://link.springer.com/article/10.1007/s11548-023-02918-x) if you build upon our work.

```
@Article{hernandez2023relocalization,
author={Hern{\'a}ndez, Isabela
and Soberanis-Mukul, Roger
and Mangulabnan, Jan Emily
and Sahu, Manish
and Winter, Jonas
and Vedula, Swaroop
and Ishii, Masaru
and Hager, Gregory
and Taylor, Russell H.
and Unberath, Mathias},
title={Investigating Keypoint Descriptors for Camera Relocalization in Endoscopy Surgery},
journal={International Journal of Computer Assisted Radiology and Surgery},
year={2023},
month={May},
day={09},
abstract={Recent advances in computer vision and machine learning have resulted in endoscopic video-based solutions for dense reconstruction of the anatomy. To effectively use these systems in surgical navigation, a reliable image-based technique is required to constantly track the endoscopic camera's position within the anatomy, despite frequent removal and re-insertion. In this work, we investigate the use of recent learning-based keypoint descriptors for six degree-of-freedom camera pose estimation in intraoperative endoscopic sequences and under changes in anatomy due to surgical resection.},
issn={1861-6429},
doi={10.1007/s11548-023-02918-x},
url={https://doi.org/10.1007/s11548-023-02918-x}
}
```

## Instructions

1. Install all necessary packages as established in ```requirements.txt```. Run the command:
```pip install -r requirements.txt```, and activate the conda environment named "localization".

The remaining set of instructions assumes you have a pre-trained feature descriptor, as described in this [work](https://ieeexplore.ieee.org/document/9157241). For detailed instructions on dense descriptor training, we refer you to this [repository](https://github.com/lppllppl920/DenseDescriptorLearning-Pytorch).

2. Run the file ```run_prerequisites.sh``` with the appropriate parameters and directory paths. This code will sequentially reproduce all the steps required for the camera relocalization pipeline (Figure 1a). See Section **Other descriptors** for further information on integrating alternative feature descriptors to our pipeline.

3. Run the file ```run_camera_relocalization.sh``` with the appropriate parameters and directory paths. 

    This script is set to perform the two main stages of our pipeline: *Keypoint Relocalization* and *Pose Estimation* (Figure 1b-c). The first stage uses intraoperative descriptor representations to find matches with point cloud descriptors and generates an intermediate database with query 2D-3D correspondences. The second stage retrieves these correspondence set and uses a standard PnP solver with RANSAC to perform initial pose estimation.  

    The ```mode``` parameter determines the stage(s) for camera relocalization to perform. Set ```--mode "localize"``` to perform *Keypoint Relocalization*, ```--mode "pnp"``` to perform *Pose Estimation*, or ```--mode "all"``` to perform both in a sequential manner. 

4. Run the file ```run_relocalization_evaluation.sh``` with the appropriate parameters and directory paths.

    This script is set to perform the last stages of our pipeline: *Trajectory Post-processing* and *Pose Evaluation* (Figure 1c). The former stage filters the initial pose estimates under three different strategies, to discard large deviations in this trajectory. The latter stage performs evaluation under the translation and rotation metrics defined in the manuscript (Section 2.5).

**Note:** All main files are left with exemplary parameters and directory paths as guide.

## Other descriptors

Our pipeline offers the possibility to employ alternative feature descriptors as means for *Keypoint Relocalization* (see Section 2.3 in manuscript for technical details). This involves computing a preoperative anatomy reconstruction using SfM in conjunction with dense descriptors, as a structural base to provide initial 2D-3D point correspondences. Once these locations are established, the original descriptor's projections will be replaced by the alternative's, and then employed to generate correspondences with the unseen intraoperative images.

To execute this replacement in the corresponding database files, run the following command with the appropriate parameters and model paths:
```
/path/to/python feature_population.py \
--image_downsampling 4.0 --network_downsampling 64 --sequence_root "/path/to/preop/data/specific/directory" \
--feature_trained_model_path "/path/to/dense/descriptor/model" \
--feature_length 256 --filter_growth_rate 8 --gpu_id 0 \
--matching_model_description "short description for base model" \
--feature_model_description "short description for feature-providing model"
```

Be aware that the camera relocalization pipeline should be slightly modified to accomodate the alternative descriptor's image loading strategy, to ensure correct keypoint relocalization.