import os
import csv
import math
import copy
import numpy as np
import open3d as o3d
from plyfile import PlyData
from  scipy import signal as sp
import pytransform3d.transformations as pt
from scipy.spatial.transform import Rotation as R

# Helper classes
class TemporalPoseBuffer:
    """
    Accumulates poses in temporal order: ... t-size//2, ..., t-1, t0, t+1, t+2, ..., t+size//2
    "t" will always be central, so the array must have odd size.
    """
    def __init__(self, size) -> None:
        self._size = size
        self._num_put_items = 0
        self._pose_array = []
        for i in range(size):
            identity = np.eye(4)
            self._pose_array.append(identity)
        self._pose_array = np.asarray(self._pose_array)
        
    def __len__(self):
        return self._size
    
    def __getittem__(self, key):
        return self._pose_array[key % self._size]
    
    def __setitem__(self, key, pose):
        self._pose_array[key % self._size] = pose

    def put_pose(self, pose):
        self._num_put_items += 1
        self._pose_array[:(self._size - 1)] = self._pose_array[1:self._size]
        self._pose_array[-1] = pose

    def get_buffer_array(self):
        return self._pose_array
    
    def is_full(self):
        return self._num_put_items >= self._size


class ScaleByNorm:
    def __init__(self) -> None:
        pass

    def find_scale_to_unit(self, trajectory):
        for i, point in enumerate(trajectory):
            if i == 0:
                max_bound = np.asarray(point[:3], dtype=np.float32)
                min_bound = np.asarray(point[:3], dtype=np.float32)
            else:
                temp = np.asarray(point[:3], dtype=np.float32)
                if np.any(np.isnan(temp)):
                    continue
                max_bound = np.maximum(max_bound, temp)
                min_bound = np.minimum(min_bound, temp)

        scale_to_unit = np.linalg.norm(max_bound - min_bound, ord=2)
        return scale_to_unit


    def find_scale(self, source_trajectory, target_trajectory):  
        # Returns source to unit, target to unit, source to target scale
        scale_s = self.find_scale_to_unit(source_trajectory)
        scale_t = self.find_scale_to_unit(target_trajectory)

        return 1. / scale_s, 1. / scale_t, scale_t / scale_s

# Registration class and methods
class PoseRegistration:
    """
    Class definition to perform registration and auxiliary processes.
    We perform registration between two sets of poses, under the following assumptions:
        - The poses represent the same camera trajectories, but exist in different coordinate frames and/or scales (e.g. between tracking system info and COLMAP).
        - The poses are similar. A general visual inspection is necessary to ensure that both pose sets correspond to the same camera trajectory.
        - The number of poses in both sets is the same.
    
    Intermediate methods will find the scales between pose sets, and register using a standard SDV solving strategy.
    """
    def __init__(self) -> None:
        self.scaler = ScaleByNorm()
        self.source_pose = None
        self.target_pose = None
        self.source_to_target_transform = None
        self.source_to_unit_scale = None
        self.target_to_unit_scale = None
        self.source_to_target_scale = None
    
    @staticmethod
    def trajectory_from_poses(poses, to_mm=False):
        trajectory = []
        for pose in poses:
            if to_mm:
                trajectory.append(pose[:3, 3]*1000)
            else:
                trajectory.append(pose[:3, 3])
        return np.asarray(trajectory)

    @staticmethod
    def scale_poses(poses, scale):
        scaled_poses_list = []
        poses_copy = copy.deepcopy(poses)
        for pose in poses_copy:
            pose[:3, 3]*= scale
            scaled_poses_list.append(pose)
        scaled_poses_list = np.asarray(scaled_poses_list)

        return scaled_poses_list

    @staticmethod
    def apply_transformation_to_pose(poses, transformation):
        transformed_pose = []
        for pose in poses:
            transformed_pose.append(transformation @ pose)
        transformed_pose = np.asarray(transformed_pose)
        return transformed_pose

    @staticmethod
    def register_rigid(source, target):
        """
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
            A: Nxm numpy array of corresponding points
            B: Nxm numpy array of corresponding points
        Returns:
            T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
            R: mxm rotation matrix
            t: mx1 translation vector
        """
        assert source.shape == target.shape

        # get number of dimensions
        m = source.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(source, axis=0)
        centroid_B = np.mean(target, axis=0)
        AA = source - centroid_A
        BB = target - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t
       
        return T, R, t

    @staticmethod
    def plot_and_save_trajectory(poses, axlen, subsampling_factor=1, save_name="Trajectory.ply", draw_connections=True, connection_color=[1, 0, 0], show=False):
        tm = []
        transformation_matrices = np.empty((len(poses), 4, 4))
        points = []
        for j, cam_pose in enumerate(poses):
            if j % subsampling_factor != 0:
                continue
            rot = cam_pose[0:3, 0:3]
            transl = cam_pose[:3, 3]
            points.append(transl)        
            tm.append(pt.transform_from(R=rot, p=transl))

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

    def register_poses(self, source_poses, target_poses, save_registration_fig=False, save_base_name="reg_", save_dir='.'):
        """
        Main registration function. Stores poses and computes transformation parameters.
        If target poses are expressed in meters, it is recommended to transform them previously to mm. 
        """
        self.source_pose = source_poses
        self.target_pose = target_poses
        self.source_to_unit_scale, self.target_to_unit_scale, self.source_to_target_scale = \
            self.scaler.find_scale(source_trajectory=self.source_pose, target_trajectory=self.target_pose)

        source_pose_in_target_scale = self.scale_poses(self.source_pose, self.source_to_target_scale)

        source_trajectory = self.trajectory_from_poses(source_pose_in_target_scale)
        target_trajecrory = self.trajectory_from_poses(self.target_pose)

        self.source_to_target_transform, _ , _ = self.register_rigid(source=source_trajectory, target=target_trajecrory)

        if save_registration_fig:
            source_pose_in_target_frame = self.transform_to_internal_target_frame(self.source_pose)
            self.plot_and_save_trajectory(source_pose_in_target_frame, 
                                     axlen=1, 
                                     subsampling_factor=1, 
                                     save_name=os.path.join(save_dir, save_base_name + "source.ply"))

            self.plot_and_save_trajectory(self.target_pose, 
                                     axlen=1, 
                                     subsampling_factor=1, 
                                     save_name=os.path.join(save_dir, save_base_name + "target.ply")) 


    def transform_to_internal_target_frame(self, poses):
        source_pose_in_target_scale = self.scale_poses(poses, self.source_to_target_scale)
        return self.apply_transformation_to_pose(poses=source_pose_in_target_scale,
                                                 transformation=self.source_to_target_transform)

    def get_registration(self):
        return self.source_to_target_scale, self.source_to_target_transform

    def get_scales(self):
        return self.source_to_unit_scale, self.target_to_unit_scale, self.source_to_target_scale
 
# Evaluation classes and methods
class PoseAnalysis:
    def __init__(self, error_funct, metrics_names=None, verbose=False):
        """
        Class definition to perform evaluation for a pair of pose sets.
        error_funct is a callable object or method that computes the errors (see RelocErrorMetric).
        """ 
        self._camera_names = None
        self._gt_poses = None
        self._predicted_poses = None
        self._num_netrics = None
        self._error_funct = error_funct
        self._metric_names = metrics_names        
        self._metrics = None
        self.wokring_dir = "."
        self._num_poses = 0
        self.verbose = verbose
    
    def __len__(self):
        return self._num_poses

    def _print_named_errors(self, error_list):
        for i in range(len(error_list)):
            print("\t {} = {}".format(self._metric_names[i], error_list[i]))

    def _print_errors(self, error_list):
        for i in range(len(error_list)):
            print("\t {}".format(error_list[i]))

    def _evaluate_poses(self):
        self._metrics = []
        for i in range(self._num_poses):
            gt_pose = self._gt_poses[i]
            estimated_pose = self._predicted_poses[i]
            metrics = self._error_funct(estimated_pose=estimated_pose, gt_pose=gt_pose)
            metric_list = []
            for met in metrics:
                metric_list.append(met)
            self._metrics.append(metric_list)
        self._metrics = np.asarray(self._metrics)
        self._num_netrics = self._metrics.shape[1]

    def set_poses(self, camera_names, gt_poses, predicted_poses):
        self._num_poses = len(camera_names)
        self._camera_names = camera_names
        self._gt_poses = gt_poses
        self._predicted_poses = predicted_poses
        self._evaluate_poses()

    def save_poses(self, save_dir=None):
        if save_dir is None:
            save_dir = self.wokring_dir
        np.savetxt(os.path.join(save_dir, "cam_names.txt"), np.asanyarray(self._camera_names), fmt='%s')
        np.save(os.path.join(save_dir, "gt_poses"), self._gt_poses)
        np.save(os.path.join(save_dir, "predicted_poses"), self._predicted_poses)

    def load_poses(self, load_dir=None):
        if load_dir is None:
            load_dir = self.wokring_dir
        self._camera_names = list(np.loadtxt(os.path.join(load_dir, "cam_names.txt"), dtype=str))
        self._gt_poses = np.load(os.path.join(load_dir, "gt_poses.npy"))
        self._predicted_poses = np.load(os.path.join(load_dir, "predicted_poses.npy"))
        self._num_poses = len(self._camera_names)
        self._evaluate_poses()

    def save_errors(self, save_name="errors.csv", save_dir=None):
        if save_dir is None:
            save_dir = self.wokring_dir
        with open(os.path.join(save_dir, save_name), 'w') as f:
            csv_writer = csv.writer(f)
            if self._metric_names is not None:
                csv_writer.writerow(['name'] + self._metric_names)
            for i in range(self._num_poses):
                row = [self._camera_names[i]] + list(self._metrics[i])
                csv_writer.writerow(row)            
     
    def get_gt_pose(self, camera_name):
        pose_idx = self._camera_names.index(camera_name)
        pose = self._gt_poses[pose_idx]
        if self.verbose:
            print(pose)
        return pose

    def get_predicted_pose(self, camera_name):
        pose_idx = self._camera_names.index(camera_name)
        pose = self._predicted_poses[pose_idx]
        if self.verbose:
            print(pose)
        return pose

    def get_errors_for_pose(self, camera_name):
        pose_idx = self._camera_names.index(camera_name)
        error = self._metrics[pose_idx]
        if self.verbose:
            if self._metric_names is not None:
                self._print_named_errors(error)
            else:
                self._print_errors(error)
        return error
    
    def save_trajectories_visual(self, save_dir=None):
        if save_dir is None:
            save_dir = self.wokring_dir
        PoseRegistration.plot_and_save_trajectory(self._predicted_poses,  
                                                  axlen=1, 
                                                  subsampling_factor=1, 
                                                  save_name=os.path.join(save_dir, "predicted.ply"))

        PoseRegistration.plot_and_save_trajectory(self._gt_poses,  
                                                  axlen=1, 
                                                  subsampling_factor=1, 
                                                  save_name=os.path.join(save_dir, "reference.ply"))

    def save_pose_visual(self, camera_name, save_dir=None):
        if save_dir is None:
            save_dir = self.wokring_dir
        pose_idx = self._camera_names.index(camera_name)
        pose = self._predicted_poses[pose_idx]
        PoseRegistration.plot_and_save_trajectory([pose],  
                                                  axlen=1, 
                                                  subsampling_factor=1, 
                                                  save_name=os.path.join(save_dir, "{}_predicted.ply".format(camera_name)))        
        pose = self._gt_poses[pose_idx]
        PoseRegistration.plot_and_save_trajectory([pose],  
                                                  axlen=1, 
                                                  subsampling_factor=1, 
                                                  save_name=os.path.join(save_dir, "{}_reference.ply".format(camera_name)))


    def save_centered_pose_visual(self, camera_name, save_dir=None):
        if save_dir is None:
            save_dir = self.wokring_dir
        pose_idx = self._camera_names.index(camera_name)
        pose = copy.deepcopy(self._predicted_poses[pose_idx])
        pose[:3, 3] = [0, 0, 0]
        PoseRegistration.plot_and_save_trajectory([pose],  
                                                  axlen=1, 
                                                  subsampling_factor=1, 
                                                  save_name=os.path.join(save_dir, "{}_predicted_centered.ply".format(camera_name)))        
        pose = copy.deepcopy(self._gt_poses[pose_idx])
        pose[:3, 3] = [0, 0, 0]
        PoseRegistration.plot_and_save_trajectory([pose],  
                                                  axlen=1, 
                                                  subsampling_factor=1, 
                                                  save_name=os.path.join(save_dir, "{}_reference_centered.ply".format(camera_name)))

class RelocErrorMetric:
    """
    Class definition to encapsulate proposed error metrics.
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)

        return n < 1e-6
    
    @staticmethod
    def rotationMatrixToEulerAngles(R):
        assert RelocErrorMetric.isRotationMatrix(R)

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])

        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    @staticmethod
    def rot_to_euler(Rot):
        r = R.from_matrix(Rot)
        euler_angles = r.as_euler("xyz", degrees=False)

        return euler_angles

    @staticmethod
    def compute_errors(estimated_rot, estimated_trans, gt_rot, gt_trans):
        """
        Calculates error measure between estimated and GT rotation matrices and translation vectors.
        """ 
        dif_conponents = gt_trans - estimated_trans
        x_diff = dif_conponents[0]
        y_diff = dif_conponents[1]
        z_diff = dif_conponents[2]
        diff_trans = np.sqrt(((gt_trans - estimated_trans) ** 2).sum(axis=None))
        rot_res = estimated_rot.T @ gt_rot
        diff_rot = np.sqrt(((rot_res - np.identity(gt_rot.shape[0])) ** 2).sum(axis=None))
        diff_rad = np.arccos(0.5*(np.trace(rot_res) - 1))
        euler_est = RelocErrorMetric.rot_to_euler(Rot=estimated_rot)
        euler_gt = RelocErrorMetric.rot_to_euler(Rot=gt_rot)        
        euler_res = RelocErrorMetric.rot_to_euler(Rot=rot_res)

        return np.abs(x_diff), np.abs(y_diff), np.abs(z_diff), diff_trans, diff_rot, diff_rad, euler_est, euler_gt, euler_res

    def __call__(self, estimated_pose, gt_pose):
        est_R = estimated_pose[:3, 0:3]
        est_t = estimated_pose[:3, 3]
        gt_R = gt_pose[:3, 0:3]
        gt_t = gt_pose[:3, 3]

        return RelocErrorMetric.compute_errors(estimated_rot=est_R, estimated_trans=est_t, 
                                               gt_rot=gt_R, gt_trans=gt_t)

# Filtering classes and methods
class MedianPoseFilter:
    def __init__(self, size=5) -> None:
        self.size=size

    def apply(self, pose_sequence):    
        xs = []
        ys = []
        zs = []
        for pose in pose_sequence:
            xs.append(pose[0, 3])
            ys.append(pose[1, 3])
            zs.append(pose[2, 3])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)

        idx = self.size // 2 
        ret_pose = copy.deepcopy(pose_sequence[idx])
        ret_pose[:3, 3] = [sp.medfilt(xs, self.size)[idx], sp.medfilt(ys, self.size)[idx], sp.medfilt(zs, self.size)[idx]]

        return ret_pose
        
class PointCloudPoseFilter:
    def __init__(self, point_cloud_path) -> None:
        point_cloud = self.read_point_cloud(point_cloud_path)
        for i, point in enumerate(point_cloud):
            if i == 0:
                max_bound = np.asarray(point[:3], dtype=np.float32)
                min_bound = np.asarray(point[:3], dtype=np.float32)
            else:
                temp = np.asarray(point[:3], dtype=np.float32)
                if np.any(np.isnan(temp)):
                    continue
                max_bound = np.maximum(max_bound, temp)
                min_bound = np.minimum(min_bound, temp)

        self.min_bound, self.max_bound = min_bound, max_bound

    @staticmethod
    def read_point_cloud(path):
        lists_3D_points = []
        plydata = PlyData.read(path)
        for n in range(plydata["vertex"].count):
            temp = list(plydata["vertex"][n])
            lists_3D_points.append([temp[0], temp[1], temp[2], 1.0])

        return lists_3D_points

    def apply(self, pose):
        t = pose[:3, 3]
        lower_bound_check = t[0] > self.min_bound[0] and t[1] > self.min_bound[1] and t[2] > self.min_bound[2]
        upper_bound_check = t[0] < self.max_bound[0] and t[1] < self.max_bound[1] and t[2] < self.max_bound[2]

        return lower_bound_check and upper_bound_check

class MeshPoseFilter:
    def __init__(self, sampled_mesh_path) -> None:        
        point_cloud = PointCloudPoseFilter.read_point_cloud(sampled_mesh_path)
        for i, point in enumerate(point_cloud):
            if i == 0:
                max_bound = np.asarray(point[:3], dtype=np.float32)
                min_bound = np.asarray(point[:3], dtype=np.float32)
            else:
                temp = np.asarray(point[:3], dtype=np.float32)
                if np.any(np.isnan(temp)):
                    continue
                max_bound = np.maximum(max_bound, temp)
                min_bound = np.minimum(min_bound, temp)

        self.min_bound, self.max_bound = min_bound, max_bound
    
    def apply(self, pose, scale_to_unit):
        t = pose[:3, 3] * scale_to_unit

        lower_bound_check = t[0] > self.min_bound[0] and t[1] > self.min_bound[1] and t[2] > self.min_bound[2]
        upper_bound_check = t[0] < self.max_bound[0] and t[1] < self.max_bound[1] and t[2] < self.max_bound[2]

        return lower_bound_check and upper_bound_check

class DistanceBasedFilter:
    def __init__(self, preop_poses, max_distance=None, max_rotation=0.78) -> None:
        self.preop_poses = preop_poses
        self.preop_trajectories = PoseRegistration.trajectory_from_poses(poses=preop_poses)        
        self.max_rotation = max_rotation
        if max_distance is None:
            self.max_distance = 2*self.compute_average_cam_distance()
        else:
            self.max_distance = max_distance

    def compute_average_cam_distance(self):
        num_poses = len(self.preop_trajectories)
        dist = 0
        for i in range(num_poses - 1):
            dist += np.linalg.norm(self.preop_trajectories[i] - self.preop_trajectories[i + 1])
        return dist / (num_poses - 1)

    def apply(self, pose):
        t = pose[:3, 3]
        rot_mat = pose[:3, :3]
        distances = np.linalg.norm(self.preop_trajectories - t, axis=1)
        min_distance = np.argmin(distances)
        diff_distance = distances[min_distance]
        ref_rotation = self.preop_poses[min_distance][:3, :3]
        eval_rotation = ref_rotation.T @ rot_mat
        diff_rad = np.arccos(0.5*(np.trace(eval_rotation) - 1))
        return diff_distance < self.max_distance and diff_rad < self.max_rotation

# Auxiliary methods
def mesh_to_point_cloud(mesh_path, save_dir, save_name):
    mesh1 = o3d.io.read_triangle_mesh(mesh_path)
    pointcloud1 = mesh1.sample_points_poisson_disk(1000)
    o3d.io.write_point_cloud(os.path.join(save_dir, save_name + ".ply"), pointcloud1)

def compute_pc_scale(extrinsics, point_cloud_path):
    point_cloud = PointCloudPoseFilter.read_point_cloud(point_cloud_path)
    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)

    for i, extrinsic in enumerate(extrinsics):
        if i == 0:
            max_bound = extrinsic[:3, 3]
            min_bound = extrinsic[:3, 3]
        else:
            temp = extrinsic[:3, 3]
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_1 = np.linalg.norm(max_bound - min_bound, ord=2)

    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)
    for i, point in enumerate(point_cloud):
        if i == 0:
            max_bound = np.asarray(point[:3], dtype=np.float32)
            min_bound = np.asarray(point[:3], dtype=np.float32)
        else:
            temp = np.asarray(point[:3], dtype=np.float32)
            if np.any(np.isnan(temp)):
                continue
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_2 = np.linalg.norm(max_bound - min_bound, ord=2)
    ret_scale = max(norm_1, norm_2)
    
    return 1 / ret_scale