import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import glob
import imageio
import cv2
# from animatediff.data.dinov2matcher import Dinov2Matcher

# a -> [ax]
# [0, -a3, a2]
# [a3, 0, -a1]
# [-a2, a1, 0]
# input: a vector (shape: 3) or vector array (shape: N x 3)
# output: a matrix  (shape: 3 x 3) of array of matrix (shape: N x 3 x 3)
def calc_cross_product_matrix(vec):
    is_array = False if len(vec.shape) == 1 else True
    if not is_array:
        vec = vec[np.newaxis, :] # 1 x 3
    ret_mat = np.zeros(list(vec.shape)+[3])
    ret_mat[:, 0, 1] = -vec[:, 2]
    ret_mat[:, 0, 2] = vec[:, 1]
    ret_mat[:, 1, 2] = -vec[:, 0]
    ret_mat -= ret_mat.transpose((0, 2, 1))
    if not is_array:
        ret_mat = ret_mat[0]
    return ret_mat

# T_mat shape is 4 x 4
# x2 = T_mat * x1 = R_mat*x1 + t = R_mat * ( x1 - (-R_mat^T * t) )
# let R_ess = R_mat, t_ess = -R_mat^T*t
# then E_mat = R_ess*[t_ess x]
def calc_essential_matrix(T_mat): 
    R_mat = T_mat[:3, :3]
    t = T_mat[:3, 3] # t 
    t_ess = -np.matmul(R_mat.transpose(), t)
    E_mat = np.matmul(R_mat, calc_cross_product_matrix(t_ess))
    return E_mat

# T_mat: from camera 1 to camera 2
# x2 = T_mat * x1 
# because in essential matrix we have x2^t E x1 = 0, 
# and x_{1,2} = K_{1,2}^-1 * coord_{1,2}, 
# we can get F = K2^-T * E * K1^-1
def calc_fundamental_matrix(T_mat, K_mat1, K_mat2):
    E_mat = calc_essential_matrix(T_mat)
    
    K2_invT = np.linalg.inv(K_mat2).transpose() 
    K1_inv = np.linalg.inv(K_mat1)
    F_mat = np.matmul(np.matmul(K2_invT, E_mat), K1_inv)

    return F_mat

# Assume cx=H/2, cy=W/2
def K_mat_from_fov(fov_deg, H, W):
    fx = (W/2) / math.tan(fov_deg/2) 
    fy = (H/2) / math.tan(fov_deg/2) 
    K_mat = np.array(
        [
            [fx, 0, W/2],
            [0, fy, H/2],
            [0, 0, 1]
        ]
    )
    return K_mat

class Camera(object):
    def __init__(self, entry):
        self.cid = entry[0]
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

class ValRealEstate10KPoseFolded(Dataset):
    def __init__(
            self,
            sample_n_frames=16,
            relative_pose=False,
            sample_size=256,
            validation_prompts=None,
            validation_negative_prompts=None,
            mode="train",
            pose_file_0=None,
            pose_file_1=None
    ):
        self.relative_pose = relative_pose
        self.sample_n_frames = sample_n_frames
        self.validation_prompts = validation_prompts
        self.validation_negative_prompts = validation_negative_prompts
        self.mode = mode
        self.pose_file_0 = pose_file_0
        self.pose_file_1 = pose_file_1

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size

        pixel_transforms = transforms.Compose([transforms.Resize(sample_size[0]),
                            transforms.CenterCrop(self.sample_size),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

        self.pixel_transforms = pixel_transforms
    
    def get_relative_pose(self, c2w_list, tar_idx=0):
        abs2rel = np.linalg.inv(c2w_list[tar_idx])
        ret_poses = [abs2rel @ c2w for c2w in c2w_list]
        return np.array(ret_poses, dtype=np.float32)

    def load_cameras_specific(self):
        # To extract two camera poses from the same start, here we first load two trajectory files, 
        # then normalize the poses to the same start (identical matrix), and finally calculate the relative poses

        pose_file_0 = os.path.join(self.pose_file_0)
        with open(pose_file_0, 'r') as f:
            poses_0 = f.readlines()
        pose_file_1 = os.path.join(self.pose_file_1)
        with open(pose_file_1, 'r') as f:
            poses_1 = f.readlines()
        poses_0 = [pose.strip().split(' ') for pose in poses_0[1:]]
        cam_params_0 = [[float(x) for x in pose] for pose in poses_0]
        cam_params_0 = [Camera(cam_param) for cam_param in cam_params_0]
        poses_1 = [pose.strip().split(' ') for pose in poses_1[1:]]
        cam_params_1 = [[float(x) for x in pose] for pose in poses_1]
        cam_params_1 = [Camera(cam_param) for cam_param in cam_params_1]
        cam_params_1.reverse()

        c2w_pose_list_0 = []
        K_mat_list_0 = []
        intrinsic_list_0 = []
        for frame_idx in range(len(cam_params_0)):
            cam = cam_params_0[frame_idx]
            H, W = 1280, 720

            crop_size = min(H, W)
            rescale = self.sample_size[0] / crop_size 
            dH, dW = (H-crop_size)/2, (W-crop_size)/2
            K_mat = np.array([[W*rescale*cam.fx, 0, (W*cam.cx-dW)*rescale], [0, H*rescale*cam.fy, (H*cam.cy-dH)*rescale], [0, 0, 1]])
            intrinsics = [K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]]
            # While the statement in realestate10K states that the extrinsics are w2c,
            # Seems they are c2w instead
            c2w_pose_list_0.append(cam.c2w_mat)
            K_mat_list_0.append(K_mat)
            intrinsic_list_0.append(intrinsics)

        c2w_pose_list_1 = []
        K_mat_list_1 = []
        intrinsic_list_1 = []
        for frame_idx in range(len(cam_params_1)):
            cam = cam_params_1[frame_idx]
            H, W = 1280, 720

            crop_size = min(H, W)
            rescale = self.sample_size[0] / crop_size 
            dH, dW = (H-crop_size)/2, (W-crop_size)/2
            K_mat = np.array([[W*rescale*cam.fx, 0, (W*cam.cx-dW)*rescale], [0, H*rescale*cam.fy, (H*cam.cy-dH)*rescale], [0, 0, 1]])
            intrinsics = [K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]]
            # While the statement in realestate10K states that the extrinsics are w2c,
            # Seems they are c2w instead
            c2w_pose_list_1.append(cam.c2w_mat)
            K_mat_list_1.append(K_mat)
            intrinsic_list_1.append(intrinsics)

        c2w_pose_list_0 = self.get_relative_pose(c2w_pose_list_0, tar_idx = 0)
        c2w_pose_list_1 = self.get_relative_pose(c2w_pose_list_1, tar_idx = 0)
        c2w_pose_list = np.concatenate([c2w_pose_list_0[1:][::-1], c2w_pose_list_1], axis=0)
        # force k mat to be the same
        K_mat_list = np.concatenate([K_mat_list_0[1:][::-1], K_mat_list_0], axis=0)
        intrinsic_list = np.concatenate([intrinsic_list_0[1:][::-1], intrinsic_list_1], axis=0)
        return c2w_pose_list, K_mat_list, intrinsic_list

    def get_batch(self, validation_idx):

        validation_prompt = self.validation_prompts[validation_idx]

        if self.validation_negative_prompts is not None:
            validation_negative_prompt = self.validation_negative_prompts[validation_idx]
        else:
            validation_negative_prompt = None

        c2w_pose_list, K_mat_list, intrinsic_list = self.load_cameras_specific()

        intrinsics = torch.as_tensor(np.array(intrinsic_list)).float()[None]   # [1, n_frame, 4]
        c2w = torch.as_tensor(c2w_pose_list)[None] # [1, n_frame, 4, 4]

        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu'
                                          )[0].permute(0, 3, 1, 2) # n_frame, channel, H, W
        
        # Folding camera poses
        F_mat_list = []
        for i in range(self.sample_n_frames):
            sid = self.sample_n_frames - 1 - i
            tid = self.sample_n_frames - 1 + i
            s2t = np.linalg.inv(c2w_pose_list[tid]) @ c2w_pose_list[sid]
            F_mat = calc_fundamental_matrix(s2t, K_mat_list[sid], K_mat_list[tid])
            F_mat_list.append(torch.from_numpy(F_mat))

        F_mats = torch.as_tensor(np.array(F_mat_list)).float() # [n_frame, 3, 3]
        
        # Fold all vectors
        F_mats = torch.cat([F_mats, F_mats.permute(0, 2, 1)], dim=0).contiguous()
        fold_indices = torch.arange(self.sample_n_frames)
        fold_indices = torch.cat([self.sample_n_frames - 1 - fold_indices, 
                                  self.sample_n_frames - 1 + fold_indices])
        
        plucker_embedding = plucker_embedding[fold_indices].contiguous()

        ret_c2w = c2w[:, fold_indices]
        ret_K_mats = np.stack(K_mat_list, axis=0)[fold_indices]

        return plucker_embedding, F_mats, validation_prompt, validation_negative_prompt, ret_c2w, ret_K_mats

    def __len__(self):
        return len(self.validation_prompts)

    def __getitem__(self, idx):
        plucker_embedding, F_mats, validation_prompt, validation_negative_prompt, ret_c2w, ret_K_mats = self.get_batch(idx)

        ret_sample = {
            "validation_prompt": validation_prompt,
            "plucker_embedding": plucker_embedding,
            "F_mats": F_mats, 
            "ret_c2w": ret_c2w,
            "ret_K_mats": ret_K_mats
        }
        if validation_negative_prompt is not None:
            ret_sample["validation_negative_prompt"] = validation_negative_prompt

        return ret_sample
