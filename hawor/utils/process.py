import torch
from lib.models.mano_wrapper import MANO
from hawor.utils.geometry import aa_to_rotmat
import numpy as np
import sys
import os

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

# ---------------------------------------------------------------------------
# Cached MANO singletons — avoids reloading from disk on every call
# ---------------------------------------------------------------------------
_mano_right_cpu = None
_mano_right_cuda = None
_mano_left_cpu = None
_mano_left_cuda = None

_faces_right = None   # precomputed right-hand faces (with extra faces)
_faces_left = None    # precomputed left-hand faces

_FACES_NEW = np.array([
    [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
    [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
    [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
    [120, 108, 78], [78, 108, 79],
])


def _get_mano_right(use_cuda=True):
    global _mano_right_cpu, _mano_right_cuda
    if use_cuda:
        if _mano_right_cuda is None:
            block_print()
            m = MANO(data_dir='_DATA/data/', model_path='_DATA/data/mano',
                     gender='neutral', num_hand_joints=15, create_body_pose=False)
            enable_print()
            _mano_right_cuda = m.cuda()
        return _mano_right_cuda
    else:
        if _mano_right_cpu is None:
            block_print()
            m = MANO(data_dir='_DATA/data/', model_path='_DATA/data/mano',
                     gender='neutral', num_hand_joints=15, create_body_pose=False)
            enable_print()
            _mano_right_cpu = m
        return _mano_right_cpu


def _get_mano_left(use_cuda=True, fix_shapedirs=True):
    global _mano_left_cpu, _mano_left_cuda
    if use_cuda:
        if _mano_left_cuda is None:
            block_print()
            m = MANO(data_dir='_DATA/data_left/', model_path='_DATA/data_left/mano_left',
                     gender='neutral', num_hand_joints=15, create_body_pose=False,
                     is_rhand=False)
            enable_print()
            if fix_shapedirs:
                m.shapedirs[:, 0, :] *= -1
            _mano_left_cuda = m.cuda()
        return _mano_left_cuda
    else:
        if _mano_left_cpu is None:
            block_print()
            m = MANO(data_dir='_DATA/data_left/', model_path='_DATA/data_left/mano_left',
                     gender='neutral', num_hand_joints=15, create_body_pose=False,
                     is_rhand=False)
            enable_print()
            if fix_shapedirs:
                m.shapedirs[:, 0, :] *= -1
            _mano_left_cpu = m
        return _mano_left_cpu


def _get_faces():
    global _faces_right, _faces_left
    if _faces_right is None:
        mano = _get_mano_right(use_cuda=False)
        _faces_right = np.concatenate([mano.faces, _FACES_NEW], axis=0)
        _faces_left = _faces_right[:, [0, 2, 1]]
    return _faces_right, _faces_left


def get_mano_faces():
    faces_right, _ = _get_faces()
    return faces_right


def run_mano(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True):
    """
    Forward pass of the MANO model.

    trans : B x T x 3
    root_orient : B x T x 3
    hand_pose : B x T x J*3
    betas : (optional) B x T x D
    """
    mano = _get_mano_right(use_cuda=use_cuda)

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    global_orient = aa_to_rotmat(root_orient.reshape(B*T, -1)).view(B*T, 1, 3, 3)
    hand_pose_rot = aa_to_rotmat(hand_pose.reshape(B*T*NUM_JOINTS, 3)).view(B*T, NUM_JOINTS, 3, 3)
    transl = trans.reshape(B*T, 3)
    betas_flat = betas.reshape(B*T, -1)

    params = {'global_orient': global_orient, 'hand_pose': hand_pose_rot,
              'betas': betas_flat, 'transl': transl}

    if use_cuda:
        mano_output = mano(**{k: v.float().cuda() for k, v in params.items()}, pose2rot=False)
    else:
        mano_output = mano(**{k: v.float() for k, v in params.items()}, pose2rot=False)

    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if is_right is not None:
        faces_right, faces_left = _get_faces()
        faces_n = len(faces_right)
        is_right_np = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_right_expanded = faces_right[np.newaxis, np.newaxis]
        faces_left_expanded = faces_left[np.newaxis, np.newaxis]
        faces_result = np.where(is_right_np[..., np.newaxis, np.newaxis],
                                faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))

    return outputs

def run_mano_left(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True, fix_shapedirs=True):
    """
    Forward pass of the left-hand MANO model.

    trans : B x T x 3
    root_orient : B x T x 3
    hand_pose : B x T x J*3
    betas : (optional) B x T x D
    """
    mano = _get_mano_left(use_cuda=use_cuda, fix_shapedirs=fix_shapedirs)

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    global_orient = aa_to_rotmat(root_orient.reshape(B*T, -1)).view(B*T, 1, 3, 3)
    hand_pose_rot = aa_to_rotmat(hand_pose.reshape(B*T*NUM_JOINTS, 3)).view(B*T, NUM_JOINTS, 3, 3)
    transl = trans.reshape(B*T, 3)
    betas_flat = betas.reshape(B*T, -1)

    params = {'global_orient': global_orient, 'hand_pose': hand_pose_rot,
              'betas': betas_flat, 'transl': transl}

    if use_cuda:
        mano_output = mano(**{k: v.float().cuda() for k, v in params.items()}, pose2rot=False)
    else:
        mano_output = mano(**{k: v.float() for k, v in params.items()}, pose2rot=False)

    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if is_right is not None:
        faces_right, faces_left = _get_faces()
        faces_n = len(faces_right)
        is_right_np = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_right_expanded = faces_right[np.newaxis, np.newaxis]
        faces_left_expanded = faces_left[np.newaxis, np.newaxis]
        faces_result = np.where(is_right_np[..., np.newaxis, np.newaxis],
                                faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))

    return outputs

def run_mano_twohands(init_trans, init_rot, init_hand_pose, is_right, init_betas, use_cuda=True, fix_shapedirs=True):
    outputs_left = run_mano_left(init_trans[0:1], init_rot[0:1], init_hand_pose[0:1], None, init_betas[0:1], use_cuda=use_cuda, fix_shapedirs=fix_shapedirs)
    outputs_right = run_mano(init_trans[1:2], init_rot[1:2], init_hand_pose[1:2], None, init_betas[1:2], use_cuda=use_cuda)
    outputs_two = {
        "vertices": torch.cat((outputs_left["vertices"], outputs_right["vertices"]), dim=0),
        "joints": torch.cat((outputs_left["joints"], outputs_right["joints"]), dim=0)
    }
    return outputs_two
