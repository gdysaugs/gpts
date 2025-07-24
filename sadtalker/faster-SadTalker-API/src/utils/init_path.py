import os
import torch 
from pathlib import Path

def init_path(checkpoint_dir, config_dir, size=512, old_version=False, preprocess='crop'):
    """
    Initialize paths for SadTalker model checkpoints and configurations
    """
    sadtalker_paths = {
        'checkpoint_dir': checkpoint_dir,
        'config_dir': config_dir,
        'size': size,
        'old_version': old_version,
        'preprocess': preprocess
    }
    
    # Model checkpoint paths
    if old_version:
        sadtalker_paths['wav2lip_checkpoint'] = os.path.join(checkpoint_dir, 'wav2lip.pth')
        sadtalker_paths['audio2pose_checkpoint'] = os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth')
        sadtalker_paths['audio2exp_checkpoint'] = os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth')
        sadtalker_paths['free_view_checkpoint'] = os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar')
    else:
        sadtalker_paths['wav2lip_checkpoint'] = os.path.join(checkpoint_dir, 'wav2lip_gan.pth')
        sadtalker_paths['audio2pose_checkpoint'] = os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth')
        sadtalker_paths['audio2exp_checkpoint'] = os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth')
        sadtalker_paths['free_view_checkpoint'] = os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar')
    
    # Additional model paths
    sadtalker_paths['mapping_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00109-model.pth.tar')
    sadtalker_paths['facerender_yaml'] = os.path.join(config_dir, 'facerender.yaml')
    sadtalker_paths['mediapipe_model'] = os.path.join(checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
    sadtalker_paths['wingate_checkpoint'] = os.path.join(checkpoint_dir, 'epoch_20.pth')
    
    # Use different mapping checkpoint based on preprocess mode
    if preprocess == 'full':
        sadtalker_paths['mapping_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00229-model.pth.tar')
    
    # Face detection model
    sadtalker_paths['face_detector_checkpoint'] = os.path.join(checkpoint_dir, 's3fd-619a316812.pth')
    
    return sadtalker_paths