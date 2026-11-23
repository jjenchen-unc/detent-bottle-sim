import numpy as np


def request(obs, prompt, external_camera='agentview'):
    return {'observation/exterior_image_1_left': obs[external_camera],
            'observation/wrist_image_left': obs['left_wrist'],
            'observation/joint_position': obs['joint_pos'][:7].copy(),
            'observation/gripper_position': np.array([np.clip(1 - obs['gripper_qpos'][:2].sum() / .09, 0, 1)]),
            'prompt': prompt}
