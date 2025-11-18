import torch
import numpy as np
from scipy.spatial import ConvexHull

def yaw_to_quaternion(yaw):
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.tensor(yaw, dtype=torch.float32)

    half_yaw = yaw * 0.5
    w = torch.cos(half_yaw)
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = torch.sin(half_yaw)
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_multiply(q, r):
        w0, x0, y0, z0 = q.unbind(-1)
        w1, x1, y1, z1 = r.unbind(-1)   
        return torch.stack([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1
        ], dim=-1)

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    )
    matrix = matrix.unflatten(matrix.dim() - 1, (3, 3))
    return matrix

def quaternion_to_yaw(rot):
    w, x, y, z = rot[..., 0], rot[..., 1], rot[..., 2], rot[..., 3]
    
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    return yaw

def process_quaternion(drone_state, rot_z_45):
    _, rot, _, _ = torch.split(drone_state, [3, 4, 3, 3], dim=-1)
    yaw = quaternion_to_yaw(rot)
    rot = torch.nn.functional.normalize(rot, p=2, dim=-1)
    rot = quaternion_multiply(rot, rot_z_45)
    R = quaternion_to_rotation_matrix(rot)
    return R.transpose(-1, -2), yaw

def transform_velocity(velocity, R_transpose):
    if velocity.dim() == 1:
        velocity = velocity.unsqueeze(0)
    return torch.matmul(velocity, R_transpose).squeeze(0)

def apply_control(drone, drone_state, controller, target_vel, action_name):
    action = controller.compute(drone_state, target_vel=target_vel)
    drone.apply_action(action)

def perform_attitude_control(drone, drone_state, controller, yaw, action_name):
    action = controller.compute(drone_state, target_yaw=yaw)
    drone.apply_action(action)

def control_drone(drone, drone_state, depth1_noisy, depth2_noisy, vel_forward, vel_backward, vel_side, rot_z_45, controller, yaw_left, yaw_right, MIN_THRESHOLD, MAX_THRESHOLD, counter= 0):
    R_transpose, current_yaw = process_quaternion(drone_state, rot_z_45)
    
    if MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        CF_world = transform_velocity(vel_side, R_transpose)
        apply_control(drone, drone_state, controller, CF_world, "fly side way")
        counter -= 1
    elif MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and depth2_noisy < MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_right
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn right")
        counter -= 1
    elif MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and depth2_noisy > MAX_THRESHOLD:
        target_yaw = current_yaw + yaw_left
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn left")
        counter -= 1
    elif depth1_noisy < MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_left
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn left")
        counter -= 1
    elif depth1_noisy > MAX_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_right
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn right")
        counter -= 1
    elif depth1_noisy < MIN_THRESHOLD and depth2_noisy > MAX_THRESHOLD:
        target_yaw = current_yaw + yaw_left
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn left")
        counter -= 1
    elif depth1_noisy > MAX_THRESHOLD and depth2_noisy < MIN_THRESHOLD:
        target_yaw = current_yaw + yaw_right
        perform_attitude_control(drone, drone_state, controller, target_yaw, "turn right")
        counter -= 1
    elif depth1_noisy > MAX_THRESHOLD and depth2_noisy > MAX_THRESHOLD:
        forward_world = transform_velocity(vel_forward, R_transpose)
        apply_control(drone, drone_state, controller, forward_world, "fly forward")
    else:
        backward_world = transform_velocity(vel_backward, R_transpose)
        apply_control(drone, drone_state, controller, backward_world, "fly backward")
        counter -= 1
    
    return counter



def compute_angle(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle


def potential_function(x, significant_points, c=1.0):
    distances = np.linalg.norm(x[:, np.newaxis, :] - significant_points[np.newaxis, :, :], axis=2)
    P = -np.exp(-distances**2 / (2 * c**2))
    return np.min(P, axis=1)

def normalize_angle(rad):
    rad = (rad + np.pi) % (2 * np.pi) - np.pi
    return torch.rad2deg(rad)
