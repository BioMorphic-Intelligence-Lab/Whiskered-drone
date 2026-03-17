import torch
import hydra
import omni
import numpy as np

from enum import Enum, auto
from dataclasses import dataclass
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from utlis_cf import (
    control_drone,
    process_quaternion,
    transform_velocity,
    apply_control,
    perform_attitude_control,
    normalize_angle
)


class DroneMode(Enum):
    FORWARD = auto()
    CONTOUR_FOLLOWING = auto()
    BACKWARD = auto()
    ROTATING = auto()


@dataclass
class FSMContext:
    mode: DroneMode = DroneMode.FORWARD
    cf_steps: int = 0
    backward_steps: int = 0
    rotate_steps: int = 0
    before_yaw: torch.Tensor | None = None
    target_yaw: torch.Tensor | None = None


def read_lidar_with_noise(lidar_interface, lidar_path: str, mean=0.0, std=0.01) -> float:
    depth = lidar_interface.get_linear_depth_data("/World" + lidar_path)
    noise = np.random.normal(mean, std, np.asarray(depth).shape)
    return float(np.asarray(depth + noise).squeeze())


def should_trigger_contact(depth1: float, depth2: float, max_threshold: float) -> bool:
    return depth1 < max_threshold or depth2 < max_threshold


def enter_contour_following(ctx: FSMContext, sim_device):
    ctx.mode = DroneMode.CONTOUR_FOLLOWING
    ctx.cf_steps = 220
    ctx.backward_steps = 250
    ctx.rotate_steps = 200
    random_direction_rad = np.deg2rad(np.random.uniform(-180, 180))
    ctx.target_yaw = torch.tensor([random_direction_rad], device=sim_device)


def step_contour_following(ctx, drone, drone_state, depth1, depth2,
                           cf_vel_forward, cf_vel_backward, vel_side,
                           controller, yaw_left, yaw_right,
                           min_threshold, max_threshold):
    ctx.cf_steps = control_drone(
        drone, drone_state, depth1, depth2,
        cf_vel_forward, cf_vel_backward, vel_side,
        controller, yaw_left, yaw_right,
        min_threshold, max_threshold, ctx.cf_steps
    )
    if ctx.cf_steps <= 0:
        ctx.mode = DroneMode.BACKWARD


def step_backward(ctx, drone, drone_state, vel_backward, controller):
    r_transpose, _ = process_quaternion(drone_state)
    backward_world = transform_velocity(vel_backward, r_transpose)
    apply_control(drone, drone_state, controller, backward_world, "fly backward")
    ctx.backward_steps -= 1
    if ctx.backward_steps <= 0:
        _, ctx.before_yaw = process_quaternion(drone_state)
        ctx.mode = DroneMode.ROTATING


def step_rotating(ctx, drone, drone_state, controller):
    target_yaw = ctx.before_yaw + ctx.target_yaw
    _, current_yaw = process_quaternion(drone_state)
    perform_attitude_control(drone, drone_state, controller, target_yaw, "change orientation")
    ctx.rotate_steps -= 1

    if torch.abs(normalize_angle(current_yaw) - normalize_angle(target_yaw)) < 0.1:
        ctx.rotate_steps = 0

    if ctx.rotate_steps <= 0:
        ctx.mode = DroneMode.FORWARD


def step_forward(ctx, drone, drone_state, depth1, depth2,
                 vel_forward, vel_backward, vel_side,
                 controller, yaw_left, yaw_right,
                 min_threshold, max_threshold):
    _ = control_drone(
        drone, drone_state, depth1, depth2,
        vel_forward, vel_backward, vel_side,
        controller, yaw_left, yaw_right,
        min_threshold, max_threshold, 0
    )


@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import carb
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core import World
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots.drone import MultirotorBase
    from omni.isaac.range_sensor import _range_sensor
    from pxr import Gf

    carb.settings.get_settings().set("/app/show_developer_preference_section", True)

    sim = World(
        stage_units_in_meters=1.0,
        physics_dt=cfg.sim.dt,
        rendering_dt=cfg.sim.dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    scene_utils.design_scene()
    scene_utils.create_wall2(0)

    drone_cls = MultirotorBase.REGISTRY[cfg.drone_model]
    drone = drone_cls()

    translations = torch.zeros(1, 3, device=sim.device)
    translations[:, 2] = 1.0
    orientations = torch.zeros(1, 4, device=sim.device)
    drone.spawn(translations=translations, orientations=orientations)

    lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

    lidar_path1 = "/envs/env_0/Crazyflie_0/base_link/LidarSensor1"
    lidar_path2 = "/envs/env_0/Crazyflie_0/base_link/LidarSensor2"

    omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path=lidar_path1,
        parent="/World",
        min_range=0.05,
        max_range=0.4,
        draw_points=False,
        draw_lines=True,
        horizontal_fov=1,
        vertical_fov=1,
        horizontal_resolution=1,
        vertical_resolution=1,
        rotation_rate=0.0,
        high_lod=False,
        yaw_offset=0,
        enable_semantics=False,
    )

    omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path=lidar_path2,
        parent="/World",
        min_range=0.05,
        max_range=0.4,
        draw_points=False,
        draw_lines=True,
        horizontal_fov=1,
        vertical_fov=1,
        horizontal_resolution=1,
        vertical_resolution=1,
        rotation_rate=0.0,
        high_lod=False,
        yaw_offset=0,
        enable_semantics=False,
    )

    omni.kit.commands.execute(
        "TransformMultiPrimsSRTCpp",
        count=1,
        paths=["/World" + lidar_path1],
        new_translations=[0.0, 0.05, 0.05],
        new_rotation_eulers=[0.0, 0.0, 0.0],
        new_rotation_orders=[1, 0, 2],
        new_scales=[1.0, 1.0, 1.0],
        old_translations=[0.0, 0.0, 0.0],
        old_rotation_eulers=[0.0, 0.0, 0.0],
        old_rotation_orders=[1, 0, 2],
        old_scales=[1.0, 1.0, 1.0],
        time_code=0.0,
    )

    omni.kit.commands.execute(
        "TransformMultiPrimsSRTCpp",
        count=1,
        paths=["/World" + lidar_path2],
        new_translations=[0.0, -0.05, 0.05],
        new_rotation_eulers=[0.0, 0.0, 0.0],
        new_rotation_orders=[1, 0, 2],
        new_scales=[1.0, 1.0, 1.0],
        old_translations=[0.0, 0.0, 0.0],
        old_rotation_eulers=[0.0, 0.0, 0.0],
        old_rotation_orders=[1, 0, 2],
        old_scales=[1.0, 1.0, 1.0],
        time_code=0.0,
    )

    sim.reset()
    drone.initialize()

    controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)

    vel_forward = torch.tensor([0.2, 0.0, 0.0], device=sim.device)
    vel_side = torch.tensor([0.0, -0.2, 0.0], device=sim.device)
    vel_backward = torch.tensor([-0.2, 0.0, 0.0], device=sim.device)
    cf_vel_forward = torch.tensor([0.05, 0.0, 0.0], device=sim.device)
    cf_vel_backward = torch.tensor([-0.05, 0.0, 0.0], device=sim.device)

    yaw_right = torch.tensor([np.deg2rad(-25)], device=sim.device)
    yaw_left = torch.tensor([np.deg2rad(25)], device=sim.device)

    min_threshold = 0.1
    max_threshold = 0.15

    ctx = FSMContext()
    drone_state = drone.get_state()[..., :13].squeeze(0)

    from tqdm import tqdm
    for i in tqdm(range(30000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue

        depth1_noisy = read_lidar_with_noise(lidar_interface, lidar_path1, mean=0.0, std=0.01)
        depth2_noisy = read_lidar_with_noise(lidar_interface, lidar_path2, mean=0.0, std=0.01)

        if ctx.mode == DroneMode.FORWARD:
            if should_trigger_contact(depth1_noisy, depth2_noisy, max_threshold):
                enter_contour_following(ctx, sim.device)
                print("Enter CONTOUR_FOLLOWING")
            else:
                step_forward(
                    ctx, drone, drone_state, depth1_noisy, depth2_noisy,
                    vel_forward, vel_backward, vel_side,
                    controller, yaw_left, yaw_right,
                    min_threshold, max_threshold
                )
                print("FORWARD")

        elif ctx.mode == DroneMode.CONTOUR_FOLLOWING:
            step_contour_following(
                ctx, drone, drone_state, depth1_noisy, depth2_noisy,
                cf_vel_forward, cf_vel_backward, vel_side,
                controller, yaw_left, yaw_right,
                min_threshold, max_threshold
            )
            print("CONTOUR_FOLLOWING")

        elif ctx.mode == DroneMode.BACKWARD:
            step_backward(ctx, drone, drone_state, vel_backward, controller)
            print("BACKWARD")

        elif ctx.mode == DroneMode.ROTATING:
            step_rotating(ctx, drone, drone_state, controller)
            print("ROTATING")

        sim.step(render=(i % 5 == 0))
        drone_state = drone.get_state()[..., :13].squeeze(0)

    simulation_app.close()


if __name__ == "__main__":
    main()