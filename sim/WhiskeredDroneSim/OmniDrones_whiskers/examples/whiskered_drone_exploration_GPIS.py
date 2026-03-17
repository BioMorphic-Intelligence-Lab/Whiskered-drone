import torch
import pandas as pd
import hydra
import omni
import numpy as np

from enum import Enum, auto
from dataclasses import dataclass, field
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from utlis_cf import *
from GPIS_cf import GPISModel
from datetime import datetime
from pathlib import Path

def build_log_filename(angle_deg: float, status: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    angle_str = str(int(angle_deg)) if float(angle_deg).is_integer() else str(angle_deg).replace(".", "p")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return str(log_dir / f"whisker_GPIS_{angle_str}_{status}_{timestamp}.csv")

class DroneMode(Enum):
    FORWARD = auto()
    CONTOUR_FOLLOWING = auto()
    BACKWARD = auto()
    ROTATING = auto()
    GOAL_SEEKING = auto()


@dataclass
class GPISLog:
    state_xs: list = field(default_factory=list)
    state_ys: list = field(default_factory=list)
    state_yaws: list = field(default_factory=list)
    state_lasers1: list = field(default_factory=list)
    state_lasers2: list = field(default_factory=list)
    laser_values1: list = field(default_factory=list)
    laser_values2: list = field(default_factory=list)


@dataclass
class FSMContext:
    mode: DroneMode = DroneMode.FORWARD

    cf_steps: int = 0
    backward_steps: int = 0
    rotate_steps: int = 0
    goal_steps: int = 0

    before_yaw: torch.Tensor | None = None
    target_yaw: torch.Tensor | None = None
    random_yaw: torch.Tensor | None = None

    depth_now: float = 0.5
    depth_last: float = 0.5

    laser_value1: int = 2
    laser_value2: int = 2

    direction_changes_completed: int = 0
    finish_cf: bool = False

    state_x: float = 0.0
    state_y: float = 0.0
    state_yaw: float = 0.0




def read_lidar_with_noise(lidar_interface, lidar_path: str, mean=0.0, std=0.01) -> float:
    depth = lidar_interface.get_linear_depth_data("/World" + lidar_path)
    depth = np.asarray(depth).squeeze()
    noise = np.random.normal(mean, std, np.asarray(depth).shape)
    return float(np.asarray(depth + noise).squeeze())


def save_log_to_csv(log: GPISLog, filename: str):
    data = {
        "state_xs": log.state_xs,
        "state_ys": log.state_ys,
        "state_yaws": log.state_yaws,
        "state_lasers1": log.state_lasers1,
        "state_lasers2": log.state_lasers2,
        "laser_values1": log.laser_values1,
        "laser_values2": log.laser_values2,
    }
    pd.DataFrame(data).to_csv(filename, index=False)


def update_log(log: GPISLog, ctx: FSMContext, depth1: float, depth2: float):
    log.state_xs.append(ctx.state_x)
    log.state_ys.append(ctx.state_y)
    log.state_yaws.append(ctx.state_yaw)
    log.state_lasers1.append(depth1)
    log.state_lasers2.append(depth2)
    log.laser_values1.append(ctx.laser_value1)
    log.laser_values2.append(ctx.laser_value2)


def reset_laser_labels(ctx: FSMContext):
    ctx.laser_value1 = 2
    ctx.laser_value2 = 2


def enter_contour_following(ctx: FSMContext, sim_device):
    ctx.mode = DroneMode.CONTOUR_FOLLOWING
    ctx.cf_steps = 250
    ctx.backward_steps = 250
    ctx.rotate_steps = 300
    ctx.finish_cf = True
    ctx.random_yaw = torch.tensor([np.deg2rad(-90)], device=sim_device)
    print("Touch detected")


def run_gpis_and_set_next_heading(ctx: FSMContext, log: GPISLog, sim_device):
    gpis = GPISModel(
        log.state_xs,
        log.state_ys,
        log.state_yaws,
        log.state_lasers1,
        log.state_lasers2,
        log.laser_values1,
        log.laser_values2,
        curvature_threshold=-0.8,
    )
    gpis.sample_data()
    gpis.train_model()
    gpis.predict()

    next_point = gpis.find_max_uncertainty_point()
    target_yaw = np.arctan2(next_point[1] - ctx.state_y, next_point[0] - ctx.state_x)
    ctx.target_yaw = torch.tensor([target_yaw], device=sim_device) 
    ctx.rotate_steps = 500
    ctx.finish_cf = False

    gpis.plot_results(filename="gpis_results.png")
    print("GPIS target selected")

def should_trigger_contact(depth1: float, depth2: float, max_threshold: float) -> bool:
    return depth1 < max_threshold or depth2 < max_threshold

def step_goal_seeking(ctx, drone, drone_state, controller, vel_side):
    r_transpose, _ = process_quaternion(drone_state)
    goal_world = transform_velocity(vel_side, r_transpose)
    apply_control(drone, drone_state, controller, goal_world, "Find the goal")
    ctx.goal_steps -= 1
    if ctx.goal_steps <= 0:
        ctx.mode = DroneMode.FORWARD


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

    # Positive contact samples during contour following
    if depth1 < max_threshold and ctx.cf_steps % 40 == 0:
        ctx.laser_value1 = 0
    if depth2 < max_threshold and ctx.cf_steps % 40 == 0:
        ctx.laser_value2 = 0

    if ctx.cf_steps <= 0:
        ctx.mode = DroneMode.BACKWARD

    ctx.depth_last = ctx.depth_now
    ctx.depth_now = depth2
    residual = ctx.depth_now - ctx.depth_last

    if residual > 0.08:
        ctx.goal_steps = 100
        ctx.cf_steps = 0
        ctx.backward_steps = 0
        ctx.rotate_steps = 0
        ctx.mode = DroneMode.GOAL_SEEKING


def step_backward(ctx, drone, drone_state, controller, vel_backward):
    r_transpose, _ = process_quaternion(drone_state)
    backward_world = transform_velocity(vel_backward, r_transpose)
    apply_control(drone, drone_state, controller, backward_world, "Fly backward")
    ctx.backward_steps -= 1

    if ctx.backward_steps <= 0:
        _, ctx.before_yaw = process_quaternion(drone_state)
        ctx.target_yaw = ctx.before_yaw + ctx.random_yaw
        ctx.mode = DroneMode.ROTATING
        ctx.laser_value1 = -1
        ctx.laser_value2 = -1


def step_rotating(ctx, drone, drone_state, controller, sim_device, log: GPISLog):
    if ctx.direction_changes_completed >= 3 and ctx.finish_cf:
        run_gpis_and_set_next_heading(ctx, log, sim_device)
    _, current_yaw = process_quaternion(drone_state)
    perform_attitude_control(drone, drone_state, controller, ctx.target_yaw, "Change orientation")
    ctx.rotate_steps -= 1

    if torch.abs(normalize_angle(current_yaw) - normalize_angle(ctx.target_yaw)) < 1.5:
        ctx.rotate_steps = 0
        ctx.direction_changes_completed += 1
        ctx.mode = DroneMode.FORWARD


    print(torch.rad2deg(current_yaw))
    print(torch.rad2deg(ctx.target_yaw))


def step_forward(
    drone,
    drone_state,
    depth1,
    depth2,
    vel_forward,
    vel_backward,
    vel_side,
    controller,
    yaw_left,
    yaw_right,
    min_threshold,
    max_threshold,
):
    control_drone(
        drone,
        drone_state,
        depth1,
        depth2,
        vel_forward,
        vel_backward,
        vel_side,
        controller,
        yaw_left,
        yaw_right,
        min_threshold,
        max_threshold,
    )


def is_out_of_room(x, y, room_x_min, room_x_max, room_y_min, room_y_max):
    return x < room_x_min or x > room_x_max or y < room_y_min or y > room_y_max


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
    env_rotation_deg = float(cfg.env_rotation_deg)
    scene_utils.create_wall2(env_rotation_deg)

    n = 1
    max_threshold = 0.15
    min_threshold = 0.1

    room_x_min, room_x_max = -2.8, 2.8
    room_y_min, room_y_max = -2.8, 2.8

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

    vel_forward = torch.tensor([0.2, 0.0, 0.0], device=sim.device)
    vel_side = torch.tensor([0.0, -0.2, 0.0], device=sim.device)
    vel_backward = torch.tensor([-0.2, 0.0, 0.0], device=sim.device)

    cf_vel_forward = torch.tensor([0.05, 0.0, 0.0], device=sim.device)
    cf_vel_backward = torch.tensor([-0.05, 0.0, 0.0], device=sim.device)

    yaw_right = torch.tensor([np.deg2rad(-25)], device=sim.device)
    yaw_left = torch.tensor([np.deg2rad(25)], device=sim.device)


    controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)

    drone_state = drone.get_state()[..., :13].squeeze(0)

    ctx = FSMContext()
    log = GPISLog()
    success_filename = build_log_filename(env_rotation_deg, "success")
    fail_filename = build_log_filename(env_rotation_deg, "fail")
    from tqdm import tqdm

    for i in tqdm(range(25000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue

        depth1_noisy = read_lidar_with_noise(lidar_interface, lidar_path1, mean=0.0, std=0.00)
        depth2_noisy = read_lidar_with_noise(lidar_interface, lidar_path2, mean=0.0, std=0.00)

        # print(f"Depth1: {depth1_noisy:.4f}")
        # print(f"Depth2: {depth2_noisy:.4f}")

        if ctx.mode == DroneMode.GOAL_SEEKING:
            step_goal_seeking(ctx, drone, drone_state, controller, vel_side)

        elif ctx.mode == DroneMode.CONTOUR_FOLLOWING:
            step_contour_following(
                ctx, drone, drone_state, depth1_noisy, depth2_noisy,
                cf_vel_forward, cf_vel_backward, vel_side,
                controller, yaw_left, yaw_right,
                min_threshold, max_threshold
            )
           

        elif ctx.mode == DroneMode.BACKWARD:
            step_backward(ctx, drone, drone_state, controller, vel_backward)
            

        elif ctx.mode == DroneMode.ROTATING:
            step_rotating(ctx, drone, drone_state, controller, sim.device, log)

        elif ctx.mode == DroneMode.FORWARD:
            if should_trigger_contact(depth1_noisy, depth2_noisy, max_threshold):
                enter_contour_following(ctx, sim.device)
            else:
                step_forward(
                    drone,
                    drone_state,
                    depth1_noisy,
                    depth2_noisy,
                    vel_forward,
                    vel_backward,
                    vel_side,
                    controller,
                    yaw_left,
                    yaw_right,
                    min_threshold,
                    max_threshold,
                )

        sim.step(render=(i % 10 == 0))
        drone_state = drone.get_state()[..., :13].squeeze(0)

        ctx.state_x = drone.get_state()[..., 0].item()
        ctx.state_y = drone.get_state()[..., 1].item()
        _, state_yaw_tensor = process_quaternion(drone_state)
        ctx.state_yaw = state_yaw_tensor.item()

        update_log(log, ctx, depth1_noisy, depth2_noisy)
        reset_laser_labels(ctx)

        # print(drone_state)
        # print(ctx.direction_changes_completed)

        if is_out_of_room(
            ctx.state_x,
            ctx.state_y,
            room_x_min,
            room_x_max,
            room_y_min,
            room_y_max,
        ):
            print(
                f"Drone left the room boundary "
                f"(x={ctx.state_x:.3f}, y={ctx.state_y:.3f}). Stopping the run."
            )
            save_log_to_csv(log, success_filename)
            break

    else:
        save_log_to_csv(log, fail_filename)

    simulation_app.close()


if __name__ == "__main__":
    main()