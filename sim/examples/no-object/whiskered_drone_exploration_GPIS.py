import torch
import pandas as pd
import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
import omni
import numpy as np
from utlis import *
from GPIS2 import GPISModel

@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core import World
    from omni_drones.controllers import LeePositionController, AttitudeController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion
    from omni.isaac.dynamic_control import _dynamic_control
    from pxr import Gf, Usd, UsdGeom, Sdf
    import typing
    from omni.isaac.range_sensor import _range_sensor

    from omni_drones.utils.torch import quaternion_to_rotation_matrix
    import carb


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
    scene_utils.create_wall2()
    
    n = 1 
    MAX_THRESHOLD = 0.45
    MIN_THRESHOLD = 0.4

    drone_cls = MultirotorBase.REGISTRY[cfg.drone_model]
    drone = drone_cls()

    translations = torch.zeros(n, 3, device = sim.device)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 1
    orientations = torch.zeros(n, 4, device = sim.device)
    orientations[:, 0] = 0.9238795325  # w
    orientations[:, 3] = 0.3826834324  # z
    drone.spawn(translations=translations,orientations=orientations)

    lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
    
    lidarPath1 = "/envs/env_0/Hummingbird_0/base_link/LidarSensor1"
    omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=lidarPath1,
            parent = "/World",
            min_range=0.3,
            max_range=0.5,
            draw_points=False,
            draw_lines=True,
            horizontal_fov=1,
            vertical_fov=1,
            horizontal_resolution=1,
            vertical_resolution=1,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=-45.0,
            enable_semantics=False,
        )
    
    lidarPath2 = "/envs/env_0/Hummingbird_0/base_link/LidarSensor2"
    omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=lidarPath2,
            parent = "/World",
            min_range=0.3,
            max_range=0.5,
            draw_points=False,
            draw_lines=True,
            horizontal_fov=1,
            vertical_fov=1,
            horizontal_resolution=1,
            vertical_resolution=1,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=-45.0,
            enable_semantics=False,
        )
    
    position1 = Gf.Vec3d(0.05,0.05,0) 
    rotation1 = Gf.Vec3d(0,0,0)
    scale1 = Gf.Vec3d(1,1,1) 
    omni.kit.commands.execute("TransformMultiPrimsSRTCpp",
            count=1,
            paths=['/World'+lidarPath1],
            new_translations=[position1[0], position1[1], position1[2]],
            new_rotation_eulers=[rotation1[0], rotation1[1], rotation1[2]],
            new_rotation_orders=[1, 0, 2],
            new_scales=[scale1[0], scale1[1], scale1[2]],
            old_translations=[0.0, 0.0, 0.0],
            old_rotation_eulers=[0.0, -0.0, -0.0],
            old_rotation_orders=[1, 0, 2],
            old_scales=[1.0, 1.0, 1.0],
            time_code=0.0,
            )

    position2 = Gf.Vec3d(-0.05,-0.05,0) 
    rotation2 = Gf.Vec3d(0,0,0)
    scale2 = Gf.Vec3d(1,1,1) 
    omni.kit.commands.execute("TransformMultiPrimsSRTCpp",
            count=1,
            paths=['/World'+lidarPath2],
            new_translations=[position2[0], position2[1], position2[2]],
            new_rotation_eulers=[rotation2[0], rotation2[1], rotation2[2]],
            new_rotation_orders=[1, 0, 2],
            new_scales=[scale2[0], scale2[1], scale2[2]],
            old_translations=[0.0, 0.0, 0.0],
            old_rotation_eulers=[0.0, -0.0, -0.0],
            old_rotation_orders=[1, 0, 2],
            old_scales=[1.0, 1.0, 1.0],
            time_code=0.0,
            )
    

    sim.reset()
    drone.initialize()


    init_pos = translations
    init_rot = orientations
    init_vels = torch.zeros(n, 6, device=sim.device)

    pos = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
    vel_forward = torch.tensor([0.2, 0.0, 0.0], device=sim.device)
    vel_side = torch.tensor([0, -0.2, 0.0], device=sim.device)
    vel_backward = torch.tensor([-0.2, 0.0, 0.0], device=sim.device)
    yaw = torch.tensor([0.0], device=sim.device)
    yaw_right_rad = np.deg2rad(-25)
    yaw_left_rad = np.deg2rad(25)
    yaw_right = torch.tensor([yaw_right_rad], device=sim.device)
    yaw_left = torch.tensor([yaw_left_rad], device=sim.device)
    vel_backward = torch.tensor([-0.2, 0, 0], device=sim.device)
    cf_vel_backward = torch.tensor([-0.05, 0, 0], device=sim.device)
    cf_vel_forward = torch.tensor([0.05, 0, 0], device=sim.device)
    rot = 0
    depth_now = 0.5
    depth_last = 0.5

    random_direction = 0
    random_direction_rad = 0
    random_yaw = 0
    CF_action_counter = 0
    backward_action_counter = 0
    direction_change_counter = 0
    goal_counter = 0
    before_yaw = 0
    current_yaw = 0
    depth1_noisy = 0
    depth2_noisy = 0

    # Define the rotation quaternion for a -45 degree rotation around the Z-axis
    theta = -45.0  # degrees
    theta_rad = np.radians(theta)
    cos_half_theta = np.cos(theta_rad / 2)
    sin_half_theta = np.sin(theta_rad / 2)
    rot_z_45 = torch.tensor([cos_half_theta, 0.0, 0.0, sin_half_theta], device=sim.device)
    state_yaw = 0
    state_x = 0
    state_y = 0
    state_xs = []
    state_ys = []
    state_yaws = []
    state_lasers1 = []
    state_lasers2 = []
    laser_value1 = 0
    laser_value2 = 0
    laser_values1 = []
    laser_values2 = []
    direction_changes_completed = 0
    finish_CF = False
    ROOM_X_MIN, ROOM_X_MAX = -2.8, 2.8
    ROOM_Y_MIN, ROOM_Y_MAX = -2.8, 2.8
    

    controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)
    # Att_controller = AttitudeController(g=9.81, uav_params=drone.params).to(sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)
        sim._physics_sim_view.flush()

    reset()
    drone_state = drone.get_state()[..., :13].squeeze(0)

    from tqdm import tqdm
    for i in tqdm(range(12000)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue
        depth1 = lidarInterface.get_linear_depth_data("/World" + lidarPath1)
        depth2 = lidarInterface.get_linear_depth_data("/World" + lidarPath2)

        print("Original depth1:", depth1)
        print("Original depth2:", depth2)

        mean = 0.0
        std_dev = 0.01
        noise1 = np.random.normal(mean, std_dev, depth1.shape)
        noise2 = np.random.normal(mean, std_dev, depth2.shape)


        depth1_noisy = depth1 + noise1
        depth2_noisy = depth2 + noise2

        print("Noisy depth1:", depth1_noisy)
        print("Noisy depth2:", depth2_noisy)
        if goal_counter > 0:
            R_transpose, _ = process_quaternion(drone_state, rot_z_45)
            goal_world = transform_velocity(vel_side, R_transpose)
            apply_control(drone, drone_state, controller, goal_world, "Find the goal")
            goal_counter -= 1
        elif CF_action_counter > 0:
            CF_action_counter = control_drone(drone, drone_state, depth1_noisy, depth2_noisy, cf_vel_forward, cf_vel_backward, 
                                                vel_side, rot_z_45, controller, yaw_left, yaw_right, 
                                                MIN_THRESHOLD, MAX_THRESHOLD, CF_action_counter)
            depth_last = depth_now
            depth_now = depth2_noisy
            residuals = depth_now - depth_last
            if residuals > 0.08:
                goal_counter = 100
                CF_action_counter = 0
                backward_action_counter = 0
                direction_change_counter = 0
        elif backward_action_counter > 0:
            R_transpose, _ = process_quaternion(drone_state, rot_z_45)
            backward_world = transform_velocity(vel_backward, R_transpose)
            apply_control(drone, drone_state, controller, backward_world, "fly backward")
            backward_action_counter -= 1
            if backward_action_counter == 0:
                laser_value1 = -1
                laser_value2 = -1
                depth1_noisy = depth1 + noise1
                depth2_noisy = depth2 + noise2
                _, before_yaw = process_quaternion(drone_state, rot_z_45)
            target_yaw = before_yaw + random_yaw
        elif direction_change_counter > 0:
            _, current_yaw = process_quaternion(drone_state, rot_z_45)
            perform_attitude_control(drone, drone_state, controller, target_yaw, "change orientation")
            direction_change_counter -= 1
            if torch.abs(normalize_angle(current_yaw) - normalize_angle(target_yaw)) < 1.5:
                direction_change_counter = 0
                direction_changes_completed += 1
            if direction_changes_completed >= 1 and finish_CF:
                gpis = GPISModel(state_xs, state_ys, state_yaws, state_lasers1,state_lasers2, laser_values1, laser_values2, curvature_threshold=-0.8)
                gpis.sample_data()
                gpis.train_model()
                gpis.predict()
                next_point = gpis.find_max_uncertainty_point()
                target_yaw = torch.tensor([np.arctan2(next_point[1] - state_y, next_point[0] - state_x)], device=sim.device) + 0.7853981
                direction_change_counter = 500
                gpis.plot_results(filename='gpis_results.png')
                finish_CF = False
            print(torch.rad2deg(current_yaw + 0.7853981))
            print(torch.rad2deg(target_yaw))

        else:
            if MAX_THRESHOLD > depth1_noisy > MIN_THRESHOLD and MAX_THRESHOLD > depth2_noisy > MIN_THRESHOLD:
                CF_action_counter = 220
                backward_action_counter = 250
                direction_change_counter = 300
                finish_CF = True
                random_direction_rad = np.deg2rad(-90)
                random_yaw = torch.tensor([random_direction_rad], device=sim.device)
                laser_value1 = 1
                laser_value2 = 1
                print("Touch")
            else:
                control_drone(drone, drone_state, depth1_noisy, depth2_noisy, vel_forward, 
                            vel_backward, vel_side, rot_z_45, controller, 
                            yaw_left, yaw_right,  MIN_THRESHOLD, MAX_THRESHOLD)
        sim.step(render=(i % 10 == 0))
        # lidarInterface.update() 
        drone_state = drone.get_state()[..., :13].squeeze(0)
        print(drone_state)
        print(direction_changes_completed)
        state_x = drone.get_state()[..., 0].item()
        state_y = drone.get_state()[..., 1].item()
        state_xs.append(state_x)
        state_ys.append(state_y) 
        _ , state_yaw = process_quaternion(drone_state, rot_z_45)
        state_yaws.append(state_yaw.item())
        state_lasers1.append(depth1_noisy.item()) 
        state_lasers2.append(depth2_noisy.item())
        laser_values1.append(laser_value1)
        laser_values2.append(laser_value2)
        laser_value1 = 0
        laser_value2 = 0 

        if state_x < ROOM_X_MIN or state_x > ROOM_X_MAX or state_y < ROOM_Y_MIN or state_y > ROOM_Y_MAX:
            print(f"Drone flew out of the environment (x={state_x}, y={state_y}), mission completed.")
            data = {
                'state_xs': state_xs,
                'state_ys': state_ys,
                'state_yaws': state_yaws,
                'state_lasers1': state_lasers1,
                'state_lasers2': state_lasers2,
                'laser_values1': laser_values1,
                'laser_values2': laser_values2,
            }
            df = pd.DataFrame(data)
            df.to_csv('T-3-GPIS_success.csv', index=False)  
            break  


    data = {
        'state_xs': state_xs,
        'state_ys': state_ys,
        'state_yaws': state_yaws,
        'state_lasers1': state_lasers1,
        'state_lasers2': state_lasers2,
        'laser_values1': laser_values1,
        'laser_values2': laser_values2,
    }
    df = pd.DataFrame(data)
    df.to_csv('T-3-GPIS_fail.csv', index=False)

    simulation_app.close()

if __name__ == "__main__":
    main()
