import os
from typing import Dict, Optional
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from tensordict import TensorDict
import dataclasses
from omni_drones.utils.torch import quaternion_to_euler

# 这里假设 LeePositionController 已经导入
from omni_drones.controllers import LeePositionController

@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=cfg.sim.dt,
        rendering_dt=cfg.sim.dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    n = 4
    drone_model_cfg = cfg.drone_model
    drone, _ = MultirotorBase.make(drone_model_cfg.name, "LeePositionController", cfg.sim.device)

    # spawn drones
    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    drone.spawn(translations=translations)
    scene_utils.design_scene()

    # camera setup
    camera_cfg = PinholeCameraCfg(sensor_tick=0, resolution=(320, 240), data_types=["rgb", "distance_to_camera"])
    camera_sensor = Camera(camera_cfg)
    camera_sensor.spawn([f"/World/envs/env_0/{drone.name}_{i}/base_link/Camera" for i in range(n)])
    camera_vis = Camera(dataclasses.replace(camera_cfg, resolution=(960, 720)))

    sim.reset()
    camera_sensor.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera")
    camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()

    init_pos, init_rot = drone.get_world_poses(True)
    init_vels = torch.zeros(n, 6, device=sim.device)
    target_height = 1.5 + 0.5 * torch.arange(n, device=sim.device).float()
    target_yaw = torch.zeros(n, 1, device=sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)

    # 初始化 Lee Controller
    controller = LeePositionController(
        g=9.81,
        uav_params=drone.params  # 直接使用 drone 内部参数
    ).to(sim.device)


    reset()

    frames_sensor = []
    frames_vis = []

    from tqdm import tqdm
    for i in tqdm(range(300)):  
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue

        # 获取 drone state
        state13 = drone.get_state()[..., :13]
        pos = state13[..., 0:3]
        quat = state13[..., 3:7]
        vel_lin = state13[..., 7:10]
        body_rates = state13[..., 10:13]

        # Lee Controller 计算 thrust + rotor commands
        target_pos = torch.zeros_like(pos)
        target_pos[..., 2] = target_height
        cmd = controller.compute(
            root_state=state13,
            target_pos=target_pos,
            target_vel=torch.zeros_like(vel_lin),
            target_acc=torch.zeros_like(vel_lin),
            target_yaw=target_yaw
        )

        # 打印调试信息
        euler = quaternion_to_euler(quat)
        print(f"Step {i}:")
        print(f"  pos: {pos.detach().cpu().numpy()}")
        print(f"  euler (r,p,y): {euler.detach().cpu().numpy()}")
        print(f"  lin vel: {vel_lin.detach().cpu().numpy()}")
        print(f"  body rates: {body_rates.detach().cpu().numpy()}")
        print(f"  target height: {target_height.cpu().numpy()}")
        print(f"  action: {cmd.detach().cpu().numpy()}")
        print("-"*60)

        # 应用控制
        drone.apply_action(cmd)
        sim.step(render=True)

        if i % 2 == 0:
            frames_sensor.append(camera_sensor.get_images().cpu())
            frames_vis.append(camera_vis.get_images().cpu())

        if i % 300 == 0:
            reset()

    # 保存视频
    from torchvision.io import write_video
    for image_type, arrays in torch.stack(frames_sensor).items():
        for drone_id, arrays_drone in enumerate(arrays.unbind(1)):
            if image_type == "rgb":
                arrays_drone = arrays_drone.permute(0, 2, 3, 1)[..., :3]
                write_video(f"demo_rgb_{drone_id}.mp4", arrays_drone, fps=1/cfg.sim.dt)
            elif image_type == "distance_to_camera":
                arrays_drone = -torch.nan_to_num(arrays_drone, 0).permute(0, 2, 3, 1)
                arrays_drone = arrays_drone.expand(*arrays_drone.shape[:-1], 3)
                write_video(f"demo_depth_{drone_id}.mp4", arrays_drone, fps=0.5/cfg.sim.dt)

    simulation_app.close()

if __name__ == "__main__":
    main()
