# --- attitude controller 测试版 ---
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

from omni_drones.controllers import LeePositionController

@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=cfg.sim.dt,
        rendering_dt=cfg.sim.dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    n = 1  # 先只测试一个无人机
    drone_model_cfg = cfg.drone_model
    drone, _ = MultirotorBase.make(drone_model_cfg.name, "LeePositionController", cfg.sim.device)

    # spawn drone
    translations = torch.zeros(n, 3)
    translations[:, 2] = 1.0  # 悬停高度
    drone.spawn(translations=translations)
    scene_utils.design_scene()

    sim.reset()
    drone.initialize()
    init_pos, init_rot = drone.get_world_poses(True)
    init_vels = torch.zeros(n, 6, device=sim.device)

    def reset():
        drone._reset_idx(torch.tensor([0]))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)

    # 初始化 Lee Controller
    controller = LeePositionController(
        g=9.81,
        uav_params=drone.params
    ).to(sim.device)

    reset()

    # attitude 测试目标 (r,p,y) 单位: rad
    target_roll = torch.tensor([[0.1]], device=sim.device)   # 约 5.7°
    target_pitch = torch.tensor([[-0.1]], device=sim.device) # -5.7°
    target_yaw = torch.tensor([[0.5]], device=sim.device)    # 约 28.6°

    from tqdm import tqdm
    for i in tqdm(range(300)):
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue

        state13 = drone.get_state()[..., :13]
        pos = state13[..., 0:3]
        quat = state13[..., 3:7]

        # 姿态控制测试：固定位置，零速度
        target_pos = pos.clone()  # 当前位置
        target_vel = torch.zeros_like(pos)
        target_acc = torch.zeros_like(pos)

        cmd = controller.compute(
            root_state=state13,
            target_pos=target_pos,
            target_vel=target_vel,
            target_acc=target_acc,
            target_yaw=target_yaw
        )

        # 输出当前姿态
        from omni_drones.utils.torch import quaternion_to_euler
        euler = quaternion_to_euler(quat)
        print(f"Step {i}: euler (r,p,y) = {euler.detach().cpu().numpy()} | cmd = {cmd.detach().cpu().numpy()}")

        drone.apply_action(cmd)
        sim.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
