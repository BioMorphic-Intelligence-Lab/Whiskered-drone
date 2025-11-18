# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Sequence, Union, Optional
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as script_utils
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid,DynamicCylinder,FixedCylinder,FixedCuboid 
from omni_drones.utils.torch import euler_to_quaternion
import torch
import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import omni_drones.utils.kit as kit_utils
from pxr import Usd, UsdGeom, Gf
import numpy as np
import omni.isaac.core.utils.stage as stage_utils

from pxr import UsdGeom, Gf, UsdPhysics
def set_transform_safely(xform_prim, position=None, scale=None):
    xform = UsdGeom.Xformable(xform_prim)
    existing_ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}

    if position is not None:
        pos = [float(v) for v in position]
        if 'xformOp:translate' in existing_ops:
            existing_ops['xformOp:translate'].Set(Gf.Vec3d(*pos))
        else:
            xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

    if scale is not None:
        scl = [float(v) for v in scale]
        if 'xformOp:scale' in existing_ops:
            existing_ops['xformOp:scale'].Set(Gf.Vec3f(*scl))
        else:
            xform.AddScaleOp().Set(Gf.Vec3f(*scl))

def add_colliders(root_prim):
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            print(f"[Collider] Adding to: {desc_prim.GetPath()}")

            # 通用 USD 碰撞 API
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)

            # 添加 PhysX 属性（不设置 approximation）
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            physx_api.CreateContactOffsetAttr().Set(0.01)
            physx_api.CreateRestOffsetAttr().Set(0.0)

            # ✅ 注意：不调用 CreateCollisionApproximationAttr()，因为你当前不支持

            # 日志
            print(f"[Collider] ✅ Done for {desc_prim.GetName()}")



def add_custom_asset_with_physics(stage, prim_path, usd_path, position, scale, mass=1.0):
    # 加载模型并设置变换
    xform_prim = UsdGeom.Xform.Define(stage, prim_path)
    xform_prim.GetPrim().GetReferences().AddReference(usd_path)
    set_transform_safely(xform_prim, position=position, scale=scale)
    prim = xform_prim.GetPrim()

    # ✅ 添加子 mesh 的 collider
    add_colliders(prim)

    # 设置质量属性（可选）
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass)

    return prim



def ensure_physics_scene():
    stage = stage_utils.get_current_stage()
    scene_path = "/physicsScene"

    if not stage.GetPrimAtPath(scene_path):
        print("Creating /physicsScene ...")
        physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    else:
        print("/physicsScene already exists.")

    # 🧠 关键：确保 SimulationContext 知道使用该 scene
    from omni.isaac.core.simulation_context.simulation_context import SimulationContext
    SimulationContext.instance()._physics_scene_path = scene_path



def rotate_position(position, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians),  np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    rotated_position = np.dot(rotation_matrix, position)
    return rotated_position


def design_scene():
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True,
    )
    # prim_utils.create_prim(
    #     "/World/Light/GreySphere",
    #     "SphereLight",
    #     translation=(4.5, 3.5, 10.0),
    # )
    # # Lights-2
    # prim_utils.create_prim(
    #     "/World/Light/WhiteSphere",
    #     "SphereLight",
    #     translation=(-4.5, 3.5, 10.0),
    # )

def create_wall():
    ensure_physics_scene() 
    world = World()
    cubes = [
            {"name": "fancy_cube1", "position": np.array([0, -2.1, 0]), "scale": np.array([4, 0.2, 2])},
            {"name": "fancy_cube2", "position": np.array([-2.1, 0, 0]), "scale": np.array([0.2, 4, 2])},
            {"name": "fancy_cube3", "position": np.array([2.1, 0, 0]), "scale": np.array([0.2, 4, 2])},
            {"name": "fancy_cube4", "position": np.array([-0.9, 2.15, 0]), "scale": np.array([2.2, 0.3, 2])},
            {"name": "fancy_cube5", "position": np.array([1.7, 2.15, 0]), "scale": np.array([0.6, 0.3, 2])}
        ]
    # 生成一个随机旋转角度（以度为单位）
    random_angle = np.random.uniform(0, 360)
    # 将随机角度转换为弧度，并生成对应的欧拉角张量
    euler_angles = torch.tensor([0, 0, np.radians(random_angle)])  # 这里假设旋转轴是z轴
    random_rotation = euler_to_quaternion(euler_angles).numpy()
    
    for cube in cubes:
            rotated_position = rotate_position(cube['position'], random_angle)
            world.scene.add(DynamicCuboid(
                prim_path=f"/World/{cube['name']}",
                name=cube['name'],
                position=rotated_position,
                scale=cube['scale'],
                orientation=random_rotation,
                mass = 0.5,
                color=np.array([0, 0, 1.0])
            ))

def create_wall2(orientation = 0):
    ensure_physics_scene() 
    world = World()
    cubes = [
            {"name": "fancy_cube1", "position": np.array([-1.5, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube2", "position": np.array([0.5, -2.1, 0]), "scale": np.array([3, 0.2, 2])},
            {"name": "fancy_cube3", "position": np.array([-1.1, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube4", "position": np.array([-2.1, 0.5, 0]), "scale": np.array([0.2, 3.4, 2])},
            {"name": "fancy_cube5", "position": np.array([2.1, 0.0, 0]), "scale": np.array([0.2, 4.4, 2])},
            {"name": "fancy_cube6", "position": np.array([-0.6, 2.1, 0]), "scale": np.array([2.8, 0.2, 2])}
        ]

    scale_factor = 1.33333

    for cube in cubes:
        cube["position"] = cube["position"] * scale_factor
        cube["scale"] = cube["scale"] * scale_factor
    # 生成一个随机旋转角度（以度为单位）
    random_angle = orientation
    # 将随机角度转换为弧度，并生成对应的欧拉角张量
    euler_angles = torch.tensor([0, 0, np.radians(random_angle)])  # 这里假设旋转轴是z轴
    random_rotation = euler_to_quaternion(euler_angles).numpy()
    
    for cube in cubes:
            rotated_position = rotate_position(cube['position'], random_angle)
            world.scene.add(FixedCuboid(
                prim_path=f"/World/{cube['name']}",
                name=cube['name'],
                position=rotated_position,
                scale=cube['scale'],
                orientation=random_rotation,
                color=np.array([0, 0, 1.0])
            ))

def create_wall3(orientation = 0):
    ensure_physics_scene() 
    world = World()
    cubes = [
            # 左下角凹
            {"name": "fancy_cube1", "position": np.array([-1.5, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube2", "position": np.array([0, -2.1, 0]), "scale": np.array([2, 0.2, 2])},
            {"name": "fancy_cube3", "position": np.array([-1.1, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube4", "position": np.array([-2.1, 0, 0]), "scale": np.array([0.2, 2.4, 2])},
            {"name": "fancy_cube5", "position": np.array([2.1, 0.5, 0]), "scale": np.array([0.2, 3.4, 2])},
            {"name": "fancy_cube6", "position": np.array([-0.1, 2.1, 0]), "scale": np.array([1.8, 0.2, 2])},

            # 左上角凹
            {"name": "fancy_cube7", "position": np.array([-1.5, 1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube8", "position": np.array([-1.1, 1.7, 0]), "scale": np.array([0.2, 1, 2])},

            # 右下角凹
            {"name": "fancy_cube9", "position": np.array([1.5, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube10", "position": np.array([1.1, -1.7, 0]), "scale": np.array([0.2, 1, 2])}
    ]

    scale_factor = 1.33333

    for cube in cubes:
        cube["position"] = cube["position"] * scale_factor
        cube["scale"] = cube["scale"] * scale_factor
    # 生成一个随机旋转角度（以度为单位）
    random_angle = orientation
    # 将随机角度转换为弧度，并生成对应的欧拉角张量
    euler_angles = torch.tensor([0, 0, np.radians(random_angle)])  # 这里假设旋转轴是z轴
    random_rotation = euler_to_quaternion(euler_angles).numpy()
    
    for cube in cubes:
            rotated_position = rotate_position(cube['position'], random_angle)
            world.scene.add(FixedCuboid(
                prim_path=f"/World/{cube['name']}",
                name=cube['name'],
                position=rotated_position,
                scale=cube['scale'],
                orientation=random_rotation,
                color=np.array([0, 0, 1.0])
            ))

def create_wall4(orientation = 0):
    ensure_physics_scene() 
    world = World()
    
    cubes = [
            # 左下角凹
            {"name": "fancy_cube1", "position": np.array([-1.25, -2.1, 0]), "scale": np.array([1.5, 0.2, 2])},
            {"name": "fancy_cube2", "position": np.array([0, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube3", "position": np.array([-0.4, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube4", "position": np.array([-2.1, 0, 0]), "scale": np.array([0.2, 4.4, 2])},
            {"name": "fancy_cube5", "position": np.array([2.1, 0.6, 0]), "scale": np.array([0.2, 3.2, 2])},
            {"name": "fancy_cube6", "position": np.array([0, 1.1, 0]), "scale": np.array([1, 0.2, 2])},

            # 左上角凹
            {"name": "fancy_cube7", "position": np.array([-1.25, 2.1, 0]), "scale": np.array([1.5, 0.2, 2])},
            {"name": "fancy_cube8", "position": np.array([-0.4, 1.7, 0]), "scale": np.array([0.2, 1, 2])},

            # 右下角凹
            {"name": "fancy_cube9", "position": np.array([1.25, -2.1, 0]), "scale": np.array([1.5, 0.2, 2])},
            {"name": "fancy_cube10", "position": np.array([0.4, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube11", "position": np.array([0.4, 1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube12", "position": np.array([1.25, 2.1, 0]), "scale": np.array([1.5, 0.2, 2])},
    ]


    scale_factor = 1.33333

    for cube in cubes:
        cube["position"] = cube["position"] * scale_factor
        cube["scale"] = cube["scale"] * scale_factor
    # 生成一个随机旋转角度（以度为单位）
    random_angle = orientation
    # 将随机角度转换为弧度，并生成对应的欧拉角张量
    euler_angles = torch.tensor([0, 0, np.radians(random_angle)])  # 这里假设旋转轴是z轴
    random_rotation = euler_to_quaternion(euler_angles).numpy()
    
    for cube in cubes:
            rotated_position = rotate_position(cube['position'], random_angle)
            world.scene.add(FixedCuboid(
                prim_path=f"/World/{cube['name']}",
                name=cube['name'],
                position=rotated_position,
                scale=cube['scale'],
                orientation=random_rotation,
                color=np.array([0, 0, 1.0])
            ))

def create_wall5(orientation = 0):
    ensure_physics_scene() 
    world = World()
    stage = stage_utils.get_current_stage()
    cubes = [
            {"name": "fancy_cube1", "position": np.array([-1.5, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube2", "position": np.array([0.5, -2.1, 0]), "scale": np.array([3, 0.2, 2])},
            {"name": "fancy_cube3", "position": np.array([-1.1, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube4", "position": np.array([-2.1, 0.5, 0]), "scale": np.array([0.2, 3.4, 2])},
            {"name": "fancy_cube5", "position": np.array([2.1, 0.0, 0]), "scale": np.array([0.2, 4.4, 2])},
            {"name": "fancy_cube6", "position": np.array([-0.6, 2.1, 0]), "scale": np.array([2.8, 0.2, 2])},
        ]
    
    # 你的障碍物列表
    custom_assets = [
        {
            "name": "fancy_asset1",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([0, -0.9, 0]),
            "scale": np.array([0.1125, 0.1125, 0.1])
        },        
        {
            "name": "fancy_asset2",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([1.1, 1.1, 0]),
            "scale": np.array([0.1, 0.1, 0.1])
        },
        {
            "name": "fancy_asset3",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([-1.0,0.9, 0]),
            "scale": np.array([0.125, 0.125, 0.1])
        }
    ]

    for asset in custom_assets:
        add_custom_asset_with_physics(
            stage,
            f"/World/{asset['name']}",
            asset["usd_path"],
            position=asset["position"],
            scale=asset["scale"],
            mass=1.0,
        )



    scale_factor = 1.33333

    for cube in cubes:
        cube["position"] = cube["position"] * scale_factor
        cube["scale"] = cube["scale"] * scale_factor
    # 生成一个随机旋转角度（以度为单位）
    random_angle = orientation
    # 将随机角度转换为弧度，并生成对应的欧拉角张量
    euler_angles = torch.tensor([0, 0, np.radians(random_angle)])  # 这里假设旋转轴是z轴
    random_rotation = euler_to_quaternion(euler_angles).numpy()
    
    for cube in cubes:
            rotated_position = rotate_position(cube['position'], random_angle)
            world.scene.add(FixedCuboid(
                prim_path=f"/World/{cube['name']}",
                name=cube['name'],
                position=rotated_position,
                scale=cube['scale'],
                orientation=random_rotation,
                color=np.array([0, 0, 1.0])
            ))

def create_wall6(orientation = 0):
    ensure_physics_scene() 
    world = World()
    stage = stage_utils.get_current_stage()
    # 你的障碍物列表
    custom_assets = [    
        {
            "name": "fancy_asset2",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([1.1, 1.1, 0]),
            "scale": np.array([0.1125, 0.1125, 0.1])
        },
        {
            "name": "fancy_asset3",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([-0.55,0, 0]),
            "scale": np.array([0.1, 0.1, 0.1])
        }
    ]

    for asset in custom_assets:
        add_custom_asset_with_physics(
            stage,
            f"/World/{asset['name']}",
            asset["usd_path"],
            position=asset["position"],
            scale=asset["scale"],
            mass=1.0,
        )



    cubes = [
            # 左下角凹
            {"name": "fancy_cube1", "position": np.array([-1.5, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube2", "position": np.array([0, -2.1, 0]), "scale": np.array([2, 0.2, 2])},
            {"name": "fancy_cube3", "position": np.array([-1.1, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube4", "position": np.array([-2.1, 0, 0]), "scale": np.array([0.2, 2.4, 2])},
            {"name": "fancy_cube5", "position": np.array([2.1, 0.5, 0]), "scale": np.array([0.2, 3.4, 2])},
            {"name": "fancy_cube6", "position": np.array([-0.1, 2.1, 0]), "scale": np.array([1.8, 0.2, 2])},

            # 左上角凹
            {"name": "fancy_cube7", "position": np.array([-1.5, 1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube8", "position": np.array([-1.1, 1.7, 0]), "scale": np.array([0.2, 1, 2])},

            # 右下角凹
            {"name": "fancy_cube9", "position": np.array([1.5, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube10", "position": np.array([1.1, -1.7, 0]), "scale": np.array([0.2, 1, 2])}
    ]

    scale_factor = 1.33333

    for cube in cubes:
        cube["position"] = cube["position"] * scale_factor
        cube["scale"] = cube["scale"] * scale_factor
    # 生成一个随机旋转角度（以度为单位）
    random_angle = orientation
    # 将随机角度转换为弧度，并生成对应的欧拉角张量
    euler_angles = torch.tensor([0, 0, np.radians(random_angle)])  # 这里假设旋转轴是z轴
    random_rotation = euler_to_quaternion(euler_angles).numpy()
    
    for cube in cubes:
            rotated_position = rotate_position(cube['position'], random_angle)
            world.scene.add(FixedCuboid(
                prim_path=f"/World/{cube['name']}",
                name=cube['name'],
                position=rotated_position,
                scale=cube['scale'],
                orientation=random_rotation,
                color=np.array([0, 0, 1.0])
            ))
    
def create_wall7(orientation = 0):
    ensure_physics_scene() 
    world = World()
    stage = stage_utils.get_current_stage()
    # 你的障碍物列表
    custom_assets = [    
        {
            "name": "fancy_asset2",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([1.2, 0, 0]),
            "scale": np.array([0.085, 0.085, 0.1])
        },
        {
            "name": "fancy_asset3",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([-1.2,0, 0]),
            "scale": np.array([0.085, 0.085, 0.1])
        }
    ]

    for asset in custom_assets:
        add_custom_asset_with_physics(
            stage,
            f"/World/{asset['name']}",
            asset["usd_path"],
            position=asset["position"],
            scale=asset["scale"],
            mass=1.0,
        )

    cubes = [
            # 左下角凹
            {"name": "fancy_cube1", "position": np.array([-1.25, -2.1, 0]), "scale": np.array([1.5, 0.2, 2])},
            {"name": "fancy_cube2", "position": np.array([0, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube3", "position": np.array([-0.4, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube4", "position": np.array([-2.1, 0, 0]), "scale": np.array([0.2, 4.4, 2])},
            {"name": "fancy_cube5", "position": np.array([2.1, 0.6, 0]), "scale": np.array([0.2, 3.2, 2])},
            {"name": "fancy_cube6", "position": np.array([0, 1.1, 0]), "scale": np.array([1, 0.2, 2])},

            # 左上角凹
            {"name": "fancy_cube7", "position": np.array([-1.25, 2.1, 0]), "scale": np.array([1.5, 0.2, 2])},
            {"name": "fancy_cube8", "position": np.array([-0.4, 1.7, 0]), "scale": np.array([0.2, 1, 2])},

            # 右下角凹
            {"name": "fancy_cube9", "position": np.array([1.25, -2.1, 0]), "scale": np.array([1.5, 0.2, 2])},
            {"name": "fancy_cube10", "position": np.array([0.4, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube11", "position": np.array([0.4, 1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube12", "position": np.array([1.25, 2.1, 0]), "scale": np.array([1.5, 0.2, 2])},
    ]


    scale_factor = 1.4

    for cube in cubes:
        cube["position"] = cube["position"] * scale_factor
        cube["scale"] = cube["scale"] * scale_factor
    # 生成一个随机旋转角度（以度为单位）
    random_angle = orientation
    # 将随机角度转换为弧度，并生成对应的欧拉角张量
    euler_angles = torch.tensor([0, 0, np.radians(random_angle)])  # 这里假设旋转轴是z轴
    random_rotation = euler_to_quaternion(euler_angles).numpy()
    
    for cube in cubes:
            rotated_position = rotate_position(cube['position'], random_angle)
            world.scene.add(FixedCuboid(
                prim_path=f"/World/{cube['name']}",
                name=cube['name'],
                position=rotated_position,
                scale=cube['scale'],
                orientation=random_rotation,
                color=np.array([0, 0, 1.0])
            ))

def create_cylinder(orientation = 0):
    ensure_physics_scene() 
    stage = stage_utils.get_current_stage()
    # 你的障碍物列表
    custom_assets = [    
        {
            "name": "fancy_asset1",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([1.5, -1.2, 0]),
            "scale": np.array([0.1, 0.3, 0.1])
        },
        {
            "name": "fancy_asset2",
            "usd_path": "/home/chaoxiangye/OmniDrones/cylinder.usdc",
            "position": np.array([1.5, 1.2, 0]),
            "scale": np.array([0.1, 0.3, 0.1])
        },
    ]

    for asset in custom_assets:
        add_custom_asset_with_physics(
            stage,
            f"/World/{asset['name']}",
            asset["usd_path"],
            position=asset["position"],
            scale=asset["scale"],
            mass=1.0,
        )

def create_cylinder2(orientation = 0):
    ensure_physics_scene() 
    stage = stage_utils.get_current_stage()
    # 你的障碍物列表
    custom_assets = [    
        {
            "name": "fancy_asset2",
            "usd_path": "/home/chaoxiangye/OmniDrones/spiral.usdc",
            "position": np.array([6, 0, 0]),
            "scale": np.array([0.3, 0.3, 3])
        },
    ]

    for asset in custom_assets:
        add_custom_asset_with_physics(
            stage,
            f"/World/{asset['name']}",
            asset["usd_path"],
            position=asset["position"],
            scale=asset["scale"],
            mass=500.0,
        )

def create_cylinder3(orientation = 0):
    ensure_physics_scene() 
    stage = stage_utils.get_current_stage()
    # 你的障碍物列表
    custom_assets = [    
        {
            "name": "fancy_asset2",
            "usd_path": "/home/chaoxiangye/OmniDrones/wave2.usdc",
            "position": np.array([8,-1, 0]),
            "scale": np.array([0.3, 0.3, 0.1])
        },
    ]

    for asset in custom_assets:
        add_custom_asset_with_physics(
            stage,
            f"/World/{asset['name']}",
            asset["usd_path"],
            position=asset["position"],
            scale=asset["scale"],
            mass=500.0,
        )
        
def load_room_environment():
    from omni.isaac.core.utils.stage import add_reference_to_stage
    ensure_physics_scene()

    add_reference_to_stage(
        usd_path="/Isaac/Environments/Simple_Room/simple_room.usd",
        prim_path="/World/Room"
    )

def create_rope(
    xform_path: str = "/World/rope",
    translation=(0, 0, 0),
    from_prim: Union[str, Usd.Prim] = None,
    to_prim: Union[str, Usd.Prim] = None,
    num_links: int = 24,
    link_length: float = 0.06,
    rope_damping: float = 10.0,
    rope_stiffness: float = 1.0,
    color=(0.4, 0.2, 0.1),
    enable_collision: bool = False,
):
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    stage = stage_utils.get_current_stage()
    ropeXform = UsdGeom.Xform.Define(stage, xform_path)
    ropeXform.AddTranslateOp().Set(Gf.Vec3f(*translation))
    ropeXform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
    link_radius = 0.02
    joint_offset = link_length / 2 - link_length / 8

    links = []
    for i in range(num_links):
        link_path = f"{xform_path}/seg_{i}"
        location = (i * (link_length - link_length / 4), 0, 0)

        capsuleGeom = UsdGeom.Capsule.Define(stage, link_path)
        capsuleGeom.CreateHeightAttr(link_length / 2)
        capsuleGeom.CreateRadiusAttr(link_radius)
        capsuleGeom.CreateAxisAttr("X")
        capsuleGeom.AddTranslateOp().Set(location)
        capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
        capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        capsuleGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateMassAttr().Set(0.01)

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        # physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        # physxCollisionAPI.CreateContactOffsetAttr().Set(0.02)
        capsuleGeom.GetPrim().GetAttribute("physics:collisionEnabled")

        if len(links) > 0:
            # jointPath = f"{link_path}/joint_{i}"
            # joint = UsdPhysics.Joint.Define(stage, jointPath)
            # joint.CreateBody0Rel().SetTargets([links[-1].GetPath()])
            # joint.CreateBody1Rel().SetTargets([link_path])

            # joint.CreateLocalPos0Attr().Set(Gf.Vec3f(joint_offset, 0, 0))
            # joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            # joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-joint_offset, 0, 0))
            # joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

            # # locked DOF (lock - low is greater than high)
            # d6Prim = joint.GetPrim()
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)

            # # Moving DOF:
            # dofs = ["rotY", "rotZ"]
            # for d in dofs:
            #     limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
            #     limitAPI.CreateLowAttr(-110)
            #     limitAPI.CreateHighAttr(110)

            #     # joint drives for rope dynamics:
            #     driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
            #     driveAPI.CreateTypeAttr("force")
            #     driveAPI.CreateDampingAttr(rope_damping)
            #     driveAPI.CreateStiffnessAttr(rope_stiffness)
            joint: Usd.Prim = script_utils.createJoint(
                stage, "D6", links[-1], capsuleGeom.GetPrim()
            )
            joint.GetAttribute("physics:localPos0").Set((joint_offset, 0.0, 0.0))
            joint.GetAttribute("physics:localPos1").Set((-joint_offset, 0.0, 0.0))
            joint.GetAttribute("limit:rotY:physics:low").Set(-110)
            joint.GetAttribute("limit:rotY:physics:high").Set(110)
            joint.GetAttribute("limit:rotZ:physics:low").Set(-110)
            joint.GetAttribute("limit:rotZ:physics:high").Set(110)
            UsdPhysics.DriveAPI.Apply(joint, "rotY")
            UsdPhysics.DriveAPI.Apply(joint, "rotZ")
            joint.GetAttribute("drive:rotY:physics:damping").Set(rope_damping)
            joint.GetAttribute("drive:rotY:physics:stiffness").Set(rope_stiffness)
            joint.GetAttribute("drive:rotZ:physics:damping").Set(rope_damping)
            joint.GetAttribute("drive:rotZ:physics:stiffness").Set(rope_stiffness)

        links.append(capsuleGeom.GetPrim())

    if from_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "Fixed", from_prim, links[-1])
        # joint.GetAttribute('physics:excludeFromArticulation').Set(True)

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "Fixed", links[0], to_prim)
        joint.GetAttribute("physics:excludeFromArticulation").Set(True)

    return links


def create_bar(
    prim_path: str,
    length: float,
    translation=(0, 0, 0),
    from_prim: str = None,
    to_prim: str = None,
    mass: float = 0.02,
    enable_collision=False,
    color=(0.4, 0.4, 0.2),
):
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    stage = stage_utils.get_current_stage()

    capsuleGeom = UsdGeom.Capsule.Define(stage, f"{prim_path}/Capsule")
    capsuleGeom.CreateHeightAttr(length)
    capsuleGeom.CreateRadiusAttr(0.012)
    capsuleGeom.CreateAxisAttr("Z")
    capsuleGeom.AddTranslateOp().Set(Gf.Vec3f(*translation))
    capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
    capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    capsuleGeom.CreateDisplayColorAttr().Set([color])

    UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
    massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
    massAPI.CreateMassAttr().Set(mass)

    UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
    prim: Usd.Prim = capsuleGeom.GetPrim()
    prim.GetAttribute("physics:collisionEnabled").Set(enable_collision)

    if from_prim is not None:
        sphere = prim_utils.create_prim(
            f"{prim_path}/Sphere",
            "Sphere",
            translation=(0, 0, -length),
            attributes={"radius": 0.02},
        )
        UsdPhysics.RigidBodyAPI.Apply(sphere)
        UsdPhysics.CollisionAPI.Apply(sphere)
        sphere.GetAttribute("physics:collisionEnabled").Set(False)

        script_utils.createJoint(stage, "Fixed", from_prim, sphere)
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim, sphere)
        joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint.GetAttribute("limit:rotX:physics:high").Set(120)
        joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        joint.GetAttribute("drive:rotX:physics:damping").Set(0.0002)
        joint.GetAttribute("drive:rotY:physics:damping").Set(0.0002)

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim, to_prim)
        joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint.GetAttribute("limit:rotX:physics:high").Set(120)
        joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        joint.GetAttribute("drive:rotX:physics:damping").Set(0.0002)
        joint.GetAttribute("drive:rotY:physics:damping").Set(0.0002)

    return prim


