from setuptools import find_packages, setup

setup(
    name="omni_drones_whiskers",
    version="0.1.1",
    author="TacYee",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "imageio",
        "plotly",
        "einops",
        "pandas",
        "moviepy",
        "av",
        "torchrl==0.3.1",  # for torch==2.2.2
        "tensordict==0.3.2",
    ],
)
