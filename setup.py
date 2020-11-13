from setuptools import setup, find_packages

setup(
    name="bugan",
    url="https://github.com/buganart/BUGAN",
    author="buganart",
    description="3D GANs and more.",
    packages=find_packages(),
    install_requires=[
        "ConfigArgParse",
        "disjoint-set==0.6.3",
        "Pillow",
        "pytorch_lightning",
        "numpy",
        "torch",
        "torchsummary",
        "tqdm",
        "trimesh",
        "scipy",
        "wandb",
    ],
    version_format="{tag}.dev{commitcount}+{gitsha}",
    setup_requires=["setuptools-git-version"],
)
