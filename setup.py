from setuptools import setup, find_packages

setup(
    name="bugan",
    url="https://github.com/buganart/BUGAN",
    author="buganart",
    description="3D GANs and more.",
    packages=find_packages(),
    install_requires=[
        "ConfigArgParse==1.7",
        "disjoint-set==0.6.3",
        "Pillow==10.4.0",
        "pytorch_lightning==1.8.0",
        "numpy==1.21.6",
        "torch==1.13.1",
        "torchsummary==1.5.1",
        "tqdm==4.66.5",
        "trimesh==4.4.8",
        "scipy==1.10.1",
        "scikit-image==0.21.0",
        "wandb==0.17.8",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
