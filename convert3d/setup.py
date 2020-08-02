from setuptools import setup, find_packages

setup(
    name="convert3d",
    version="0.0.1",
    url="https://github.com/buganart/BUGAN/convert3d",
    author="buganart",
    description="Convert between 3D object representations",
    packages=find_packages(),
    install_requires=["click", "open3d", "joblib"],
    entry_points={"console_scripts": ["convert3d = convert3d.__main__:main"]},
)
