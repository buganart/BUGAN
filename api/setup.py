from setuptools import setup, find_packages

setup(
    name="bugan-api",
    url="https://github.com/buganart/BUGAN",
    author="buganart",
    description="3D GANs and more.",
    packages=find_packages(),
    install_requires=[
        "flask",
        "click",
    ],
)
