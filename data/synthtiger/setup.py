import os

from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))


def read_version():
    data = {}
    path = os.path.join(ROOT, "synthtiger", "_version.py")
    with open(path, "r", encoding="utf-8") as fp:
        exec(fp.read(), data)
    return data["__version__"]


def read_long_description():
    path = os.path.join(ROOT, "README.md")
    with open(path, "r", encoding="utf-8") as fp:
        text = fp.read()
    return text


setup(
    name="synthtiger",
    version=read_version(),
    description="Synthetic text image generator for OCR model",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Moonbin Yim, Yoonsik Kim, Han-Cheol Cho, Sungrae Park",
    url="https://github.com/clovaai/synthtiger",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "arabic-reshaper==3.0.0",
        "blend-modes==2.2.0",
        "fonttools==4.60.1",
        "imgaug==0.4.0",
        "numpy==1.24.4",
        "opencv-python==4.11.0.86",
        "pillow==11.3.0",
        "pygame==2.6.1",
        "python-bidi==0.6.6",
        "pytweening==1.2.0",
        "pyyaml==6.0.3",
        "regex==2025.9.18",
        "scipy==1.15.3",
        "matplotlib==3.10.6",
        "requests==2.32.5",
        "svg-path==7.0",
    ],
    entry_points={
        "console_scripts": [
            "tiger = synthtiger.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
## 安装JPEG开发库和其他可能需要的图像库
# apt-get update && apt-get install -y libjpeg-dev zlib1g-dev libpng-dev libtiff-dev libfreetype6-dev