from setuptools import setup, find_packages
import os

# Read version from __version__.py
version_file = os.path.join(os.path.dirname(__file__), "eggroll", "__version__.py")
with open(version_file, "r") as f:
    exec(f.read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eggroll-pytorch",
    version=__version__,
    author="Unofficial PyTorch Implementation",
    description="Unofficial PyTorch implementation of EGGROLL (Evolution Guided General Optimization via Low-rank Learning)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/eggroll-pytorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
)

