# Copyright (c) 2022-2025, The Psi Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'psilab' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # open television requirements
    "pytransform3d==3.5.0",
    "dex_retargeting==0.1.1",
    "vuer[all]==0.0.32rc7",
    "aiohttp==3.9.5",
    "aiortc==1.8.0",
    # zarr
    "zarr== 2.12.0",
    # diffusion policy
    "dill==0.3.5.1",
    "diffusers==0.33.1",
    "timm==1.0.12",
    "numba==0.57.0",
    "accelerate==0.13.2",
    # "diffusers==0.11.1",

    # 
    "numpy==1.23.5",
    # 
    "sapien==2.2.2",
    # psi-glove
    "pyserial==3.5",
    "openvr==1.23.701"
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Installation operation
setup(
    name="psilab",
    author="Psi Lab Project Developers",
    maintainer="Psi Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["psilab"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
        "Isaac Lab :: 2.0.2",

    ],
    zip_safe=False,
)
