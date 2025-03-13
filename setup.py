# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import urllib
import urllib.error
import urllib.request
from pathlib import Path

import torch
from setuptools import find_packages, setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except BaseException:
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except BaseException:
        sys.exit("Both latest setuptools and wheel package are not found.  Please upgrade to latest setuptools: `pip install -U setuptools`")

ROCM_VERSION = os.environ.get('ROCM_VERSION', None)


if ROCM_VERSION is None and torch.version.hip:
    ROCM_VERSION = ".".join(torch.version.hip.split(".")[:2]) # print(torch.version.hip) -> 6.3.42131-fa1d09cbd
    os.environ["ROCM_VERSION"] = ROCM_VERSION

extensions = []
common_setup_kwargs = {
    "version": "0.0.0",
    "name": "rockthem",
    "author": "ModelCloud",
    "author_email": "qubitium@modelcloud.ai",
    "description": "Rockthem is the next-gen AI stack. Forget CUDA, use Rockthem.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/ModelCloud/rockthem",
    "project_urls": {
        "Homepage": "https://github.com/ModelCloud/Rockthem",
    },
    "keywords": ["rocm", "cuda", "gpu", "rockthem"],
    "platforms": ["linux", "windows", "darwin"],
    "classifiers": [
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
}

requirements = []
include_dirs = []
extensions = []

from distutils.sysconfig import get_python_lib

from torch.utils import cpp_extension as cpp_ext

conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")

print("conda_cuda_include_dir", conda_cuda_include_dir)
#if os.path.isdir(conda_cuda_include_dir):
include_dirs.append(conda_cuda_include_dir)
print(f"appending conda cuda include dir {conda_cuda_include_dir}")

extra_link_args = []
extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17",

    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__", # <-- NVCC
        "-U__CUDA_NO_HALF_CONVERSIONS__", # <-- NVCC
        # "-U__HIP_NO_HALF_OPERATORS__", # <-- ROCm/HIP FIX
        # "-U__HIP_NO_HALF_CONVERSIONS__", # <-- ROCm/HIP FIX
    ],
}

# torch >= 2.6.0 may require extensions to be build with CX11_ABI=1
CXX11_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

extra_compile_args["cxx"] += [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]
extra_compile_args["nvcc"] += [ f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}" ]

extensions = [
    cpp_ext.CUDAExtension(
        "rockthem_kernel",
        [
            "rock_kernel/q_gemm.cu",
        ],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
    ),
]

additional_setup_kwargs = {"ext_modules": extensions, "cmdclass": {"build_ext": cpp_ext.BuildExtension}}

setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require={},
    include_dirs=include_dirs,
    python_requires=">=3.9.0",
    cmdclass={"build_ext": cpp_ext.BuildExtension},
    ext_modules=extensions,
    license="Apache 2.0",
    **common_setup_kwargs
)