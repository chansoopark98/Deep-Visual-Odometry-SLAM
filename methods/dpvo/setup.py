import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

# CUDA architecture flags for FP16 support
# RTX 5090 = Blackwell (sm_120, Compute Capability 12.0)
# Requires CUDA 12.8+
NVCC_FLAGS = [
    '-O3',
    '-Wno-deprecated-gpu-targets',
    '-gencode=arch=compute_75,code=sm_75',   # Turing (RTX 20xx)
    '-gencode=arch=compute_80,code=sm_80',   # Ampere (A100)
    '-gencode=arch=compute_86,code=sm_86',   # Ampere (RTX 30xx)
    '-gencode=arch=compute_89,code=sm_89',   # Ada Lovelace (RTX 40xx)
    '-gencode=arch=compute_120,code=sm_120', # Blackwell (RTX 50xx)
]

setup(
    name='dpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': NVCC_FLAGS,
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu', 'dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': NVCC_FLAGS,
            },
            include_dirs=[
                osp.join(ROOT, '../../modules/eigen-3.4.0')]
            ),
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'dpvo/lietorch/include'),
                osp.join(ROOT, '../../modules/eigen-3.4.0')],
            sources=[
                'dpvo/lietorch/src/lietorch.cpp',
                'dpvo/lietorch/src/lietorch_gpu.cu',
                'dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': NVCC_FLAGS}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
