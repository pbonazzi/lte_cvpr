from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='difflogic',
    ext_modules=[
        CUDAExtension(
            name='difflogic_cuda',
            sources=['difflogic/cuda/difflogic.cpp',
                     'difflogic/cuda/difflogic_kernel.cu'],
            extra_compile_args={'nvcc': ['-lineinfo']}),
        CUDAExtension(
            name='conv_difflogic_cuda',
            sources=['difflogic/cuda/conv_difflogic.cpp',
                     'difflogic/cuda/conv_difflogic_kernel.cu'],
            extra_compile_args={'nvcc': ['-lineinfo']})],
    cmdclass={'build_ext': BuildExtension}
)
