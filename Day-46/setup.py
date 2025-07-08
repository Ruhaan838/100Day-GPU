from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='muon_extension',
    ext_modules=[
        CUDAExtension(
            name='muon_extension',
            sources=['Muon.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
