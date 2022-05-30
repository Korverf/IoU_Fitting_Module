import os.path as osp
from setuptools import setup, Extension

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension
import torch
import os


ext_args = dict(
    include_dirs=[np.get_include()],
    language='c++',
    extra_compile_args={
        'cc': ['-Wno-unused-function', '-Wno-write-strings'],
        'nvcc': ['-c', '--compiler-options', '-fPIC'],
    },
)


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to cc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if osp.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', 'nvcc')
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['cc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


name='convex_ext'
sources = ['src/convex_cpu.cpp',
           'src/convex_ext.cpp']
sources_cuda = ['src/convex_cuda.cu']

define_macros = []
extra_compile_args = {'cxx': []}

if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
    define_macros += [('WITH_CUDA', None)]
    extension = CUDAExtension
    extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
    sources += sources_cuda
else:
    print(f'Compiling {name} without CUDA')
    extension = CppExtension
setup(
    name=name,
    ext_modules=[
        extension(name, sources),
    ],
    cmdclass={'build_ext': BuildExtension})
