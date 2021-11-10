from setuptools import setup, Extension

from pybind11.setup_helpers import Pybind11Extension

__version__ = "0.0.1"

ext_module = Pybind11Extension("pyme._C",
    ["src/pyme.cpp"],
    cxx_std=17,
)

# ext_module.extra_compile_args.extend(['-O0', '-g',])
# ext_module.extra_compile_args.extend(['-fsanitize=address'])
# ext_module.extra_link_args.extend(['-fsanitize=address'])

setup(
    name="pyme",
    version=__version__,
    author="胡玮文",
    author_email="huww98@outlook.com",
    url="https://github.com/huww98/pyme",
    description="Video motion estimation routines",
    long_description="",
    ext_modules=[ext_module],
    python_requires=">=3.6",
)
