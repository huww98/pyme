from setuptools import setup, Extension

__version__ = "0.0.1"

ext_modules = [
    Extension("pyme._C",
        ["src/pyme.cpp"],
    ),
]

setup(
    name="pyme",
    version=__version__,
    author="胡玮文",
    author_email="huww98@outlook.com",
    url="https://github.com/huww98/pyme",
    description="Video motion estimation routines",
    long_description="",
    ext_modules=ext_modules,
    python_requires=">=3.6",
)
