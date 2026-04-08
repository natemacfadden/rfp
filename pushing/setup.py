from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

setup(
    ext_modules=cythonize(
        [Extension(
            "pushing.pushing",
            sources=["pushing/pushing.pyx"],
            include_dirs=["src"],
            define_macros=[("PUSHING_IMPLEMENTATION", None)],
            extra_compile_args=["-O3"],
            language="c",
        )],
        compiler_directives={"language_level": "3"},
    )
)
