import os

try:
    from numpy.distutils.core import Extension, setup
except ImportError:
    raise ImportError("NumPy is not installed. Please install NumPy manually before.")

# get text of README.md
current_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_path, "README.md")) as f:
    readme_text = f.read()

setup(
    name="UmbrellaIntegrator",
    version='0.6.1',
    description="Umbrella Integration of PMF calculations - 1D & 2D ",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="https://github.com/boneta/UmbrellaIntegrator",
    author="Sergio Boneta",
    author_email="boneta@unizar.es",
    license="GPLv3",
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
        ],
    setup_requires=["numpy"],
    install_requires=[
        "numpy",
        "scipy"
        ],
    ext_modules=[Extension(name="umbrellaint_fortran",
                           sources=["umbrellaint_fortran.f90"],
                           extra_compile_args=["-O3", "-fopenmp"],
                           extra_link_args=["-lgomp"])
        ],
    py_modules=["umbrellaint"],
    include_package_data=True,
    entry_points={
        "console_scripts": ["umbrellaint=umbrellaint:main"]
        },
    )
