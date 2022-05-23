import setuptools
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'pyomo>=3.5',
    'numpy>=1.21',
    'pandas>=1.3',
    'geopy>=2.2',
    'matplotlib>=3.5',
    'scipy>=1.7',
    'folium>=0.12',
    'branca>=0.4',
    'SRTM.py>=0.3',
    'meteostat>=1.5',
    'partridge>=1.1',
    'tqdm>=4.64'
]

if sys.version_info < (3, 7):
    install_requires.append('dataclasses')

# tests_require = [
#     "pytest",
#     "pytest-timeout",
#     "hypothesis[numpy]"
# ]

setuptools.setup(
    name="RouteZero",
    version="0.0.5",
    description="Electric bus energy usage prediction and depot charging feasibility. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=setuptools.find_packages(),
    classifiers=[
    ],
    install_requires=install_requires,
    python_requires='>=3.7',
    # extras_require={
    #     "validation": ["mypy"],
    #     "test": tests_require,
    # },
    setup_requires=['pytest-runner'],
    # tests_require=tests_require
)
