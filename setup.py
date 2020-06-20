import io
import os
from pathlib import Path

from setuptools import find_packages, setup


# Package meta-data.
NAME = "tox_block"
DESCRIPTION = "This is a python package for classifying text in 6 categories of verbal toxicity using deep learning."
URL = "https://github.com/Pascal-Bliem/tox-block"
EMAIL = "pascal@bliem.de""
AUTHOR = "Pascal Bliem"
REQUIRES_PYTHON = ">=3.7.0"


# Packages that are required for this module to be executed
def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "tox_block"
about = {}
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


# The actual setup
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"tox_block": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Natural Language :: English",
        "Operating System :: OS Independent"
    ],
)
