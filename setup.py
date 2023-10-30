"""Python setup.py bird classifier"""
import io
import os
from setuptools import find_packages, setup

__version__ = '0.0.0'

def read(*paths, **kwargs):
    """Read the contents of a text file safely.

    Args:
        *paths (str): Paths to the file to read.
        **kwargs (str): Encoding to use when reading the file.
    
    Returns:
        str: Contents of the file
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    """Read the requirements file.

    Args:
        path (str): Path to the requirements file
    
    Returns:
        list: List of requirements
    """

    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


# Run setup
setup(
    name="bird_classifier",
    version=__version__,
    author="Connor McShane",
    description="This is app serves a model from tensorflow hub through an API. It is a basic inference app complete with logging, monitoring and testing.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/ConnorMcShane/bird-classifier-app",
    packages=find_packages(exclude=["tests"]),
    install_requires=read_requirements("requirements.txt"),
    classifiers=(
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Natural Language :: English',
    ),
)
