"""Python setup.py bird classifier"""
import io
import os
from setuptools import find_packages, setup


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
    name="bridclassifier",
    version="0.0.0",
    description="Bird classifier",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Connor McShane",
    packages=find_packages(exclude=["tests"]),
    install_requires=read_requirements("requirements.txt")
)
