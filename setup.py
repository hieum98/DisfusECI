"""Utilities for setuptools integration."""
import logging
import os
from typing import List, Tuple

from setuptools import find_packages, setup


LOG = logging.getLogger(__name__)


def read(rel_path: str) -> str:
    """Read text from a file.

    Based on https://github.com/pypa/pip/blob/main/setup.py#L7.

    Args:
        rel_path (str): Relative path to the target file.

    Returns:
        str: Text from the file.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    """Read the version number from the top-level __init__.py.

    Based on https://github.com/pypa/pip/blob/main/setup.py#L15.

    Args:
        rel_path (str): Path to the top-level __init__.py.

    Raises:
        RuntimeError: Failed to read the version number.

    Returns:
        str: The version number.
    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def requirements(rel_path: str) -> Tuple[List[str], List[str]]:
    """Parse pip-formatted requirements file.

    Args:
        rel_path (str): Path to a requirements file.

    Returns:
        Tuple[List[str], List[str]]: Extra package index URLs and setuptools-compatible package specifications.
    """
    packages = read(rel_path).splitlines()
    result = []
    dependency_links = []
    for pkg in packages:
        if pkg.startswith("--extra-index-url"):
            dependency_links.append(pkg.split(" ")[-1])
            continue
        if pkg.strip().startswith("#") or not pkg.strip():
            continue
        result.append(pkg)
    return dependency_links, result


setup(
    name="DiffusIE",
    packages=find_packages(exclude=["scripts"]),
    version='1.0.0',
    description="Implementation of Diffusion for IE",
    author="Hieu Man",
    author_email="hieum@uoregon.edu",
)
