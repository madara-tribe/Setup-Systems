import os, sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'trainer'

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
     name=PACKAGE_NAME,
      version='1.0.1',
      description='GCP system pyhthon pacage',
      keywords ='systems'
      )
