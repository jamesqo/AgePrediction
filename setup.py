import os
from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='age_prediction',
      version='0.1',
      packages=['age_prediction'],
      install_requires=reqs)
