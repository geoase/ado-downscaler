#!/usr/bin/env python
import setuptools

from distutils.core import setup

setup(name='ado_downscaler',
      version='0.1',
      description='Alpine Drought Observatory Downscaler',
      license='MIT',
      author='Georg A. Seyerl',
      author_email='g.seyerl@geoase.eu',
      packages=setuptools.find_packages(),
      keywords=['climate', 'quantile mapping', ],
      # tests_require=['pytest'],
     )
