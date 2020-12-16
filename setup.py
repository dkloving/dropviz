from setuptools import setup, find_packages


setup(name='dropviz',
      version='0.0.1',
      description='Visualize dropout as data augmentation.',
      author='David Loving',
      author_email='user@email.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      license='MIT',
      )
