from setuptools import setup, find_packages

setup(name='gym-commons-game',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym', 'pycolab', 'matplotlib', 'numpy', 'scipy', 'torch', 'torchvision', 'torchaudio', 'pandas']
)