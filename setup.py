from distutils.core import setup
  

setup(name='dinobot',
      version='1.0.0',
      install_requires=[
            'pybullet',
      ],
      packages=['dinobot', 'dinobot.pybullet_tools', 'dinobot.pybullet_tools.ikfast', 'dinobot.env'],
     )


from setuptools import setup

setup()
