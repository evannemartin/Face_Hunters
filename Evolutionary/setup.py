from setuptools import find_packages, setup
setup(
    name='evolutionary',
    packages=find_packages(include=['evolutionary']),
    version='0.1.0',
    description='Library containing functions related to evolutionary algorithms',
    author='us',
    license='MIT',
    install_requires=['numpy']
)
