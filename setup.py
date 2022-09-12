from setuptools import setup, find_packages

setup(
    name='stainlib',
    version='0.6.1',
    description='A library for H&E image stain normalization and augmentation.',
    author='Sebastian Otálora',
    author_email='sebastianffx@gmail.com',
    url='https://github.com/sebastianffx/stainlib',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'opencv-python',
        'scikit-image',
        'pillow',
        'spams-bin'
    ]
)
