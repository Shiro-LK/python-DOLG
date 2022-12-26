# -*- coding: utf-8 -*-
from setuptools import setup
from io import open

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='dolg',
    version='0.1.1',    
    description='Re-implementation of DOLG paper in torch and tensorflow with converted checkpoints',
    author='Shiro-LK',
    author_email='shirosaki94@gmail.com',
    license='MIT License',
    packages=['dolg', "dolg.utils", "dolg.tests"],
    long_description=readme(),
    long_description_content_type="text/markdown",
    install_requires=['numpy', 
                      "pytest"
                      ],
    url='https://github.com/Shiro-LK/python-DOLG',
    download_url='https://github.com/Shiro-LK/python-DOLG.git',
    keywords=["DOLG for torch and tensorflow", "pretrained weights", "tensorflow", "tf", "pytorch", "torch"],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
