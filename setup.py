from setuptools import setup, find_packages

setup(
    name='ACTUARIAL_DF',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    description='A Python package for (Deep) Distributional Regression using PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'torch',
    ],
    extras_require={
        'test': ['pytest'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
