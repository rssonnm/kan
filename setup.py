"""
KAN - Kolmogorov-Arnold Networks

A PyTorch implementation based on https://arxiv.org/abs/2404.19756
"""

from setuptools import setup, find_packages

setup(
    name="kan",
    version="1.0.0",
    author="KAN Implementation",
    description="Kolmogorov-Arnold Networks - PyTorch implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/kan",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "symbolic": ["sympy>=1.10.0"],
        "dev": ["pytest", "black", "isort"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning, neural-networks, kolmogorov-arnold, splines",
)
