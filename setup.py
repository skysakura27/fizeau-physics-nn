from setuptools import setup, find_packages

setup(
    name="fizeau-physics-nn",
    version="0.1.0",
    description="Physics-Informed Neural Network for Fizeau Interferometry Phase Retrieval",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/skysakura27/fizeau-physics-nn",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "pyyaml>=6.0",
        "pytest>=7.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
