"""
Setup script for Italian Text Processing Library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="italian-text-processing",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple yet powerful Italian text processing library using NLTK and spaCy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/italian-text-processing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Italian",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "italian-nlp-demo=examples.interactive_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
