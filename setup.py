from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="redbirdpy",
    packages=find_packages(exclude=["test", "test.*"]),
    version="0.2.0",
    license="GPL-3.0",
    description="A Python toolbox for Diffuse Optical Tomography (DOT) and Near-Infrared Spectroscopy (NIRS)",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Qianqian Fang",
    author_email="fangqq@gmail.com",
    maintainer="Qianqian Fang",
    url="https://github.com/fangq/redbirdpy",
    project_urls={
        "Bug Tracker": "https://github.com/fangq/redbirdpy/issues",
        "Documentation": "https://github.com/fangq/redbirdpy#readme",
        "Source Code": "https://github.com/fangq/redbirdpy",
    },
    keywords=[
        "Diffuse Optical Tomography",
        "DOT",
        "NIRS",
        "Near-Infrared Spectroscopy",
        "FEM",
        "Finite Element Method",
        "Biomedical Optics",
        "Image Reconstruction",
        "Inverse Problem",
        "Photon Migration",
        "Diffusion Equation",
        "Tissue Optics",
    ],
    platforms="any",
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.15.0",
        "scipy>=1.0.0",
        "iso2mesh>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.0",
        ],
        "all": [
            "matplotlib>=3.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            # Add CLI tools here if needed
            # "redbirdpy=redbirdpy.cli:main",
        ],
    },
)
