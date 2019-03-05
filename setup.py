import setuptools

setuptools.setup(
    name="digideep",
    version="0.0.1",
    author="Mohammadreza Sharif",
    author_email="mrsharif@ece.neu.edu",
    description="A pipeline for fast prototyping Deep RL problems using PyTorch, Gym, and dm_control",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    python_requires='>=3.4.0',
    url="https://github.com/sharif1093/digideep",
    install_requires=[
        "colorama>=0.4.1", "psutil>=5.5.1", "pyyaml>=3.13"
    ],
    packages=setuptools.find_packages(),
    # packages=['digideep'],
    license="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
