import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mingrad",
    version="0.1.0",
    author="Arnav Samal",
    author_email="samalarnav@gmail.com",
    description="a lightweight autograd engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnavs04/mingrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)