import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="macrograd",
    version="0.0.1",
    author="Abdullah Palaz",
    author_email="palazski@gmail.com",
    description="A tiny Tensor-based Autograd engine with a PyTorch like API on top of it.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/palazski/macrograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
