import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyNLControl",
    version="0.0.14",
    author="Niranjan Bhujel",
    author_email="niranjan.bhujel2014@gmail.com",
    description="Package for non-linear control and estimation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NiranjanBhujel/pyNLControl",
    project_urls={
        "Bug Tracker": "https://github.com/NiranjanBhujel/pyNLControl/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={
        "": ["templates/*.j2", "external/*.zip"],
    },
    python_requires=">=3.6",
    install_requires=[
        "casadi",
        "jinja2",
    ]
)
