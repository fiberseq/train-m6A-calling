#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = []

setup(
    author="Mitchell Robert Vollger",
    author_email="mrvollger@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="m6A calling from PacBio data using ML.",
    entry_points={
        "console_scripts": [
            "m6Acalling=m6a_calling.cli:main",
            "m6adata=m6a_calling.m6adata:main",
            "m6aXGBoost=m6a_calling.m6a_train_xgboost:main",
            "m6aMLdata=m6a_calling.m6amldata:main",
            "m6a_semi_supervised_cnn=m6a_calling.m6a_semi_supervised_cnn:main",
            "m6a_semi_supervised_cnn_predict=m6a_calling.m6a_semi_supervised_cnn_predict:main",
            "m6a_supervised_cnn=m6a_calling.m6a_supervised_cnn:main",
            "m6a_semi_supervised_cnn_gen=m6a_calling.m6a_semi_supervised_cnn_gen:main",
            "m6a_semi_supervised_cnn_predict_gen=m6a_calling.m6a_semi_supervised_cnn_predict_gen:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="m6a_calling",
    name="m6a_calling",
    packages=find_packages(include=["m6a_calling", "m6a_calling.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/mrvollger/m6a_calling",
    version="0.1.0",
    zip_safe=False,
)
