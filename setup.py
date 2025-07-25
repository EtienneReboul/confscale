#!/usr/bin/env python
import re
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup


def read(*names, **kwargs):
    with Path(__file__).parent.joinpath(*names).open(encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()


setup(
    name="confscale",
    use_scm_version={
        "local_scheme": "dirty-tag",
        "write_to": "src/confscale/_version.py",
        "fallback_version": "0.0.0",
    },
    license="LGPL-2.1-or-later",
    description="at scale generation of conformers with RDKit",
    long_description="{}\n{}".format(
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.rst")),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    author="Etienne Reboul",
    author_email="reboul@ibpc.fr",
    url="https://github.com/EtienneReboul/confscale",
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    py_modules=[path.stem for path in Path("src").glob("*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        # uncomment if you test on these interpreters:
        # "Programming Language :: Python :: Implementation :: IronPython",
        # "Programming Language :: Python :: Implementation :: Jython",
        # "Programming Language :: Python :: Implementation :: Stackless",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://confscale.readthedocs.io/",
        "Changelog": "https://confscale.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/EtienneReboul/confscale/issues",
    },
    keywords=[
        # eg: "keyword1", "keyword2", "keyword3",
    ],
    python_requires=">=3.9",
    install_requires=[
        "dask>=2024.1.0",
        "rdkit>=2023.9.1",
    ],
    extras_require={
        # eg:
        #   "rst": ["docutils>=0.11"],
        #   ":python_version=='3.8'": ["backports.zoneinfo"],
    },
    setup_requires=[
        "setuptools_scm>=3.3.1",
    ],
    entry_points={
        "console_scripts": [
            "confscale = confscale.cli:run",
        ]
    },
)
