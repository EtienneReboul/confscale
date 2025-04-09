========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |github-actions| |coveralls| |codecov|
    * - package
      - |version| |wheel| |supported-versions| |supported-implementations| |commits-since|
.. |docs| image:: https://readthedocs.org/projects/confscale/badge/?style=flat
    :target: https://readthedocs.org/projects/confscale/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/EtienneReboul/confscale/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/EtienneReboul/confscale/actions

.. |coveralls| image:: https://coveralls.io/repos/github/EtienneReboul/confscale/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://coveralls.io/github/EtienneReboul/confscale?branch=main

.. |codecov| image:: https://codecov.io/gh/EtienneReboul/confscale/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/EtienneReboul/confscale

.. |version| image:: https://img.shields.io/pypi/v/confscale.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/confscale

.. |wheel| image:: https://img.shields.io/pypi/wheel/confscale.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/confscale

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/confscale.svg
    :alt: Supported versions
    :target: https://pypi.org/project/confscale

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/confscale.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/confscale

.. |commits-since| image:: https://img.shields.io/github/commits-since/EtienneReboul/confscale/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/EtienneReboul/confscale/compare/v0.0.0...main



.. end-badges

at scale generation of conformers with RDKit

* Free software: GNU Lesser General Public License v2.1 or later (LGPLv2+)

Installation
============

::

    pip install confscale

You can also install the in-development version with::

    pip install https://github.com/EtienneReboul/confscale/archive/main.zip


Documentation
=============


https://confscale.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
