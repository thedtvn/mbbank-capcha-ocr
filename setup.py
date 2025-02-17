import re
from setuptools import setup

with open("requirements.txt") as f:
    req = f.read().splitlines()

with open("README.MD") as f:
    ldr = f.read()

with open('mb_capcha_ocr/__init__.py') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='mbbank-lib',
    version=version,
    license="Apache License, Version 2.0",
    description='An pytorch ocr base library for MBBank lib',
    long_description=ldr,
    long_description_content_type="text/markdown",
    url='https://github.com/thedtvn/mbbank-capcha-ocr',
    author='The DT',
    packages=["mb_capcha_ocr"],
    install_requires=req,
    include_package_data=True
)