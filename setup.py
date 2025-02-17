import re
from setuptools import setup

with open("requirements.txt") as f:
    req = f.read().splitlines()

with open("README.md") as f:
    ldr = f.read()

with open('mb_capcha_ocr/__init__.py') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='mb_capcha_ocr',
    version=version,
    license="MIT",
    description='An pytorch ocr base library for MBBank lib',
    long_description=ldr,
    long_description_content_type="text/markdown",
    url='https://github.com/thedtvn/mbbank-capcha-ocr',
    author='The DT',
    packages=["mb_capcha_ocr"],
    package_data={
        'mb_capcha_ocr': ['model.pt'],
    },
    install_requires=req,
    include_package_data=True
)