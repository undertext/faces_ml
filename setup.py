from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['tensorflow_addons', 'python-dotenv']

setup(
    name="faces_ml",
    version="0.1.0",
    author="Yaroslav Kharchenko",
    install_requires=REQUIRED_PACKAGES,
    description="Faces recognition using Tensor Flow",
    license="BSD",
    keywords="faces ml",
    packages=find_packages(),
)
