"""Setup configuration for hiring-sentiment-tracker."""
from setuptools import setup, find_packages

setup(
    name="hiring-sentiment-tracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
        "rich>=13.7.0",
        "psutil>=5.9.6"
    ]
)
