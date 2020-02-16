from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).absolute().parent

long_description = (here / Path('README.md')).read_text()

_version = {}
exec((here / Path('udacity_rl/_version.py')).read_text(), _version)

setup(
    name='udacity_rl',
    version=_version['__version__'],
    description='RL time series testbed',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SwamyDev/udacity-deep-rl-navigation',
    author='Bernhard Raml',
    packages=find_packages(include=['udacity_rl', 'udacity_rl.*']),
    include_package_data=True,
    zip_safe=False,
    install_requires=['udacity-unityagents', 'click'],
    extras_require={"test": ['pytest', 'pytest-cov', 'pytest-rerunfailures==7', 'gym-quickcheck']},
    python_requires='>=3.6,<3.8',
    entry_points={
        'console_scripts': ['p1_navigation = udacity_rl.navigation:cli']
    }
)
