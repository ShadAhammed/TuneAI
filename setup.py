"""Package setup for TuneAI.

Installs the src package so that its sub-packages (DataExp, models,
visualization) can be imported cleanly from any working directory.
"""

from setuptools import find_packages, setup

setup(
    name='tuneai',
    version='1.0.0',
    description='Automated ML model comparison and hyperparameter tuning for any tabular dataset',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Abu Shad Ahammed',
    author_email='abu.ahammed@uni-siegen.de',
    license='BSD-3-Clause',
    packages=find_packages(exclude=['data', 'data.*', 'docs', 'docs.*']),
    python_requires='>=3.9',
    install_requires=[
        'tensorflow~=2.13.0',
        'keras-tuner~=1.4.6',
        'scikit-learn~=1.3.2',
        'xgboost~=1.7.6',
        'pandas~=2.0.3',
        'numpy~=1.24.4',
        'openpyxl~=3.1.2',
        'matplotlib~=3.7.3',
        'tabulate~=0.9.0',
        'python-dotenv>=0.5.1',
    ],
    entry_points={
        'console_scripts': [
            'tuneai=run:main',
        ],
    },
)
