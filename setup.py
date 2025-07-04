from setuptools import setup, find_packages

setup(
    name='research_tools',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'click',
    ],
    entry_points={
        'console_scripts': [
            'research_tools=research_tools.cli:cli',
        ],
    },
)