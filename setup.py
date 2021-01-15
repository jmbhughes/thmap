from setuptools import setup

setup(
    name='thmap',
    version='0.0.1',
    packages=['thmap'],
    url='',
    license='',
    author='J. Marcus Hughes',
    author_email='j-marcus.hughes@noaa.gov',
    description='A solar thematic map manipulation class',
    test_suite='nose.collector',
    tests_require=['pytest'],
    install_requires=['scikit-learn',
                      'numpy',
                      'deepdish',
                      'astropy',
                      'goes-solar-retriever']
)
