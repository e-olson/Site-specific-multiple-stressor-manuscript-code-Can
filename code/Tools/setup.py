import setuptools
  
def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(
    name='Tools',
    version = '0.1',
    description = 'Tools package for common functions',
    long_description=readme(),
    url='none',
    author='Elise Olson',
    author_email='eo2651@princeton.edu',
    license='none',
    package_dir={"": "Tools"},
    packages=setuptools.find_packages(where="Tools"),
    install_requires=['numpy','matplotlib'],
    zip_safe=False)
