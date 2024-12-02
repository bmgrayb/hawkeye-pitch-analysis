from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="pitch_path",
    version='1.0',
    description='Python package used to calculate clusters for pitching arm paths',
    author='Brandon Graybeal',
    author_email='bmgrayb@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)