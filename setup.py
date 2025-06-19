from setuptools import setup, find_packages

setup(
    name='zhn_T9_model',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    author='weiqizhang',
    description='T9 pinyin-to-hanzi prediction system',
    python_requires='>=3.7',
)
