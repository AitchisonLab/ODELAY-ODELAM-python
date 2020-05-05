# import setup tools
from setuptools import setup, find_packages

setup(
    name='OdelayTools',
    version ='0.1.0',
    author='Thurston Herricks', 
    description='ODELAY Tools and Image Pipeline',
    license='MIT',
    python_requires='>=3.7',
    py_modules=['odelay'],
    install_requires=[
        'ipython',
        'matplotlib',
        'numpy',
        'jinja2',
        'scipy',
        'paramiko',
        'pylint',
        'PyQt5',
        'PyQtChart',
        'h5py',
        'Click',
        'opencv-python',
        'opencv-contrib-python',
        'pandas',
        'xlrd',
        'sqlalchemy',
        'fast_histogram',
        'jupyterlab'
    ],
    packages=find_packages(),
    entry_points='''
    [console_scripts]
    odelay=odelay:cli
    ''',
)
