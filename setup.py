from setuptools import setup

setup(
    name='soft_sort_jax',
    version='0.1',
    packages=['soft_sort_jax', 'soft_sort_jax.third_party'],
    url='',
    license='',
    author='singletc',
    author_email='',
    description='',
    install_requires=[
        'numba',
        'numpy',
        'jax',
        'jaxlib',
        'scipy'
    ],
)
