from setuptools import setup, find_namespace_packages, find_packages

setup(
  name = 'jaxrk',
  packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
  version = '0.1.1',
  license='MIT',
  description = 'JaxRK is a library for working with (vectors of) RKHS elements and RKHS operators using JAX for automatic differentiation.',   # Give a short description about your library
  author = 'Ingmar Schuster',
  author_email = 'ingmar.schuster@zalando.de',
  url = 'https://github.com/zalandoresearch/jaxrk',
  download_url = 'https://github.com/zalandoresearch/jaxrk/v_01.tar.gz',
  keywords = ['Jax', 'RKHS', 'kernel'], 
  install_requires=['jax', 'numpy', 'scipy', 'matplotlib'],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)