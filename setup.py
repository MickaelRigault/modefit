#! /usr/bin/env python
#

DESCRIPTION = "modefit: Simple minuit<->mcmc fitting tools."
LONG_DESCRIPTION = """ Simple minuit<->mcmc fitting tools. """

DISTNAME = 'modefit'
AUTHOR = 'Mickael Rigault'
MAINTAINER = 'Mickael Rigault' 
MAINTAINER_EMAIL = 'mrigault@physik.hu-berlin.de'
URL = 'https://github.com/MickaelRigault/modefit/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/MickaelRigault/modefit/tarball/0.4'
VERSION = '0.4.0'

try:
    from setuptools import setup, find_packages
    _has_setuptools = True
except ImportError:
    from distutils.core import setup
    _has_setuptools = False

if __name__ == "__main__":

    if _has_setuptools:
        packages = find_packages()
        print(packages)
    else:
        # This should be updated if new submodules are added
        packages = ['modefit']

    setup(name=DISTNAME,
          author=AUTHOR,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=[
                "corner",
                "iminuit>=2.0.0",
                "matplotlib",
                "numpy>=1.21.6",
                "propobject>=0.1.3",
                "scipy>=0.16.0",
          ],
          extra_requires=[
                "emcee>=2.0.0",
          ],
          packages=packages,
          package_data={'modefit': []},
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.5',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering :: Astronomy',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'],
      )
