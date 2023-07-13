#!/usr/bin/env python
import glob
import os
import shutil
from cgi import log
from distutils.cmd import Command
from setuptools import setup, find_packages


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        global here

        here = os.getcwd()
        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print(f"Cleaning {os.path.relpath(path)}")
                shutil.rmtree(path)



setup(name='foxutils',
      version='1.0',
      description='Foxelas utils for basic AI projects',
      author='github:foxelas',
      author_email='foxelas@outlook.com',
      url='https://github.com/foxelas/foxutils',
      packages=find_packages(include=['foxutils', 'foxutils.*']),
      classifiers=[
                    'Development Status :: 0 - Dev',
                    'Intended Audience :: Developers',
                    'Topic :: Software Development :: Build Tools',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3.7',
                    'Programming Language :: Python :: 3.8',
                    'Programming Language :: Python :: 3.9',
                ],
      project_urls={
                #'Documentation': 'https://packaging.python.org/tutorials/distributing-packages/',
                #'Funding': 'https://donate.pypi.org',
                #'Say Thanks!': 'http://saythanks.io/to/example',
                'Source': 'https://github.com/foxelas/foxutils',
      },
      cmdclass={
          'clean': CleanCommand,
      },
     )
