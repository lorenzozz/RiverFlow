import os

"""
Variables declared in this file are globally accessible throughout the
file sections.
"""

SORCEROOT = os.getcwd().replace('\\', '/')
EXAMPLESROOT = '/'.join(SORCEROOT.split('/')[:-1]) + '/examples'
URLROOT = os.getcwd().replace('\\', '/')
RIVERDATAROOT = '/'.join(SORCEROOT.split('/')[:-1]) + '/RiverData'
