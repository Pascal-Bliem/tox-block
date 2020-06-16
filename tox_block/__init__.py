import os
from tox_block.config import config

VERSION_PATH = os.path.join(config.PACKAGE_ROOT, "VERSION")

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()