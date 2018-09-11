# -*- coding: utf-8 -*-

"""Top-level package for easyimages."""

__author__ = """Jakub Cieslik"""
__email__ = 'kubacieslik@gmail.com'


from .utils import get_execution_context
from .easyimages import EasyImage, EasyImageList, bbox
from .logger import logger


logger.info("[info][easyimages] is running in {} context".format(get_execution_context()))
