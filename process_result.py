# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:03:08 2016

@author: joostbloom
"""

from ml_toolbox.kaggle import KaggleResult
from ml_toolbox.files import adjustfile_path
import time
import os

a = KaggleResult("sample_submission.csv")

print a.kag_name

print a.validate()

a.upload("Should fail")


















