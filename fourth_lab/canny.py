import sys
import os

current_dir = os.path.dirname(__file__)

third_lab_path = os.path.join(current_dir, "..", "third_lab")
sys.path.append(third_lab_path)

from gauss_methods import *
