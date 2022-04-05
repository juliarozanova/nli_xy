import os 
import sys

DIR_PATH = os.path.dirname(__file__)
HOME_PATH = os.path.join(DIR_PATH, '../')
AMNESIC_PATH = os.path.join(HOME_PATH, '../../amnesic_probing/')

sys.path.append(AMNESIC_PATH)
