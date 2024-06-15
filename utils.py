import numpy as np
import pandas as pd
import pickle as pkl
import os
from patchOTDA.datasets import MMS_DATA

EXAMPLE_DATA_ = ['Query1', 'Query2', 'Query3', 'CTKE_M1', 'VISp_Viewer']

REF_DATA_ = ['CTKE_M1', 'VISp_Viewer']


#build nodes for label propagation
nodes = []