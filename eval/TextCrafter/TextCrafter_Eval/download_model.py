#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

import t2v_metrics
# Get cache directory (if set)
cache_dir = 'your_model_path'
model = t2v_metrics.VQAScore(model='clip-flant5-xxl', cache_dir=cache_dir)
print(f'finish_download')