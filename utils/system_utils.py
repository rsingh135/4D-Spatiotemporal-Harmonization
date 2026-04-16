#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = []
    for fname in os.listdir(folder):
        # Only accept entries like iteration_3000 (ignore .ply, .pth, etc.)
        if not fname.startswith("iteration_"):
            continue
        suffix = fname.split("_")[-1]
        if not suffix.isdigit():
            continue
        saved_iters.append(int(suffix))

    if not saved_iters:
        raise FileNotFoundError(
            f"No iteration_* entries found in '{folder}'. "
            "Expected entries like 'iteration_3000'."
        )

    return max(saved_iters)
