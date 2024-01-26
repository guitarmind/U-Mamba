import torch
import mamba_ssm
import glob
import os
from tqdm import tqdm
import shutil
import json
import sys
import numpy as np
import pandas as pd
import gc

import tifffile as tiff
import cv2

import argparse

sys.path.insert(0, "/kaggle/input/u-mamba/umamba")
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from scipy.spatial.qhull import QhullError
from scipy import spatial
spatial.QhullError = QhullError 

parser = argparse.ArgumentParser("Inferencer")
parser.add_argument("--model-folder", type=str, required=True,
                    help="path to model file")
parser.add_argument("--output-path", type=str, required=True,
                    help="path to save output logits")
parser.add_argument("--device", type=int, default=0, help="device ID")
# parser.add_argument("--threshold", type=float, default=0.001, help="logits threshold")
parser.add_argument("--checkpoint", type=str, default="best", help="best or final")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def main(args):
  model_folder = args.model_folder
  output_path = args.output_path
  threshold = args.threshold
  checkpoint_mode = args.checkpoint

  test_image_paths = glob.glob("/kaggle/input/blood-vessel-segmentation/test/*/images/*.tif")
  print(len(test_image_paths))

  test_dataset = []
  test_filename = []
  pred_image_paths = []
  for p in test_image_paths:
      tokens = p.split("/")
      test_dataset.append(tokens[-3])
      test_filename.append(tokens[-1].replace(".tif", ""))
      pred_image_paths.append([p])

  # instantiate the nnUNetPredictor
  predictor = nnUNetPredictor(
      tile_step_size=0.5,
      use_gaussian=True,
      use_mirroring=True,
      perform_everything_on_device=True,
      device=torch.device('cuda', int(args.device)),
      verbose=False,
      verbose_preprocessing=False,
      allow_tqdm=True
  )
  # initializes the network architecture, loads the checkpoint
  predictor.initialize_from_trained_model_folder(
      model_folder,
      # use_folds=(0,1,2,3,4),
      use_folds=(0,),
      # use_folds="all",
      checkpoint_name='checkpoint_best.pth' if checkpoint_mode == "best" else 'checkpoint_final.pth',
  )

  # variant 2.5, returns segmentations
  predicted_segmentations = predictor.predict_from_files(pred_image_paths, None,
                                   save_probabilities=True, overwrite=True,
                                   num_processes_preprocessing=2,
                                   num_processes_segmentation_export=2,
                                   folder_with_segs_from_prev_stage=None,
                                   num_parts=1, part_id=0)
  print("Inference finished:", len(predicted_segmentations))
  with open(output_path, "wb") as f:
      pickle.dump(predicted_segmentations, f, protocol=pickle.HIGHEST_PROTOCOL)
  

if __name__ == '__main__':
    main(args)
