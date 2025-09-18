# dataset_and_training.py
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_and_restructure_dataset(input_json, image_dir, output_dir, val_split=0.2):
    # (Code for dataset preparation, same as before)
    ...

def train_yolo_model(dataset_dir, model_weights='yolov8x.pt', epochs=10):
    # (Code for training YOLO, same as before)
    ...
