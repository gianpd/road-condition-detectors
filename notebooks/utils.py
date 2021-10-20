import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import PIL

def build_dataset(data_dir: Path, subset: str, img_size: Tuple):
  return tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=.20,
      subset=subset,
      label_mode="categorical",
      # Seed needs to provided when using validation_split and shuffle = True.
      # A fixed seed is used so that the validation set is stable across runs.
      seed=123,
      image_size=img_size,
      batch_size=1)

def get_augmentation_model(normalization_layer: tf.keras.layers) -> tf.keras.Sequential:
    preprocessing_model = tf.keras.Sequential([normalization_layer])
    preprocessing_model.add(
        tf.keras.layers.RandomRotation(40))
    preprocessing_model.add(
        tf.keras.layers.RandomTranslation(0, 0.2))
    preprocessing_model.add(
        tf.keras.layers.RandomTranslation(0.2, 0))
    # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
    # image sizes are fixed when reading, and then a random zoom is applied.
    # If all training inputs are larger than image_size, one could also use
    # RandomCrop with a batch size of 1 and rebatch later.
    preprocessing_model.add(
       tf.keras.layers.RandomZoom(0.2, 0.2))
    preprocessing_model.add(
       tf.keras.layers.RandomFlip(mode="horizontal"))
    return preprocessing_model



