from __future__ import annotations
import os
from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
from typing import List
from .io import ThematicMap, ImageSet
from typing import Dict, Optional


class Classifier:
    """
    An abstract representation of a classifier
    """
    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        """
        generic initialization
        """
        self.kind: str = 'abstract'  # this gets replaced for each classifier type to indicate what kind it is

        # these get set up training
        self.channel_order: Optional[List[str]] = None
        self.theme_index: Optional[Dict[int, str]] = None
        self.is_trained: bool = False
        self.images: Optional[List[ImageSet]] = None

        self.dtype = np.uint8  # what data type the output thematic maps will be

    @staticmethod
    def _load_composites(thematic_map: ThematicMap) -> ImageSet:
        """
        Fetch the composites associated with a specific thematic map
        :param thematic_map:
        :return:
        """
        return ImageSet.retrieve(thematic_map.date_obs)

    @abstractmethod
    def train(self, thematic_maps: List[ThematicMap], image_sets: Optional[List[ImageSet]] = None) -> None:
        """
        Fit the classifier to some example data.
        Every train method must set is_trained to true
        :param thematic_maps: a list of thematic maps to train from
        :param image_sets: if specified, will use these instead of fetching the associated thematic maps
        """
        if image_sets is not None:
            if len(image_sets) != len(thematic_maps):
                raise RuntimeError("There must be as many image sets provided as thematic maps.")
            self.images = image_sets
        else:
            self.images = []
            for thematic_map in thematic_maps:
                self.images.append(Classifier._load_composites(thematic_map))

    @staticmethod
    @abstractmethod
    def load(path: str) -> Classifier:
        """
        load the classifier from a file
        :param path: path to load from
        :return: nothing, initializes object
        """
        if not os.path.isfile(path):
            msg = "Cannot load from {} because file does not exist"
            raise OSError(msg.format(path))
        return Classifier()

    @abstractmethod
    def save(self, path: str) -> None:
        """
        write the classifier out to a file
        :param path: path to save at
        """
        if not os.path.isdir(os.path.dirname(path)):
            msg = "Cannot save to {} because directory does not exist"
            raise OSError(msg.format(path))

    @abstractmethod
    def classify(self, images: ImageSet) -> ThematicMap:
        """
        Classify an image using the classifier
        :param images: a set of input images as set
        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not yet been trained.")
        return ThematicMap.create_empty()
