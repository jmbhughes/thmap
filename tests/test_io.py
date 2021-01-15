from unittest import TestCase
from thmap.io import Image, ImageSet, ThematicMap
from datetime import datetime
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestImageSet(TestCase):
    def test_empty_creation(self):
        imgs = ImageSet.create_empty()
        self.assertTrue(isinstance(imgs, ImageSet))
        keys = ['94', '131', '171', '195', '284', '304']
        self.assertListEqual(list(imgs.images.keys()), keys)

    def test_retrieve(self):
        date = datetime(2021, 1, 1, 6, 0)
        imgs = ImageSet.retrieve(date)
        self.assertTrue(isinstance(imgs, ImageSet))
        keys = ['94', '131', '171', '195', '284', '304']
        self.assertListEqual(list(imgs.images.keys()), keys)
        self.assertTrue(isinstance(imgs['94'], Image))


class TestThematicMap(TestCase):
    def test_init(self):
        data = np.zeros((1280, 1280))
        metadata = {'DATE-OBS': "2021-1-1T06:00"}
        theme_mapping = {'outer_space': 1,
                         'bright_region': 3,
                         'filament': 4,
                         'prominence': 5,
                         'coronal_hole': 6,
                         'quiet_sun': 7,
                         'limb': 8,
                         'flare': 9}
        thmap = ThematicMap(data, metadata, theme_mapping)
        self.assertTrue(isinstance(thmap, ThematicMap))

    def test_load(self):
        path = os.path.join(THIS_DIR, "test_thmap.fits")
        thmap = ThematicMap.load(path)
        self.assertTrue(isinstance(thmap, ThematicMap))

