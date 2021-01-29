from __future__ import annotations
import os
from abc import ABC, abstractmethod
import warnings
import numpy as np
from typing import List
from .io import ThematicMap, ImageSet, NUMBER_OF_CHANNELS
from typing import Dict, Optional, List
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
import deepdish as dd
from sklearn.tree import tree, DecisionTreeClassifier
from sklearn.base import clone


class PixelClassifier(ABC):
    """
    An abstract representation of a classifier
    """

    def __init__(self) -> None:
        """
        generic initialization
        """
        self.kind: str = 'abstract pixel'  # this gets replaced for each classifier type to indicate what kind it is

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
        if len(thematic_maps) == 0:
            raise RuntimeError("At least one thematic map must be specified.")

        if image_sets is not None:
            if len(image_sets) != len(thematic_maps):
                raise RuntimeError("There must be as many image sets provided as thematic maps.")
            self.images = image_sets
        else:
            self.images = []
            for thematic_map in thematic_maps:
                self.images.append(PixelClassifier._load_composites(thematic_map))

    @staticmethod
    @abstractmethod
    def load(path: str) -> None:
        """
        load the classifier from a file
        :param path: path to load from
        :return: nothing, initializes object
        """
        if not os.path.isfile(path):
            msg = "Cannot load from {} because file does not exist"
            raise OSError(msg.format(path))

    @abstractmethod
    def save(self, path: str) -> None:
        """
        write the classifier out to a file
        :param path: path to save at
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving.")

        if not os.path.isdir(os.path.dirname(path)):
            msg = "Cannot save to {} because directory does not exist"
            raise OSError(msg.format(path))

    @abstractmethod
    def save_for_spades(self, path: str) -> None:
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


class ConcentricPixelClassifier(PixelClassifier):

    def __init__(self, solar_indices: Dict[str, int],
                 themes_in_radii: List[List[str]],
                 radii: Optional[List[int]] = None) -> None:
        """
        generic initialization
        """
        super().__init__()
        # this gets replaced for each classifier type to indicate what kind it is
        self.kind: str = 'abstract concentric pixel'

        self.models: dict = dict()
        self.image_size: int = 1280  # this is specific for SUVI implementations, assumes a square image

        self.solar_indices: Dict[str, int] = solar_indices
        self.radii: Optional[List[int]] = radii
        self.mask: np.ndarray = self._create_mask()
        self.themes_in_radii: List[List[str]] = themes_in_radii
        if self.radii is not None:
            if len(self.radii) + 1 != len(themes_in_radii):
                raise RuntimeError("There is a mismatch between the number of radii and the defined themes for radii.")
        else:  # self.radii is None
            if len(themes_in_radii) != 1:
                raise RuntimeError("There is a mismatch between the number of radii and the defined themes for radii.")
        self.pixel_db = None  # will be setup during training

    def _create_mask(self):
        """
        Creates a mask for the concentric rings of the classifier models
        :return: starting at 1 each pixel is assigned an integer for its concentric ring
        """
        if self.radii is None:
            return np.ones((self.image_size, self.image_size))
        else:  # radii are specified
            # Define image center
            center = (self.image_size / 2) - 0.5

            # Create mesh grid of image coordinates
            xm, ym = np.meshgrid(np.linspace(0, self.image_size - 1, num=self.image_size),
                                 np.linspace(0, self.image_size - 1, num=self.image_size))

            # Center each mesh grid (zero at the center)
            xm_c = xm - center
            ym_c = ym - center

            # Create array of radii
            rad = np.sqrt(xm_c ** 2 + ym_c ** 2)

            # Define radii bounds
            radii_new = self.radii.copy()
            radii_new.append(np.amax(rad))
            radii_new.insert(0, 0)

            # Create empty mask of same size as the image
            mask = np.zeros((self.image_size, self.image_size))

            # Iterate through radii
            for i in range(len(radii_new) - 1):
                # Find where in the mask radii are between certain bounds
                mask[((rad > radii_new[i]) & (rad <= radii_new[i + 1]))] = i + 1

            return mask

    def _define_pixel_database(self, thmaps: List[ThematicMap], counts_per_theme=2000) \
            -> Dict[int, Dict[str, np.ndarray]]:
        # setup the blank and empty pixel database
        pix_db = dict()
        # Iterate through number of layers in mask
        for i in range(len(self.radii) + 1):
            # Make temporary dictionary
            themes_temp = self.themes_in_radii[i]
            # Create temporary dictionary
            dict_theme = dict()
            # Iterate through themes in layer
            for theme_temp in themes_temp:
                # Create dictionary entry for each theme with empty array
                dict_theme[theme_temp] = np.empty((0, NUMBER_OF_CHANNELS))
            # Create entry in pixel database
            pix_db[i + 1] = dict_theme

        # populate it with values from thematic map image sets
        for i, thmap in enumerate(thmaps):
            for layer in pix_db:
                layer_mask = (self.mask == layer)
                image_set = self.images[i]
                cube = image_set.cube()
                for theme in self.themes_in_radii[layer - 1]:
                    theme_mask = layer_mask * (thmap.data == self.solar_indices[theme])
                    theme_pixels = cube[theme_mask, :]
                    pix_db[layer][theme] = np.concatenate([pix_db[layer][theme], theme_pixels])

        # Resample it to the number of counts per theme requested
        for layer in pix_db:
            for theme in pix_db[layer]:
                indices = np.random.choice(range(pix_db[layer][theme].shape[0]), counts_per_theme)
                pix_db[layer][theme] = pix_db[layer][theme][indices]
        return pix_db

    @abstractmethod
    def train(self, thematic_maps: List[ThematicMap], image_sets: Optional[List[ImageSet]] = None,
              counts_per_theme: int = 2000) -> None:
        super().train(thematic_maps, image_sets)
        self.pixel_db = self._define_pixel_database(thematic_maps, counts_per_theme=counts_per_theme)

    @staticmethod
    @abstractmethod
    def load(path: str) -> None:
        super(ConcentricPixelClassifier, ConcentricPixelClassifier).load(path)

    @abstractmethod
    def save(self, path: str) -> None:
        super().save(path)

    @abstractmethod
    def save_for_spades(self, path: str) -> None:
        super().save(path)

    @abstractmethod
    def classify(self, images: ImageSet) -> ThematicMap:
        thmap = super().classify(images)
        return thmap


class ConcentricRandomForest(ConcentricPixelClassifier):
    def __init__(self, solar_indices: Dict[str, int],
                 themes_in_radii: List[List[str]],
                 radii: Optional[List[int]] = None,  n_trees=20, max_depth=7, min_samples_leaf=100,
                 weights=None, criterion='entropy', n_cores=3, bootstrap=False) -> None:
        super().__init__(solar_indices, themes_in_radii, radii)
        self.kind: str = "ConcentricRandomForest"

        self.models = [skRandomForestClassifier(bootstrap=bootstrap, n_estimators=n_trees,
                                                class_weight=weights,
                                                min_samples_leaf=min_samples_leaf, criterion=criterion,
                                                max_depth=max_depth, n_jobs=n_cores)
                       for _ in range(len(themes_in_radii))]

    def train(self, thematic_maps: List[ThematicMap], image_sets: Optional[List[ImageSet]] = None,
              counts_per_theme: int = 2000) -> None:
        super().train(thematic_maps, image_sets, counts_per_theme)
        for layer, layer_db in self.pixel_db.items():
            x = np.concatenate([values for _, values in layer_db.items()])
            y = np.concatenate([np.repeat(self.solar_indices[theme], values.shape[0])
                                for theme, values in layer_db.items()])
            self.models[layer-1].fit(x, y)
        self.is_trained = True

    @staticmethod
    def load(path: str) -> ConcentricRandomForest:
        super(ConcentricRandomForest, ConcentricRandomForest).load(path)
        full_contents = dd.io.load(path)

        def load_sk_forest(contents):
            rf = skRandomForestClassifier()
            for k, v in contents.items():
                setattr(rf, k, v)
            rf.base_estimator_ = DecisionTreeClassifier()
            for k, v in contents['base_estimator'].items():
                setattr(rf.base_estimator_, k, v)
            rf.base_estimator = clone(rf.base_estimator_)
            rf.estimators_ = [clone(rf.base_estimator) for _ in range(len(contents['trees']))]

            for estimator, tree_values in zip(rf.estimators_, contents['trees']):
                for k, v in contents['extra_base_terms'].items():
                    setattr(estimator, k, v)
                tree_values['nodes'] = np.array([tuple(row) for row in tree_values['nodes']],
                                                dtype=contents['node_type'])
                estimator.tree_ = tree.Tree(contents['extra_base_terms']['n_features_'],
                                            np.zeros(1, dtype=np.intp) + contents['extra_base_terms']['n_classes_'],
                                            contents['extra_base_terms']['n_outputs_'])
                estimator.tree_.__setstate__(tree_values)
            return rf

        rf = ConcentricRandomForest(full_contents['solar_indices'], full_contents['themes_in_radii'],
                                    full_contents['radii'])
        for k, v in full_contents.items():
            if k != 'sk_models_':
                setattr(rf, k, v)
        rf.models = []
        for model in full_contents['sk_models_']:
            rf.models.append(load_sk_forest(model))

        return rf

    def save(self, path: str) -> None:
        super().save(path)

        def save_sk_forest(rf):
            excluded_terms = ['base_estimator', 'base_estimator_', 'estimators_']
            contents = {k: v for k, v in vars(rf).items() if k not in excluded_terms}
            contents['base_estimator'] = vars(getattr(rf, excluded_terms[0]))
            base_estimator_vars = list(contents['base_estimator'].keys())
            contents['extra_base_terms'] = {k: v for k, v in vars(rf.estimators_[0]).items()
                                            if k not in base_estimator_vars and k != 'tree_'}

            def get_tree_state(tree):
                state = tree.__getstate__()
                nodes_type = state['nodes'].dtype
                state['nodes'] = np.array(state['nodes'].tolist())
                return state, nodes_type

            contents['trees'] = [get_tree_state(estimator.tree_)[0] for estimator in rf.estimators_]
            contents['node_type'] = get_tree_state(rf.estimators_[0].tree_)[1]
            return contents

        contents = dict()
        contents['sk_models_'] = list()
        for model_i, model in enumerate(self.models):
            contents['sk_models_'].append(save_sk_forest(model))

        variables_to_save = ['kind', 'image_size', 'solar_indices',
                             'radii', 'mask', 'themes_in_radii', 'is_trained',
                             'theme_index', 'channel_order', 'dtype']
        for k, v in vars(self).items():
            if k in variables_to_save:
                contents[k] = v
        dd.io.save(path, contents)

    def save_for_spades(self, path: str) -> None:
        super().save_for_spades(path)

        def save_sk_forest(rf):
            excluded_terms = ['base_estimator', 'base_estimator_', 'estimators_']
            contents = {k: v for k, v in vars(rf).items() if k not in excluded_terms}
            contents['base_estimator'] = vars(getattr(rf, excluded_terms[0]))
            base_estimator_vars = list(contents['base_estimator'].keys())
            contents['extra_base_terms'] = {k: v for k, v in vars(rf.estimators_[0]).items()
                                            if k not in base_estimator_vars and k != 'tree_'}

            def get_tree_state(tree):
                state = tree.__getstate__()
                nodes_type = state['nodes'].dtype
                state['nodes'] = np.array(state['nodes'].tolist())
                return state, nodes_type

            contents['trees'] = [get_tree_state(estimator.tree_)[0] for estimator in rf.estimators_]
            contents['node_type'] = get_tree_state(rf.estimators_[0].tree_)[1]
            return contents

        if self.radii is None:
            contents = dict()
            contents['sk_objects_'] = save_sk_forest(self.models[0])
            for k, v in vars(self).items():
                if k != 'sk_object_':
                    contents[k] = v
            dd.io.save(path, contents)
        else:
            raise RuntimeError("SPADES does not support concentric random forests. Radii must be none")

    def classify(self, images: ImageSet) -> ThematicMap:
        super().classify(images)
        cube = images.cube()
        new_thmap = np.zeros((1280, 1280))
        for layer in np.unique(self.mask).astype(int):
            layer_mask = (self.mask == layer)
            rf = self.models[layer - 1]
            predictions = rf.predict(cube[layer_mask, :])
            new_thmap[layer_mask] = predictions
        theme_mapping = {value: key for key, value in self.solar_indices.items()}
        return ThematicMap(new_thmap, {'DATE-OBS': images['195'].header['DATE-OBS']}, theme_mapping)
