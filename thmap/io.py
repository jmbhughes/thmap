from __future__ import annotations
from astropy.io import fits
import numpy as np
from goessolarretriever import Product, Satellite, Retriever
from collections import namedtuple
import tempfile
import os
from dateutil.parser import parse as parse_date_str
from datetime import datetime
from typing import Dict, List, Optional
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib

NUMBER_OF_CHANNELS = 6
DEFAULT_THEMATIC_MAP_COLORS = {"unlabeled": "white",
                                 "outer_space": "black",
                                 "bright_region": "#F0E442",
                                 "filament": "#D55E00",
                                 "prominence": "#E69F00",
                                 "coronal_hole": "#009E73",
                                 "quiet_sun": "#0072B2",
                                 "limb": "#56B4E9",
                                 "flare": "#CC79A7"}

Image = namedtuple('Image', 'data header')


class ImageSet:
    def __init__(self, mapping: Dict[str, Image]) -> None:
        super().__init__()
        self.images = mapping

    @staticmethod
    def retrieve(date: datetime) -> ImageSet:
        satellite = Satellite.GOES16
        products = {"94": Product.suvi_l2_ci094,
                    "131": Product.suvi_l2_ci131,
                    "171": Product.suvi_l2_ci171,
                    "195": Product.suvi_l2_ci195,
                    "284": Product.suvi_l2_ci284,
                    "304": Product.suvi_l2_ci304}
        composites = {}
        r = Retriever()
        for wavelength, product in products.items():
            fn = r.retrieve_nearest(satellite, product, date, tempfile.gettempdir())
            with fits.open(fn) as hdus:
                data = hdus[1].data
                header = hdus[1].header
                composites[wavelength] = Image(data, header)
            os.remove(fn)
        return ImageSet(composites)

    @staticmethod
    def create_empty() -> ImageSet:
        mapping = {"94": Image(np.zeros((1280, 1280)), {}),
                   '131': Image(np.zeros((1280, 1280)), {}),
                   '171': Image(np.zeros((1280, 1280)), {}),
                   '195': Image(np.zeros((1280, 1280)), {}),
                   '284': Image(np.zeros((1280, 1280)), {}),
                   '304': Image(np.zeros((1280, 1280)), {})}
        return ImageSet(mapping)

    def __getitem__(self, key) -> Image:
        return self.images[key]

    def channels(self) -> List[str]:
        return list(self.images.keys())

    def cube(self) -> np.ndarray:
        """
        Convert the image to a np.ndarray cube
        :return: cube of an image with (1280,1280,6) shape
        """
        return np.stack([self.images[channel].data for channel in self.channels()], axis=2)

    def get_solar_radius(self, channel="304"):
        """
        Gets the solar radius from the header of the specified channel
        :param channel: channel to get radius from
        :return: solar radius specified in the header
        """
        if channel not in self.channels():
            raise RuntimeError("Channel requested must be one of {}".format(self.channels()))
        try:
            solar_radius = self.images[channel].header['DIAM_SUN'] / 2
        except KeyError:
            raise RuntimeError("Header does not include the solar diameter or radius")
        else:
            return solar_radius


class ThematicMap:
    def __init__(self, data: np.ndarray, metadata: Dict, theme_mapping: Dict[int, str]) -> None:
        """
        A representation of a thematic map
        :param data: the image of numbers for the labelling
        :param metadata: dictionary of header information
        :param theme_mapping: dictionary of theme indices to theme names, the second hdu info
        """
        self.data = data
        self.metadata = metadata
        self.date_obs: datetime = parse_date_str(self.metadata['DATE-OBS'])
        self.theme_mapping: Dict[int, str] = theme_mapping

    @staticmethod
    def create_empty() -> ThematicMap:
        data: np.ndarray = np.zeros((1280, 1280))
        now: datetime = datetime.now()
        metadata = {'DATE-OBS': str(now)}
        theme_mapping: Dict[int, str] = {1: 'outer_space',
                                         3: 'bright_region',
                                         4: 'filament',
                                         5: 'prominence',
                                         6: 'coronal_hole',
                                         7: 'quiet_sun',
                                         8: 'limb',
                                         9: 'flare'}
        return ThematicMap(data, metadata, theme_mapping)

    @staticmethod
    def load(path: str) -> ThematicMap:
        """
        Load a thematic map
        :param path: path to the file
        :return: ThematicMap object that was loaded
        """
        with fits.open(path) as hdulist:
            data = hdulist[0].data
            metadata = dict(hdulist[0].header)
            theme_mapping = dict(hdulist[1].data)
            if 0 in theme_mapping:
                del theme_mapping[0]
        return ThematicMap(data, metadata, theme_mapping)

    def complies_with_mapping(self, other_theme_mapping: Dict) -> bool:
        """
        Checks that the theme mappings match, i.e. they have identical entries
        :param other_theme_mapping: a dictionary of another theme_mapping, e.g. {1: 'outer_space'}
        :return: true or false depending on the matching
        """
        for theme_i, theme_name in self.theme_mapping.items():
            if theme_i not in other_theme_mapping:
                return False
            if other_theme_mapping[theme_i] != self.theme_mapping[theme_i]:
                return False

        for theme_i, theme_name in other_theme_mapping.items():
            if theme_i not in self.theme_mapping:
                return False
            if other_theme_mapping[theme_i] != self.theme_mapping[theme_i]:
                return False
        return True

    def save(self, path: str) -> None:
        """
        Write out a thematic map FITS
        :param path: where to save thematic maps fits file
        :return:
        """
        # make sure the data is 8 bit
        self.data = self.data.astype(np.uint8)

        pri_hdu = fits.PrimaryHDU(data=self.data)
        for k, v in self.metadata.items():
            if k != 'COMMENT':
                pri_hdu.header[k] = v

        map_val = []
        map_label = []
        for key, value in sorted(self.theme_mapping.items(), key=lambda k_v: (k_v[0], k_v[1])):
            map_label.append(value)
            map_val.append(key)
        c1 = fits.Column(name="Thematic Map Value", format="B", array=np.array(map_val))
        c2 = fits.Column(name="Feature Name", format="22A", array=np.array(map_label))
        bintbl_hdr = fits.Header([("XTENSION", "BINTABLE")])
        sec_hdu = fits.BinTableHDU.from_columns([c1, c2], header=bintbl_hdr)
        hdu = fits.HDUList([pri_hdu, sec_hdu])
        hdu.writeto(path, overwrite=True, checksum=True)

    def copy_195_metadata(self, image_set: ImageSet) -> None:
        keys_to_copy = ['YAW_FLIP', 'ECLIPSE', 'WCSNAME', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
                        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2',
                        'CRPIX1', 'CRPIX2', 'DIAM_SUN', 'LONPOLE', 'CROTA', 'SOLAR_B0', 'ORIENT', 'DSUN_OBS']
        if image_set.images['195'].header != {}:
            for key in keys_to_copy:
                self.metadata[key] = image_set.images['195'].header[key]

    def generate_plot(self, save_path: Optional[str] = None, with_legend: bool = False,
                      colors: Optional[Dict[str, str]] = None) -> None:
        """
        Generates a color plot using Matplotlib of the thematic map
        :param save_path: where to save the image if specified
        :param with_legend: whether to include the legend in the plot
        :param colors: the mapping of colors, with keys of themes and values of color for that theme
        """
        # setup color table
        if colors is None:
            colors = DEFAULT_THEMATIC_MAP_COLORS

        if set(self.theme_mapping.values()).union({'unlabeled'}) != set(colors.keys()):
            raise RuntimeError("Theme mapping themes do not match colors themes.")

        colortable: List[str] = [colors[self.theme_mapping[i]] if i in self.theme_mapping else 'black'
                      for i in range(max(list(self.theme_mapping.keys())) + 1)]
        cmap = matplotlib.colors.ListedColormap(colortable)

        # do actual plotting
        fig, ax = plt.subplots()
        ax.imshow(self.data, origin='lower', cmap=cmap, vmin=-1, vmax=len(colortable))
        ax.set_axis_off()
        if with_legend:
            legend_elements = [Patch(facecolor=color, edgecolor="black", label=label.replace("_", " "))
                               for label, color in colors.items()]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                      ncol=3, fancybox=True, shadow=True)
        fig.tight_layout()
        fig.show()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
