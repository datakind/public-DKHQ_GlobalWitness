"""Functions for visualizing images in Jupyter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import folium
import folium.plugins
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from IPython.core import display


def create_google_maps_link(longitude, latitude):
    """Creates Google Maps link for a given coordinate.

    Args:
        longitude: float.
        latitude: float.

    Returns:
        IPython.core.display.display element.
    """
    GOOGLE_MAPS_LINK_TEMPLATE = (
        '<a href="https://www.google.com/maps/place/%f,%f">Google Maps</a>')
    html_text = GOOGLE_MAPS_LINK_TEMPLATE % (latitude, longitude)
    return display.display(display.HTML(html_text))


def show_image(image, bounding_box=None, title=None):
    """Shows a single satellite image.

    Args:
        image: np.array of rank 2 or 3. Image to be shown.
        bounding_box: None or BoundingBox instance. Indicates latitude,
            longitude of image.
        title: None or str. Title of axis.
    """
    # Normalize colors based on smallest, largest values in image.
    norm = colors.Normalize(vmin=np.min(image), vmax=np.max(image))
    plt.imshow(image, norm=norm)

    plt.xlabel("longitude")
    plt.ylabel("latitude")

    if bounding_box:
        keys, _ = plt.xticks()
        values = np.linspace(bounding_box.min_longitude, bounding_box.max_longitude, num=len(keys), endpoint=False)
        values = ["{:0.4f}".format(value) for value in values]
        plt.xticks(keys[1:-1], values[1:-1], rotation=45)

        keys, _ = plt.yticks()
        values = np.linspace(bounding_box.max_latitude, bounding_box.min_latitude, num=len(keys), endpoint=False)
        values = ["{:0.4f}".format(value) for value in values]
        plt.yticks(keys[1:-1], values[1:-1], rotation=45)

    if title:
        plt.title(title)


def show_color_bands(image, axs=None, ncols=None, titles=None):
    """Shows each of an image's color bands.

    Args:
        image: np.array of rank 3. Dimensions represent (longitude, latitude,
            color band).
        axs: None or np.array of of rank 2. Contains matplotlib axes for each
            color band.
        ncols: None or int. Number of columns per row. If None, defaults to 6.
        titles: None or list of strings. Title for each axes.
    """
    num_bands = image.shape[-1]
    assert num_bands == len(titles)

    ncols = ncols or 6
    titles = titles or [None for _ in range(num_bands)]

    if axs is None:
        nrows = int(np.ceil(num_bands / float(ncols)))
        fig = plt.gcf()
        axs = np.asarray([fig.add_subplot(nrows, ncols, i + 1)
                          if i < num_bands else None
                          for i in range(nrows * ncols)])
        axs = np.reshape(axs, [nrows, ncols])
    else:
        nrows, ncols = axs.shape
    assert nrows * ncols >= num_bands

    axs = axs.flatten()
    for i in range(num_bands):
        plt.sca(axs[i])
        show_image(image[..., i], title=titles[i])
        plt.xticks([])
        plt.yticks([])


def show_landsat8_color_combinations(image, axs=None, ncols=None):
    """Shows common color band combinations for Landsat 8 images.

    Args:
      image: np.array of rank 3. Represents longitude, latitude, and color
        band. Last dimension show be ordered B1...B11.
      axs: None or np.array of rank 2. Contains matplotlib Axes to plot each
        color combination on.
      ncols: None or int. Number of columns per row if creating axes from
        scratch.
    """

    # Based on
    # https://blogs.esri.com/esri/arcgis/2013/07/24/band-combinations-for-landsat-8/
    LANDAT8_BAND_COMBOS = {
        'Natural Color': [5, 4, 3],
        'False Color (urban)': [8, 7, 5],
        'Color Infrared (vegetation)': [6, 5, 4],
        'Agriculture': [7, 6, 3],
        'Atmospheric Penetration': [8, 7, 6],
        'Healthy Vegetation': [6, 7, 3],
        'Land/Water': [6, 7, 5],
        'Natural With Atmospheric Removal': [8, 6, 4],
        'Shortwave Infrared': [8, 6, 5],
        'Vegetation Analysis': [7, 6, 5],
    }
    titles, band_combos = zip(*sorted(LANDAT8_BAND_COMBOS.items()))
    images = [image[:, :, band_combo] for band_combo in band_combos]
    images = np.stack(images, axis=-1)
    show_color_bands(images, axs=axs, ncols=ncols, titles=titles)


def create_folium_map(bounding_box, tiles=None):
    """Create a folium.Map over meta's location.

    Args:
        bounding_box: BoundingBox instance.
        tiles: None or str. Name of tile provider. Includes 'Mapbox', 'ArcGIS',
            and anything folium provides.

    Returns:
        folium.Map.
    """

    MAPBOX_API_KEY = 'pk.eyJ1IjoiZHVja3dvcnRoZCIsImEiOiJjamE0YmdtcDQ5dTB1MndsZ3N5eDk4a2tiIn0.R_cSc4qJEPLxnlWnFqx-lg'

    if tiles == 'Mapbox':
        map_kwargs = {
            'attr': 'Mapbox Satellite',
            'tiles': 'http://{s}.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token=' + MAPBOX_API_KEY,
        }
    elif tiles == 'ArcGIS':
        map_kwargs = {
            'attr': 'Tile: Esri - Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            'tiles':'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        }
    else:
        map_kwargs = {
            'tiles': tiles
        }

    folium_map = folium.Map(**map_kwargs)

    min_lat = bounding_box.min_latitude
    max_lat = bounding_box.max_latitude
    min_lon = bounding_box.min_longitude
    max_lon = bounding_box.max_longitude

    folium_map.add_child(folium.LatLngPopup())
    folium_map.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
    folium_map.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])

    return folium_map


def overlay_image_on_map(folium_map, image, bounding_box, opacity=None):
    """Overlays an array onto a folium.Map.

    Args:
        folium_map: folium.Map instance.
        instance: np.array of rank 2 or 3. Image to overlay.
        bounding_box: BoundingBox instance.
        opacity: float or None. Must be in [0, 1].

    Returns:
        folium.Map
    """
    overlay = folium.plugins.ImageOverlay(
        image=image,
        bounds=[(bounding_box.min_latitude, bounding_box.min_longitude),
                (bounding_box.max_latitude, bounding_box.max_longitude)],
        opacity=opacity or 0.5,
    )
    overlay.add_to(folium_map)
    return folium_map
