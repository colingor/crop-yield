import logging
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from glob import glob
from logging import handlers
from pathlib import Path
from typing import List
from zipfile import ZipFile

import click
import earthpy.plot as ep
import geojson
import geopandas as gpd
import kml2geojson

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from dateutil import parser
from fiona.crs import from_epsg
from geopandas import GeoDataFrame
from matplotlib import gridspec
from pandas import Series
from rasterio import plot
from rasterio.mask import mask
from rasterio.plot import show
from sentinelsat.sentinel import SentinelAPI
from shapely.geometry import Polygon, MultiPolygon

pd.set_option("display.max_columns", None)  # or 1000
pd.set_option("display.max_rows", None)  # or 1000
pd.set_option("display.max_colwidth", None)

BAND_2_10M = "B02_10m"
BAND_3_10M = "B03_10m"
BAND_4_10M = "B04_10m"
BAND_8_10M = "B08_10m"
BAND_TCI_10M = "TCI_10m"

BAND_5_20M = "B05_20m"
BAND_6_20M = "B06_20m"
BAND_7_20M = "B07_20m"
BAND_8A_20M = "B8A_20m"
BAND_11_20M = "B11_20m"
BAND_12_20M = "B12_20m"
BAND_SCL_20M = "SCL_20m"
BAND_TCI_20M = "TCI_20m"

BAND_01_60M = "B01_60m"
BAND_09_60M = "B09_60m"
BAND_10_60M = "B10_60m"  # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2 (the cirrus band 10 is omitted, as it does not contain surface information)

ALL_BANDS = (
    BAND_2_10M,
    BAND_3_10M,
    BAND_4_10M,
    BAND_8_10M,
    BAND_5_20M,
    BAND_6_20M,
    BAND_7_20M,
    BAND_8A_20M,
    BAND_11_20M,
    BAND_12_20M,
    BAND_01_60M,
    BAND_09_60M,
    BAND_10_60M,
    BAND_TCI_10M,
    BAND_SCL_20M,
    BAND_TCI_20M,
)

REPORT_SUMMARY_BANDS = BAND_TCI_20M, BAND_SCL_20M, BAND_TCI_10M

log = logging.getLogger(__name__)
DATA_DIRECTORY = "data"
FARM_SENTINEL_DATA_DIRECTORY = f"{DATA_DIRECTORY}/sentinel2"
FARM_SUMMARIES_DIRECTORY = f"{FARM_SENTINEL_DATA_DIRECTORY}/farm_summaries"
FARM_LOCATIONS_DIRECTORY = f"{DATA_DIRECTORY}/farm_locations"
FARMS_KMZ = f"{FARM_LOCATIONS_DIRECTORY}/all_farms_27_03_22.kmz"
FARMS_KML = f"{FARM_LOCATIONS_DIRECTORY}/all_farms_27_03_22.kml"
FARMS_GEOJSON = f"{FARM_LOCATIONS_DIRECTORY}/all_farms_27_03_22.geojson"
SENTINEL_PRODUCTS_GEOJSON = f"{FARM_SENTINEL_DATA_DIRECTORY}/products.geojson"

# Create logs dir if it doesn't already exist
os.makedirs("logs", exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler("logs/crop.log", maxBytes=2048, backupCount=5)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s: %(levelname)s] %(message)s",
    handlers=[file_handler, logging.StreamHandler()],
)


@dataclass
class CropDataHandler:
    sentinel_date_range: tuple
    api: SentinelAPI = None
    total_bbox_32643: Polygon = None
    total_bbox: GeoDataFrame = None
    farm_bounds_32643: GeoDataFrame = None
    products_df: GeoDataFrame = None

    def __post_init__(self):
        self.initialise()

    def initialise(self):
        """
        Check kmz exists, configure API and read geometries
        :return:
        """

        if not os.path.exists(FARMS_KMZ):
            sys.exit(f"Unable to find file {FARMS_KMZ} - aborting")

        # Ensure directory to store Sentinel2 data exists
        os.makedirs(FARM_SENTINEL_DATA_DIRECTORY, exist_ok=True)
        os.makedirs(FARM_SUMMARIES_DIRECTORY, exist_ok=True)

        # Create api instance
        self.configure_api()

        log.debug("Converting KML file")

        self.extract_geometries()

    def extract_geometries(self):
        """
        Unzip the kmz and derive shapefiles, geojson and cache farm bounds and total bounding box
        """
        kmz = ZipFile(FARMS_KMZ, "r")
        log.debug("Unzipped kmz")
        kml = kmz.open("doc.kml", "r").read()
        with open(FARMS_KML, "wb") as f:
            f.write(kml)

        farms_geojson = kml2geojson.main.convert(FARMS_KML)[0]

        with open(FARMS_GEOJSON, "w") as f:
            geojson.dump(farms_geojson, f)

        # Save as shapefile in desired projection
        farm_bounds = gpd.read_file(FARMS_GEOJSON)
        self.farm_bounds_32643 = farm_bounds.to_crs({"init": "epsg:32643"})
        self.farm_bounds_32643.to_file(f"{FARM_LOCATIONS_DIRECTORY}/individual_farm_bounds.shp")

        # Plot the bounds of the fields (not coloured in)
        # self.farm_bounds_32643.boundary.plot()
        # plt.show()

        # Save overall bounding box in desired projection
        self.total_bbox_32643 = shapely.geometry.box(*self.farm_bounds_32643.total_bounds, ccw=True)

        self.total_bbox = gpd.GeoDataFrame({"geometry": self.total_bbox_32643}, index=[0], crs=from_epsg(32643))
        self.total_bbox.to_file(f"{FARM_LOCATIONS_DIRECTORY}/total_bounds.shp")

        # Update the geometry in farms datafile to make it 2D so rasterio can handle it.
        # It seems rasterio won't work with 3D geometry
        self.farm_bounds_32643.geometry = self.convert_3D_2D(self.farm_bounds_32643.geometry)

        if os.path.exists(SENTINEL_PRODUCTS_GEOJSON):
            self.products_df = gpd.read_file(SENTINEL_PRODUCTS_GEOJSON)

        log.info(f"Finished setup")

    def convert_3D_2D(self, geometry):
        """
        Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
        (Taken from https://gist.github.com/rmania/8c88377a5c902dfbc134795a7af538d8)
        """
        new_geo = []
        for p in geometry:
            if p.has_z:
                if p.geom_type == "Polygon":
                    lines = [xy[:2] for xy in list(p.exterior.coords)]
                    new_p = Polygon(lines)
                    new_geo.append(new_p)
                elif p.geom_type == "MultiPolygon":
                    new_multi_p = []
                    for ap in p:
                        lines = [xy[:2] for xy in list(ap.exterior.coords)]
                        new_p = Polygon(lines)
                        new_multi_p.append(new_p)
                    new_geo.append(MultiPolygon(new_multi_p))
        return new_geo

    def configure_api(
        self,
    ):
        """
        Initialise SentinelAPI instance
        """
        user = os.environ["SENTINEL_USER"]
        password = os.environ["SENTINEL_PASSWORD"]

        if not password or user:
            log.warning(
                "SENTINEL_USER or SENTINEL_PASSWORD environment variables are not present. "
                "You will be unable to use the Sentinel API without setting these variables"
            )
        self.api = SentinelAPI(user, password, "https://scihub.copernicus.eu/dhus")

    def _save_sentinel_product_list_and_download(self):
        """
        Write products dataframe to filesystem
        Attempt to download each product in the dataframe
        :return:
        """
        # Save the products so we have a record
        self.save_products_df_to_geojson()

        def _download(area: GeoDataFrame):
            uuid = area["uuid"]
            identifier = area["identifier"]
            if not os.path.exists(f"{FARM_SENTINEL_DATA_DIRECTORY}/{identifier}.SAFE"):
                log.debug(f"About to download {uuid}")
                try:
                    self.api.download(uuid, directory_path=FARM_SENTINEL_DATA_DIRECTORY, checksum=False)
                except Exception as e:
                    log.error(f"Problem downloading: {e}")
            else:
                log.debug(f"We already have a file for {identifier}")

        self.products_df.apply(_download, axis=1)

    def get_all_farms_bounding_box_wkt(self):
        """
        Get the bounding box for all farm fields in wkt
        :return: wkt bounding box
        """
        return shapely.geometry.box(*gpd.read_file(FARMS_GEOJSON).total_bounds, ccw=True).wkt

    def get_available_sentinel_products_df(self, verify_products=False):
        """
        Get dataframe listing available Sentinel products
        :return:
        """

        products = self.api.query(
            self.get_all_farms_bounding_box_wkt(),
            date=self.sentinel_date_range,
            # date=("20210401", "20220401"),
            platformname="Sentinel-2",
            processinglevel="Level-2A",
            cloudcoverpercentage=(0, 99),
        )

        self.products_df = self.api.to_geodataframe(products)
        # products_df = products_df.sort_values(["cloudcoverpercentage"], ascending=[True])
        self.products_df = self.products_df.sort_values(["generationdate"], ascending=[True])

        # Filter products_df on tile id
        # products_df = products_df.loc[products_df['title'].str.contains("T43PFT", case=False)]

        # Granule T43PFS contains all the farms
        self.products_df = self.products_df.loc[self.products_df["title"].str.contains("T43PFS", case=False)]
        log.debug(f"{len(self.products_df)} products available for tile number T43PFS")

        if verify_products:
            # Various plots below to debug product and farm positions

            # Read farm bounds in in same crs as products here for easy comparison (4326)
            farm_bounds = gpd.read_file(FARMS_GEOJSON)

            # Simple plot to show product positions
            plot = self.products_df.plot(column="uuid", cmap=None)
            # plt.savefig("test.jpg")
            plt.show()

            # Product positions with uuids overlaid
            ax = self.products_df.plot(column="uuid", cmap=None, figsize=(20, 20))
            # products_df.apply(lambda x: ax.annotate(s=x.uuid, xy=x.geometry.centroid.coords[0], ha='center'),axis=1)
            self.products_df.apply(
                lambda x: ax.annotate(text=x["uuid"], xy=x.geometry.centroid.coords[0], ha="center"), axis=1
            )

            # Simple plot to show red fields on white background
            base = self.products_df.plot(color="white", edgecolor="black")
            farm_bounds.plot(ax=base, marker="o", color="red", markersize=5)

            plt.show()

            # Save as a folium map
            # m = self.farm_bounds_32643.explore()
            # m.save("mymap.html")
            # *****

            # Plot the products titles to see positions
            ax = self.products_df.plot(column="title", cmap=None, figsize=(50, 50), alpha=0.3)
            self.products_df.apply(
                lambda x: ax.annotate(text=x["title"], xy=x.geometry.centroid.coords[0], ha="center"), axis=1
            )
            plt.show()

            # Plot products names and farm names
            f, ax = plt.subplots(1)
            self.products_df.plot(
                ax=ax,
                column="uuid",
                cmap="OrRd",
            )
            self.products_df.apply(
                lambda x: ax.annotate(text=x["title"], xy=x.geometry.centroid.coords[0], ha="center"), axis=1
            )

            farm_bounds.plot(ax=ax, column="name", cmap=None, figsize=(50, 50))
            farm_bounds.apply(
                lambda x: ax.annotate(text=x["name"], xy=x.geometry.centroid.coords[0], ha="center"), axis=1
            )
            ax.set_title("Fields on sentinel data", fontsize=10, pad=10)
            plt.show()

            # Show field outlines on both products
            base = self.products_df.plot(color="white", edgecolor="black")
            farm_bounds.plot(ax=base, marker="o", color="red", markersize=5)

    def download_sentinel_products(self):
        """
        Get available sentinel 2 products and download
        :return:
        """
        self.get_available_sentinel_products_df()

        self._save_sentinel_product_list_and_download()

        self.unzip_sentinel_products()

        # Validate results
        unzipped_products = glob(
            f"{FARM_SENTINEL_DATA_DIRECTORY}/*.SAFE",
            recursive=False,
        )

        total_available_products = len(self.products_df)

        remaining = total_available_products - len(unzipped_products)

        if remaining:
            msg = f"Note that there are {remaining}/{total_available_products} products which have not yet been downloaded. Please re-run the download function"
            log.info(msg)
            sys.exit(msg)

        log.debug("Product Download is complete")

    def unzip_sentinel_products(self):
        """
        Unzip all products
        """

        def _unzip_if_required(sentinal_zip):
            file_path = f"{FARM_SENTINEL_DATA_DIRECTORY}/{sentinal_zip}"
            unzipped_filename = Path(file_path).with_suffix(".SAFE")
            if not os.path.exists(unzipped_filename):
                if zipfile.is_zipfile(file_path):
                    with zipfile.ZipFile(file_path) as item:
                        log.debug(f"Unzipping {file_path}…")
                        item.extractall(FARM_SENTINEL_DATA_DIRECTORY)
            else:
                log.debug(f"Not unzipping as {unzipped_filename} already exists")

        [
            _unzip_if_required(sentinal_zip)
            for sentinal_zip in os.listdir(FARM_SENTINEL_DATA_DIRECTORY)
            if sentinal_zip.endswith(".zip")
        ]

    def crop_raster_to_geometry(
        self, raster_file: str, geom: Polygon, cropped_directory_name: str, verify_images=False, force_recreate=False
    ):
        """
        Crop the specified raster file to self.total_bbox_32643 (combined farms bounding box)
        :param force_recreate: Recreate a cropped raster even if it already exists
        :param verify_images: Output some plots to sanity check results
        :param cropped_directory_name:
        :param geom: Geometry to crop raster to
        :param raster_file: Relative path the raster file
        """

        # We want to save as tiff rather than jp2
        raster_file_path = Path(raster_file).with_suffix(".tif")

        cropped_image_dir = f"{raster_file_path.parent.parent.parent}/IMG_DATA_CROPPED"
        cropped_directory = f"{cropped_image_dir}/{cropped_directory_name}/{raster_file_path.parent.name}"
        os.makedirs(cropped_directory, exist_ok=True)

        output_raster = f"{cropped_directory}/{raster_file_path.name}"

        if not os.path.exists(output_raster) or force_recreate:

            with rasterio.open(raster_file) as src:
                # Note the geometry has to be iterable, hence the list
                out_img, out_transform = mask(src, [geom], crop=True)
                out_meta = src.meta.copy()

                # It seems the output raster is blank if we use JP2OpenJPEG, so go with Gtiff
                out_meta.update(
                    {
                        "driver": "Gtiff",
                        "height": out_img.shape[1],
                        "width": out_img.shape[2],
                        "transform": out_transform,
                    }
                )

                with rasterio.open(output_raster, "w", **out_meta) as dest:
                    dest.write(out_img)

            if verify_images:
                # Verify the images look correct
                with rasterio.open(output_raster) as clipped:

                    show((clipped, 1), cmap="terrain")

                    # Check bounding box and raster match
                    fig, ax = plt.subplots(figsize=(15, 15))
                    rasterio.plot.show(clipped, ax=ax)
                    self.total_bbox.plot(ax=ax, facecolor="none", edgecolor="r")
                    plt.show()

                    # Show the field outlines on the raster
                    fig, ax = plt.subplots(figsize=(15, 15))
                    rasterio.plot.show(clipped, ax=ax)
                    self.farm_bounds_32643.plot(ax=ax, facecolor="none", edgecolor="r")
                    plt.show()
        else:
            log.debug(f"Skipping as {raster_file_path} already exists. To recreate, set force_recreate=True")

    def process_product_dataframe_and_rasters(self, product: Series):
        """
        For every row in the products dataframe, crop the downloaded rasters to overall farm bounds and
        add band paths to products dataframe
        :param product: Series
        :return: product:Series
        """

        original_rasters: list = self.get_original_product_rasters(product)

        if original_rasters:

            # Crop all original rasters to all farms geom
            [self.crop_raster_to_geometry(raster, self.total_bbox_32643, "all_farms") for raster in original_rasters]

            def _add_band_filepath(band):
                """
                Add path to specified band
                :param band:
                :return:
                """
                search_result = [raster_file for raster_file in original_rasters if band in raster_file]
                product[band] = search_result[0] if search_result else ""

            # Add raster paths to dataframe so we can easily look them up
            [_add_band_filepath(band) for band in ALL_BANDS]

        else:
            log.debug(f"Skipping as product {product['title']} as associated rasters not found")

        return product

    def crop_rasters_to_all_fields_bbox(self):
        """
        Inspect the rasters we have in FARM_SENTINEL_DATA_DIRECTORY and clip to all farm bounds
        """
        # Check products have been downloaded
        self.check_products_geojson_exists()

        # Crop rasters to overall farm bounds and add band paths for each product
        self.products_df = self.products_df.apply(self.process_product_dataframe_and_rasters, axis=1)

        # Save updated products
        self.save_products_df_to_geojson()

        log.debug(self.products_df)

    def check_products_geojson_exists(self):
        """
        Stop execution if we don't have products geojson
        :return:
        """
        if not os.path.exists(SENTINEL_PRODUCTS_GEOJSON):
            sys.exit(
                f"Unable to find file {SENTINEL_PRODUCTS_GEOJSON} - aborting. Please run script with download flag to generate this file"
            )

    def process_products_for_farms(self, product: Series):
        """
        Iterate through products. For each, generate rasters for the bounds of each farm
        :param product:
        :return:
        """

        def _process_products_for_individual_farm(farm: Series):
            """
            Generate rasters for the specified farm for the specified product
            :param farm:
            :return:
            """

            farm_name = farm["name"]
            farm_geometry = farm["geometry"]

            # Visualise geometry
            # x,y = farm_geometry.exterior.xy
            # plt.plot(x,y)
            # plt.show()
            # farm_bbox = shapely.geometry.box(*farm_geometry.bounds, ccw=True)
            # Confirm that the bounding box is correct
            # x1,y1 = farm_bbox.exterior.xy
            # plt.plot(x1,y1, x, y)
            # plt.show()

            original_rasters = self.get_original_product_rasters(product)

            if original_rasters:
                # Crop all rasters to individual farm bboxes
                [self.crop_raster_to_geometry(raster, farm_geometry, farm_name) for raster in original_rasters]

            else:
                log.debug(f"Skipping product {product['title']} as associated rasters not found")

        self.farm_bounds_32643.apply(_process_products_for_individual_farm, axis=1)

        return product

    def get_original_product_rasters(self, product: Series) -> List:
        """
        Get list of original rasters for the specified product
        :param product:
        :return: List
        """

        product_directory = f"{FARM_SENTINEL_DATA_DIRECTORY}/{product['filename']}/GRANULE"

        if os.path.exists(product_directory):
            # Get all rasters for this product
            return glob(
                f"{product_directory}/**/IMG_DATA/**/*.jp2",
                recursive=True,
            )
        return None

    def crop_rasters_to_individual_fields_bbox(self):
        """
        Iterate through products, for each iterate through the farms and extract rasters for each farm geom
        :return:
        """

        # open products so we have extra info
        self.check_products_geojson_exists()

        self.products_df = self.products_df.apply(self.process_products_for_farms, axis=1)

        self.save_products_df_to_geojson()

    def save_products_df_to_geojson(self):
        """
        Save products dataframe to filesystem as GEOJSON

        """
        self.products_df.to_file(SENTINEL_PRODUCTS_GEOJSON, driver="GeoJSON")

    def preview_farm_bands(self):
        """
        Experiments with viewing cropped fields.  WIP
        """

        # Pick a farm
        farm_name = self.farm_bounds_32643.iloc[0]["name"]
        first_product = self.products_df.iloc[0]

        # Get all bands paths for this product
        band_paths = (
            self.get_farm_raster_for_band(farm_name, first_product[band])
            for band in (BAND_2_10M, BAND_3_10M, BAND_4_10M, BAND_8_10M)
        )

        band_paths = filter(None, band_paths)

        l = []
        for i in band_paths:
            with rasterio.open(i, "r") as f:
                l.append(f.read(1))

        arr_st = np.stack(l)
        print(f"Height: {arr_st.shape[1]}\nWidth: {arr_st.shape[2]}\nBands: {arr_st.shape[0]}")

        ep.plot_bands(arr_st, cmap="gist_earth", figsize=(20, 12), cols=6, cbar=False)
        plt.show()

        rgb = ep.plot_rgb(arr_st, rgb=(3, 2, 1), figsize=(8, 10), title="RGB Composite Image")

    def create_cropped_rgb_image(self):
        """
        Test creating an rgb image for a farm
        """

        farm_name = self.farm_bounds_32643.iloc[0]["name"]

        first_product = self.products_df.iloc[0]

        band2 = self.get_farm_raster_for_band(farm_name, first_product[BAND_2_10M])
        band3 = self.get_farm_raster_for_band(farm_name, first_product[BAND_3_10M])
        band4 = self.get_farm_raster_for_band(farm_name, first_product[BAND_4_10M])

        # export true color image
        output_raster = f"{Path(band2).parent}/rgb.tif"

        band2 = rasterio.open(band2)  # blue
        band3 = rasterio.open(band3)  # green
        band4 = rasterio.open(band4)  # red

        with rasterio.open(
            output_raster,
            "w",
            driver=band4.driver,
            width=band4.width,
            height=band4.height,
            count=3,
            crs=band4.crs,
            transform=band4.transform,
            dtype=band4.dtypes[0],
        ) as rgb:
            rgb.write(band2.read(1), 3)
            rgb.write(band3.read(1), 2)
            rgb.write(band4.read(1), 1)

        log.debug(f"Created rgb image {output_raster}")

        # with rasterio.open(output_raster, count=3) as src
        #     plot.show(src)

    def generate_all_farms_bands_summary(self):
        """
        For each farm, generate jpeg for each summary band showing how image changes for each product over time

        """
        [self.generate_all_farms_summary(band) for band in REPORT_SUMMARY_BANDS]

    def generate_all_farms_summary(self, band: str, verify_images=False):
        """
        Generate plot for all farms for specified band over time
        :param band:
        :param verify_images:
        :return:
        """
        filtered_products_df = self.products_df[self.products_df[band].notnull()]

        band_rasters = list(filter(None, map(self.get_all_farms_raster_for_band, filtered_products_df[band])))

        if band_rasters:
            number_of_raster = len(band_rasters)

            cols = 6
            rows = int(math.ceil(number_of_raster / cols))

            gs = gridspec.GridSpec(rows, cols, wspace=0.01)

            fig = plt.figure(figsize=(50, 50))
            plt.tight_layout()

            fig.suptitle(f"All farms {band}, all products", fontsize=40)

            for n in range(number_of_raster):
                ax = fig.add_subplot(gs[n])

                product = filtered_products_df.iloc[n]

                dt = parser.parse(product.generationdate)

                ax.set_title(f"{product.title}:\n{dt.day}/{dt.month}/{dt.year}", fontsize=10, wrap=True)
                # ax.set_title(f"{dt.day}/{dt.month}/{dt.year}\n{product.uuid}", fontsize=10)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

                with rasterio.open(band_rasters[n], "r") as src:
                    # Overlay individual field bounds
                    self.farm_bounds_32643.plot(ax=ax, facecolor="none", edgecolor="r")
                    plot.show(src, ax=ax, cmap="terrain")
                    # plt.tight_layout()

                    if verify_images:

                        # Show the field outlines on the raster
                        fig, ax = plt.subplots(figsize=(15, 15))
                        rasterio.plot.show(src, ax=ax)
                        self.farm_bounds_32643.plot(ax=ax, facecolor="none", edgecolor="r")
                        plt.show()

            farm_name_dir = f"{FARM_SUMMARIES_DIRECTORY}/all_farms"

            os.makedirs(farm_name_dir, exist_ok=True)

            plt.savefig(f"{farm_name_dir}/{band}.jpg", format="jpeg", bbox_inches="tight")

    def generate_individual_farm_bands_summary(self):
        """
        For each farm, plot each of the summary bands in a jpeg over time
        :return:
        """

        [
            self.generate_individual_farm_band_summary(farm_index, band)
            for farm_index in range(len(self.farm_bounds_32643))
            for band in REPORT_SUMMARY_BANDS
        ]

    def generate_individual_farm_band_summary(self, farm_df_index: int, band: str, verify_images=False):
        """
        Plot how specified band changes over time for a farm
        :param farm_df_index: Index of farm in farms dataframe
        :param verify_images: Show the plot
        :return:
        """

        # Get the farm
        try:
            farm_details = self.farm_bounds_32643.iloc[farm_df_index]
        except IndexError as e:
            log.error(e)
            sys.exit("Farm index provided is out of range - exiting")

        farm_name = farm_details["name"]

        # If we have a situation where we've not yet downloaded all the products in the dataframe, we filter out those
        # where we haven't got the desired band
        filtered_products_df = self.products_df[self.products_df[band].notnull()]

        band_rasters = [self.get_farm_raster_for_band(farm_name, band) for band in filtered_products_df[band]]

        band_rasters = list(filter(None, band_rasters))

        number_of_raster = len(band_rasters)

        cols = 6
        rows = int(math.ceil(number_of_raster / cols))

        gs = gridspec.GridSpec(rows, cols, wspace=0.01)

        fig = plt.figure(figsize=(24, 24))
        fig.suptitle(f"Farm {farm_df_index}: {farm_name.capitalize()} {band}, all products", fontsize=40)

        for n in range(number_of_raster):
            ax = fig.add_subplot(gs[n])

            product = filtered_products_df.iloc[n]

            dt = parser.parse(product.generationdate)

            # ax.set_title(f"{product.title}:\n{dt.day}/{dt.month}/{dt.year}", fontsize = 10, wrap=True )
            ax.set_title(f"{dt.day}/{dt.month}/{dt.year}\n{product.uuid}", fontsize=10)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            with rasterio.open(band_rasters[n], "r") as src:
                plot.show(src, ax=ax, cmap="terrain")

        farm_name_dir = f"{FARM_SUMMARIES_DIRECTORY}/{farm_name}"

        os.makedirs(farm_name_dir, exist_ok=True)
        plt.savefig(f"{farm_name_dir}/{band}.jpg")

        if verify_images:
            plt.show()

    def get_all_farms_raster_for_band(self, raster_path):
        if not os.path.exists(raster_path):
            return None

        raster_file_path = Path(raster_path).with_suffix(".tif")
        raster_file_path = f"{raster_file_path.parent.parent.parent}/IMG_DATA_CROPPED/all_farms/{raster_file_path.parent.name}/{raster_file_path.name}"
        return raster_file_path

    def get_farm_raster_for_band(self, farm_name: str, raster_path: str) -> str:
        """
        Given a farm name and a band path from an original product, construct a path to the farm raster for this band
        :param farm_name:
        :param raster_path:
        :return:
        """
        if not os.path.exists(raster_path):
            return None

        raster_file_path = Path(raster_path).with_suffix(".tif")
        # Construct path to raster band that has been cropped for specified farm
        raster_file_path = (
            f"{raster_file_path.parent.parent.parent}/IMG_DATA_CROPPED/{farm_name}/"
            f"{raster_file_path.parent.name}/{raster_file_path.name}"
        )
        return raster_file_path


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--download", "-d", is_flag=True, help="Download Sentinel 2 data")
@click.option("--crop-all", "-ca", is_flag=True, help="Crop Sentintel 2 rasters to bounds of all farms")
@click.option(
    "--crop-individual-farms", "-ci", is_flag=True, help="Crop Sentintel 2 rasters individual farm boundaries"
)
@click.option("--farm_summaries", "-fs", is_flag=True, help="Generate summary jpegs for specified bands over time")
@click.option(
    "--farm_summaries_all",
    "-fsa",
    is_flag=True,
    help="Generate summary jpegs for the bbox of all farms for specified bands",
)
@click.option(
    "--sentinel_date_range",
    required=True,
    type=(str, str),
    default=("20210401", "20220401"),
    help='Specify the date window to get sentinel data. Default is ("20210401", "20220401").'
    " Has to be combined with -d flag to start download",
)
def main(download, crop_all, crop_individual_farms, sentinel_date_range, farm_summaries, farm_summaries_all):
    """
    Download and process Sentinel 2 rasters.

    If you wish to download (-d), please ensure you set "SENTINEL_USER" and "SENTINEL_PASSWORD"
    environment variables. An account can be created at https://scihub.copernicus.eu/dhus/#/self-registration

    :param farm_summaries:
    :param crop_individual_farms:
    :param crop_all:
    :param download:
    :param sentinel_date_range:
    """
    crop_data_handler = CropDataHandler(sentinel_date_range)

    if download:
        crop_data_handler.download_sentinel_products()

    if crop_all:
        crop_data_handler.crop_rasters_to_all_fields_bbox()

    if crop_individual_farms:
        crop_data_handler.crop_rasters_to_individual_fields_bbox()

    if farm_summaries:
        crop_data_handler.generate_individual_farm_bands_summary()

    if farm_summaries_all:
        crop_data_handler.generate_all_farms_bands_summary()

    # crop_data_handler.create_cropped_rgb_image()
    # crop_data_handler.preview_farm_bands()

    log.debug("Done")


if __name__ == "__main__":
    main()
