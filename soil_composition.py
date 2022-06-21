import json
import logging
import math
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from glob import glob
from itertools import chain, combinations
from logging import handlers
from pathlib import Path
from typing import List
from zipfile import ZipFile

import click
import earthpy.plot as ep
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import shapely
from dateutil import parser
from fiona.crs import from_epsg
from geopandas import GeoDataFrame
from matplotlib import gridspec
from pandas import Series, DataFrame
from pyproj import Transformer
from rasterio import plot
from rasterio.mask import mask
from rasterio.plot import show
from scipy import stats
from sentinelsat.sentinel import SentinelAPI
from shapely.geometry import Polygon, MultiPolygon, shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    plot_confusion_matrix,
)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

MEAN_NDVI = "mean_ndvi"

MEAN_NDWI = "mean_ndwi"

CARBON_RATING = "carbon_rating"

POTASSIUM_RATING = "potassium_rating"

PHOSPHORUS_RATING = "phosphorus_rating"

NITROGEN_RATING = "nitrogen_rating"

pd.set_option("display.max_columns", None)  # or 1000
pd.set_option("display.max_rows", None)  # or 1000
pd.set_option("display.max_colwidth", None)

BAND_2_10M = "B02_10m"
BAND_3_10M = "B03_10m"
BAND_4_10M = "B04_10m"
BAND_8_10M = "B08_10m"
BAND_TCI_10M = "TCI_10m"

BAND_2_20M = "B02_20m"
BAND_3_20M = "B03_20m"
BAND_4_20M = "B04_20m"
BAND_5_20M = "B05_20m"
BAND_6_20M = "B06_20m"
BAND_7_20M = "B07_20m"
BAND_8A_20M = "B8A_20m"
BAND_11_20M = "B11_20m"
BAND_12_20M = "B12_20m"
BAND_SCL_20M = "SCL_20m"
BAND_TCI_20M = "TCI_20m"

BAND_1_60M = "B01_60m"
BAND_2_60M = "B02_60m"
BAND_3_60M = "B03_60m"
BAND_4_60M = "B04_60m"
BAND_5_60M = "B05_60m"
BAND_6_60M = "B06_60m"
BAND_7_60M = "B07_60m"
BAND_8A_60M = "B8A_60m"
BAND_9_60M = "B09_60m"
BAND_11_60M = "B11_60m"
BAND_12_60M = "B12_60m"
BAND_SCL_60M = "SCL_60m"

ALL_BANDS = (
    BAND_2_10M,
    BAND_3_10M,
    BAND_4_10M,
    BAND_8_10M,
    BAND_TCI_10M,
    BAND_2_20M,
    BAND_3_20M,
    BAND_4_20M,
    BAND_5_20M,
    BAND_6_20M,
    BAND_7_20M,
    BAND_8A_20M,
    BAND_11_20M,
    BAND_12_20M,
    BAND_SCL_20M,
    BAND_TCI_20M,
    BAND_1_60M,
    BAND_2_60M,
    BAND_3_60M,
    BAND_4_60M,
    BAND_5_60M,
    BAND_6_60M,
    BAND_7_60M,
    BAND_8A_60M,
    BAND_9_60M,
    BAND_11_60M,
    BAND_12_60M,
    BAND_SCL_60M,
)

REPORT_SUMMARY_BANDS = BAND_TCI_20M, BAND_SCL_20M, BAND_TCI_10M, BAND_SCL_60M
# REPORT_SUMMARY_BANDS = ALL_BANDS

log = logging.getLogger(__name__)
DATA_DIRECTORY = "data"
ALL_FARMS_FILE_NAME_EXCLUDING_PREFIX = "all_farms_24_05_22"
FARM_SENTINEL_DATA_DIRECTORY = f"{DATA_DIRECTORY}/sentinel2"
FARM_SUMMARIES_DIRECTORY = f"{FARM_SENTINEL_DATA_DIRECTORY}/farm_summaries"
FARM_LOCATIONS_DIRECTORY = f"{DATA_DIRECTORY}/farm_locations"
FARMS_GEOJSON = f"{FARM_LOCATIONS_DIRECTORY}/{ALL_FARMS_FILE_NAME_EXCLUDING_PREFIX}.geojson"
FARMS_XLSX = f"{FARM_LOCATIONS_DIRECTORY}/MASTER_DOCUMENT_V1_240522.xlsx"
NEW_KMLS = f"{FARM_LOCATIONS_DIRECTORY}/KML_Soil/KMLs"
SENTINEL_PRODUCTS_GEOJSON = f"{FARM_SENTINEL_DATA_DIRECTORY}/products.geojson"
FARMS_GEOJSON_VALID_PRODUCTS = (
    f"{FARM_LOCATIONS_DIRECTORY}/{ALL_FARMS_FILE_NAME_EXCLUDING_PREFIX}_cloud_free_products.geojson"
)
INDIVIDUAL_BOUNDS_SHAPEFILE = f"{FARM_LOCATIONS_DIRECTORY}/individual_farm_bounds.shp"
ANALYSIS_RESULTS_DIR = f"{FARM_SUMMARIES_DIRECTORY}/analysis"
# Scene classification keys
CLOUD_MEDIUM = 8
CLOUD_HIGH = 9
THIN_CIRRUS = 10
SATURATED_OR_DEFECTIVE = 1
CLOUD_SHADOWS = 3

# Create logs dir if it doesn't already exist
os.makedirs("logs", exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler("logs/crop.log", maxBytes=2048, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s] %(message)s",
    handlers=[file_handler, logging.StreamHandler()],
)


@dataclass
class DataHandler:
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

        if not os.path.exists(FARMS_XLSX):
            sys.exit(f"Unable to find file {FARMS_XLSX} - aborting")

        # Ensure directory to store Sentinel2 data exists
        os.makedirs(FARM_SENTINEL_DATA_DIRECTORY, exist_ok=True)
        os.makedirs(FARM_SUMMARIES_DIRECTORY, exist_ok=True)
        os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)

        self.extract_geometries()

    def convert_geometry_string_to_shape(self, geom_str: str) -> shape:
        """
        Parse a geometry string into a shape
        :return:
        """
        try:
            return shape(json.loads(geom_str))
        except (TypeError, AttributeError):  # Handle NaN and empty strings
            return None

    def parse_excel_and_kml_farms(self):

        field_id_farms_df = pd.read_excel(
            FARMS_XLSX,
            sheet_name="Farms identified by FieldID",
            header=[1],
            usecols=("field_id", "farm_name", "field_boundary", "Date of survey"),
        )
        g_num_df = pd.read_excel(
            FARMS_XLSX,
            sheet_name="Farms identified by G no",
            header=[1],
            usecols=("Sl.No", "Farmer name", "Date of survey"),
        )

        self._sanitize_dataframe_column_headers(field_id_farms_df)

        # Standardise column names
        g_num_df.rename(columns={"Sl.No": "field_id", "Farmer name": "farm_name"}, inplace=True)
        self._sanitize_dataframe_column_headers(g_num_df)

        # Convert geometry to shape
        field_id_farms_df["field_boundary"] = field_id_farms_df["field_boundary"].apply(
            self.convert_geometry_string_to_shape
        )

        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"

        def _read_kmz_to_geopandas(kmz_path):
            kmz = ZipFile(kmz_path, "r")
            kml = kmz.open("doc.kml", "r").read()
            kml_path = Path(kmz_path).with_suffix(".kml")
            with open(kml_path, "wb") as kml_file:
                kml_file.write(kml)
            farm_df = gpd.read_file(kml_path, driver="KML")
            # Add file name to dataframe
            farm_df["field_id"] = Path(kmz_path).stem
            return farm_df

        latest_farms = glob(
            f"{NEW_KMLS}/*.kmz",
            recursive=False,
        )

        # Read list of kmz files into dataframe
        kmz_df = pd.concat(map(_read_kmz_to_geopandas, latest_farms))
        kmz_df.rename(columns={"geometry": "field_boundary"}, inplace=True)

        r = re.compile("([a-zA-Z]+)([0-9]+)")
        # Pad out the ids with 0s so we can join with ids in the master spreadsheet
        kmz_df["field_id"] = kmz_df["field_id"].map(lambda x: f"G{r.match(x).groups()[1].zfill(3)}")

        # join g_num_df with kmz to get field_boundary geometry
        g_num_df = pd.merge(g_num_df, kmz_df[["field_id", "field_boundary"]], on="field_id", how="left")

        # Combine into single fields list
        fields_df = pd.concat([field_id_farms_df, g_num_df])
        fields_df.reset_index(drop=True, inplace=True)
        fields_df.rename(columns={"field_boundary": "geometry"}, inplace=True)

        # Read in soil test results and match to farms
        soil_df = pd.read_excel(
            FARMS_XLSX,
            sheet_name="Soil test results",
        )
        self._sanitize_dataframe_column_headers(soil_df)
        soil_df.rename(
            columns={
                "field_ids": "field_id",
                "o.c._(1_=_low;_2_=_med;_3_=_high)": "carbon_rating",
                "n(1_=_low;_2_=_med;_3_=_high)": "nitrogen_rating",
                "p_(1_=_low;_5_=_high)": "phosphorus_rating",
                "k_(1_=_low;_2_=_med;_3_=_high)": "potassium_rating",
            },
            inplace=True,
        )
        soil_df["carbon_rating"] = soil_df["carbon_rating"].fillna(0).astype(np.int64)
        soil_df["nitrogen_rating"] = soil_df["nitrogen_rating"].fillna(0).astype(np.int64)
        soil_df["phosphorus_rating"] = soil_df["phosphorus_rating"].fillna(0).astype(np.int64)
        soil_df["potassium_rating"] = soil_df["potassium_rating"].fillna(0).astype(np.int64)

        # Remove duplicate - we already have this as farm_name
        soil_df.drop("farmer_name", axis=1, inplace=True)

        fields_df = pd.merge(fields_df, soil_df, on="field_id", how="left")
        fields_df["date_of_survey"] = pd.to_datetime(fields_df["date_of_survey"])
        # d.strftime("%d/%m/%Y")
        self.farm_bounds_32643 = gpd.GeoDataFrame(fields_df, geometry="geometry")
        self.farm_bounds_32643.set_crs(epsg=4326, inplace=True)
        # Save with 4326 for sentinel api
        self.farm_bounds_32643.to_file(FARMS_GEOJSON, driver="GeoJSON")

        self.farm_bounds_32643 = self.farm_bounds_32643.to_crs({"init": "epsg:32643"})
        # Save subset of fields to shapefile - can't save date
        self.farm_bounds_32643[["field_id", "farm_name", "geometry"]].to_file(INDIVIDUAL_BOUNDS_SHAPEFILE)

        # Save overall bounding box in desired projection
        self.total_bbox_32643 = shapely.geometry.box(*self.farm_bounds_32643.total_bounds, ccw=True)

        self.total_bbox = gpd.GeoDataFrame({"geometry": self.total_bbox_32643}, index=[0], crs=from_epsg(32643))
        self.total_bbox.to_file(f"{FARM_LOCATIONS_DIRECTORY}/total_bounds.shp")

        # Update the geometry in farms datafile to make it 2D so rasterio can handle it.
        # It seems rasterio won't work with 3D geometry
        self.farm_bounds_32643.geometry = self.convert_3D_2D(self.farm_bounds_32643.geometry)

    def _sanitize_dataframe_column_headers(self, dataframe: DataFrame):
        """
        Strip, lower case and replace spaces with underscores
        :param dataframe:
        :return:
        """
        dataframe.columns = [c.strip().lower().replace(" ", "_") for c in dataframe.columns]

    def extract_geometries(self):
        """
        Unzip the kmz and derive shapefiles, geojson and cache farm bounds and total bounding box
        """

        if os.path.exists(FARMS_GEOJSON_VALID_PRODUCTS):
            self.load_farms_with_valid_products()
        else:
            self.parse_excel_and_kml_farms()

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
            else:
                # We don't have to do anything to this geometry
                new_geo.append(p)
        return new_geo

    def save_farms_with_valid_products(self):
        """
        Save farms df to geojson. List of valid products has to be converted to a string to serialize
        :return:
        """

        # Copy as we don't want to make changes to the instance version, only the version we are serialising
        copied_farm_bounds = self.farm_bounds_32643.copy(deep=True)
        copied_farm_bounds["cloud_free_products"] = copied_farm_bounds["cloud_free_products"].apply(
            lambda x: " ".join(x)
        )
        copied_farm_bounds.to_file(FARMS_GEOJSON_VALID_PRODUCTS, driver="GeoJSON")

    def load_farms_with_valid_products(self):
        """
        Load farms dataframe from geojson. Convert valid products back into list
        :return:
        """
        self.farm_bounds_32643 = gpd.read_file(FARMS_GEOJSON_VALID_PRODUCTS)

        # Convert back to list
        self.farm_bounds_32643["cloud_free_products"] = self.farm_bounds_32643["cloud_free_products"].apply(
            lambda x: x.split()
        )

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

    def download_sentinel_product_files(self):
        """
        Attempt to download each product in the dataframe
        :return:
        """

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

        # Granules T43PFS and T43PGS contain all the farms
        required_granules = ("T43PFS", "T43PGS")

        self.products_df = self.products_df.loc[
            self.products_df["title"].str.contains("|".join(required_granules), case=False)
        ]

        log.debug(f"{len(self.products_df)} products available for tiles {required_granules}")

        if verify_products:
            # Various plots below to debug product and farm positions

            # Read farm bounds in in same crs as products here for easy comparison (4326)
            farm_bounds = gpd.read_file(FARMS_GEOJSON)

            # Simple plot to show product positions
            plot = self.products_df.plot(column="uuid", cmap=None)
            # plt.savefig("test.jpg")
            plt.show()

            # Product positions with uuids overlaid
            ax = self.products_df.plot(column="uuid", cmap=None, figsize=(20, 20), alpha=0.3)
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

            farm_bounds.plot(ax=ax, column="Name", cmap=None, figsize=(50, 50))
            farm_bounds.apply(
                lambda x: ax.annotate(text=x["Name"], xy=x.geometry.centroid.coords[0], ha="center"), axis=1
            )
            ax.set_title("Fields on sentinel data", fontsize=10, pad=10)
            plt.show()

    def download_sentinel_products(self):
        """
        Get available sentinel 2 products and download
        :return:
        """

        # Set up api with credentials
        self.configure_api()

        self.get_available_sentinel_products_df(verify_products=False)

        self.download_sentinel_product_files()

        self.unzip_sentinel_products()

        # Validate results
        unzipped_products = glob(
            f"{FARM_SENTINEL_DATA_DIRECTORY}/*.SAFE",
            recursive=False,
        )

        # Add paths to the various product bands
        self.products_df = self.products_df.apply(self.add_band_paths_to_product, axis=1)

        # Save the products so we have a record
        self.save_products_df_to_geojson()

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

        def _unzip_if_required(sentinel_zip):
            file_path = f"{FARM_SENTINEL_DATA_DIRECTORY}/{sentinel_zip}"
            unzipped_filename = Path(file_path).with_suffix(".SAFE")
            if not os.path.exists(unzipped_filename):
                if zipfile.is_zipfile(file_path):
                    with zipfile.ZipFile(file_path) as item:
                        log.debug(f"Unzipping {file_path}…")
                        item.extractall(FARM_SENTINEL_DATA_DIRECTORY)
            else:
                log.debug(f"Not unzipping as {unzipped_filename} already exists")

        [
            _unzip_if_required(sentinel_zip)
            for sentinel_zip in os.listdir(FARM_SENTINEL_DATA_DIRECTORY)
            if sentinel_zip.endswith(".zip")
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

    def crop_product_rasters_to_all_fields(self, product: Series):
        """
        For every row in the products dataframe, crop the downloaded rasters to overall farm bounds
        :param product: Series
        :return: product:Series
        """

        original_rasters: list = self.get_original_product_rasters(product)

        if original_rasters:

            # Crop all original rasters to all farms geom
            [self.crop_raster_to_geometry(raster, self.total_bbox_32643, "all_farms") for raster in original_rasters]

        else:
            log.debug(f"Skipping as product {product['title']} as associated rasters not found")

        return product

    def add_band_paths_to_product(self, product: Series):
        """
        For every row in the products dataframe
        add band paths to products dataframe
        :param product: Series
        :return: product:Series
        """

        original_rasters: list = self.get_original_product_rasters(product)

        if original_rasters:

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
        self.products_df = self.products_df.apply(self.crop_product_rasters_to_all_fields, axis=1)

        # Save updated products
        self.save_products_df_to_geojson()

        log.debug("All rasters successfully cropped to overall farms bbox")

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

            field_id = farm["field_id"]
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

            # Check that the farm geometry is within the product -  this is not always the case as we have products from different granules
            if product.geometry.contains(farm_geometry):
                original_rasters = self.get_original_product_rasters(product)

                if original_rasters:
                    # Crop all rasters to individual farm bboxes
                    [self.crop_raster_to_geometry(raster, farm_geometry, field_id) for raster in original_rasters]

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

        # Make sure geometries are in the correct crs as we have to check whether each farm polygon is within each product
        self.products_df = self.products_df.to_crs({"init": "epsg:32643"})
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
            self.get_farm_raster_from_product_raster_path(farm_name, first_product[band])
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

        band2 = self.get_farm_raster_from_product_raster_path(farm_name, first_product[BAND_2_10M])
        band3 = self.get_farm_raster_from_product_raster_path(farm_name, first_product[BAND_3_10M])
        band4 = self.get_farm_raster_from_product_raster_path(farm_name, first_product[BAND_4_10M])

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

    def generate_individual_farm_bands_summary(self, filter_clouds=True):
        """
        For each farm, plot each of the summary bands in a jpeg over time
        :param filter_clouds: Whether to exclude farm rasters that have clouds
        :return:
        """

        [
            self.generate_individual_farm_band_summary(
                farm_index, band, verify_images=False, filter_clouds=filter_clouds
            )
            for farm_index in range(len(self.farm_bounds_32643))
            for band in REPORT_SUMMARY_BANDS
        ]

    def generate_individual_farm_band_summary(
        self, farm_df_index: int, band: str, verify_images=False, filter_clouds=True
    ):
        """
        Plot how specified band_to_display changes over time for a farm
        :param filter_clouds: Filter out images that Scene Classification deemed to have clouds
        :param band: Band you wish to display
        :param farm_df_index: Index of farm in farms dataframe
        :param verify_images: Show the plot
        :return:
        """

        # Get the farm
        farm_details = self.get_farm_from_dataframe(farm_df_index)

        field_id = farm_details["field_id"]
        filtered_products_df = self.load_farm_cloud_free_products_df(farm_details)

        # FIXME : We are only doing cloud free products now
        # if filter_clouds:
        #     filtered_products_df = self.get_cloud_free_products_for_farm(band, farm_details)
        # else:
        #     filtered_products_df = self.products_df[self.products_df[band].notnull()]

        # # Filter products for areas other than this farm
        # filtered_products_df = filtered_products_df[filtered_products_df.geometry.contains(farm_details.geometry)]
        #
        # try:
        #     filtered_products_df.reset_index(inplace=True)
        # except ValueError:
        #     # Ignore - this can arise if reset_index has already been called as it is in get_cloud_free_products_for_farm
        #     pass

        number_of_raster = len(filtered_products_df)

        cols = 6
        rows = int(math.ceil(number_of_raster / cols))

        gs = gridspec.GridSpec(rows, cols, wspace=0.01)

        fig = plt.figure(figsize=(24, 24))
        fig.suptitle(f"Farm {farm_df_index}: {field_id} {band}, all products", fontsize=40)

        def _add_band_image_to_grid(product, band_to_display):
            index = product.name
            ax = fig.add_subplot(gs[index])

            dt = parser.parse(product.generationdate)

            # ax.set_title(f"{product.title}:\n{dt.day}/{dt.month}/{dt.year}", fontsize = 10, wrap=True )
            ax.set_title(f"{dt.day}/{dt.month}/{dt.year}\n{product.uuid}", fontsize=10)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            raster_path = product[band_to_display]
            if raster_path:
                with rasterio.open(raster_path, "r") as src:
                    plot.show(src, ax=ax, cmap="terrain")

            return product

        filtered_products_df.apply(_add_band_image_to_grid, band_to_display=band, axis=1)

        farm_name_dir = f"{FARM_SUMMARIES_DIRECTORY}/{field_id}"

        os.makedirs(farm_name_dir, exist_ok=True)

        if filter_clouds:
            plt.savefig(f"{farm_name_dir}/{band}.jpg")
        else:
            plt.savefig(f"{farm_name_dir}/{band}_including_clouds.jpg")
        if verify_images:
            plt.show()

    def get_cloud_free_products_for_farm(self, band: str, farm_details: Series) -> GeoDataFrame:
        """
        Get dataframe containing cloud free products for the specified farm
        :param band:
        :param farm_details:
        :return: filtered_products_df GeoDataFrame
        """
        cloud_free_products = farm_details["cloud_free_products"]
        # Only want products that are cloud free for specified farm. Copy, filter and reset index
        filtered_products_df = self.products_df.query("uuid in @cloud_free_products").copy()
        filtered_products_df = filtered_products_df[filtered_products_df[band].notnull()]
        filtered_products_df.reset_index(inplace=True)
        return filtered_products_df

    def get_farm_from_dataframe(self, farm_df_index: int) -> Series:
        """
        Safely retrieve farm details at specified index
        :param farm_df_index:
        :return: Farm Series
        """
        try:
            farm_details = self.farm_bounds_32643.iloc[farm_df_index]
        except IndexError as e:
            log.error(e)
            sys.exit("Farm index provided is out of range - exiting")
        return farm_details

    def set_cloud_free_products(self, farm: Series) -> Series:
        """
        Check the scene classification raster for each product for this farm
        :param farm:
        :return: farm
        """

        cloud_free_product_ids = []

        def _check_cloud_cover_for_farm_product(uuid: str, default_scene_classification_path: str):
            """
            Get the scene classification raster for the specified farm.
            If it contains any pixels classed as cloud, we ignore it. If not we add it to a valid list of
            product uuids. See https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/#
            :param uuid:
            :param default_scene_classification_path:

            """
            scene_classification_raster = self.get_farm_raster_from_product_raster_path(
                farm["field_id"], default_scene_classification_path
            )

            if scene_classification_raster:
                with rasterio.open(scene_classification_raster, "r") as src:
                    if not any(
                        np.in1d(
                            (CLOUD_MEDIUM, CLOUD_HIGH, THIN_CIRRUS, SATURATED_OR_DEFECTIVE, CLOUD_SHADOWS), src.read()
                        )
                    ):
                        cloud_free_product_ids.append(uuid)

        [
            _check_cloud_cover_for_farm_product(uuid, default_scene_classification_20m_path)
            for uuid, default_scene_classification_20m_path in zip(
                self.products_df["uuid"], self.products_df[BAND_SCL_20M]
            )
        ]

        farm["cloud_free_products"] = cloud_free_product_ids
        return farm

    def add_cloud_free_products_to_farms_df(self):
        """
        Add a list of cloud free product uuids for each farm
        :return:
        """

        if not os.path.exists(FARMS_GEOJSON_VALID_PRODUCTS):
            self.farm_bounds_32643 = self.farm_bounds_32643.apply(self.set_cloud_free_products, axis=1)
            self.save_farms_with_valid_products()
        else:
            self.load_farms_with_valid_products()

    def generate_individual_farm_cloud_series_over_time(self, farm_df_index: int, verify_images=False):
        """
        Generate composite plot of farm true colour (RGB) images for each product
        :param farm_df_index: Index of farm in farms dataframe
        :param verify_images: Show the plot
        :return:
        """

        # Get the farm
        farm_details = self.get_farm_from_dataframe(farm_df_index)

        field_id = farm_details["field_id"]

        # If we have a situation where we've not yet downloaded all the products in the dataframe, we filter out those
        # where we haven't got the desired band
        # filtered_products_df = self.products_df[self.products_df[BAND_TCI_10M].notnull()]
        filtered_products_df = self.products_df[self.products_df[BAND_SCL_20M].notnull()]

        true_colour_rasters = [
            self.get_farm_raster_from_product_raster_path(field_id, band)
            for band in filtered_products_df[BAND_TCI_10M]
        ]
        scene_classification_rasters = [
            self.get_farm_raster_from_product_raster_path(field_id, band)
            for band in filtered_products_df[BAND_SCL_20M]
        ]

        true_colour_rasters = list(filter(None, true_colour_rasters))

        number_of_raster = len(true_colour_rasters)

        cols = 4
        rows = int(math.ceil(number_of_raster / cols))

        # gs = gridspec.GridSpec(rows, cols, wspace=0.01)

        fig = plt.figure(figsize=(24, 24))
        # gs = gridspec.GridSpec(rows, cols, wspace=0.01,figure=fig)
        gs = gridspec.GridSpec(rows, cols, wspace=1, figure=fig)
        # gridspec.GridSpec(1, 2, figure=fig)
        fig.suptitle(f"Farm {farm_df_index}: {field_id} true colour images, all products", fontsize=40)

        for n in range(number_of_raster):

            gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[n])

            ax = fig.add_subplot(gs1[0])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            product = filtered_products_df.iloc[n]

            dt = parser.parse(product.generationdate)

            ax.set_title(f"{dt.day}/{dt.month}/{dt.year}", fontsize=10)
            with rasterio.open(true_colour_rasters[n], "r") as src:
                plot.show(src, ax=ax, cmap="terrain")

            ax2 = fig.add_subplot(gs1[1])
            ax2.axes.xaxis.set_visible(False)
            ax2.axes.yaxis.set_visible(False)
            ax2.set_title(f"Clear {dt.day}/{dt.month}/{dt.year}")
            with rasterio.open(scene_classification_rasters[n], "r") as src:

                data = src.read()

                if any(np.in1d((CLOUD_MEDIUM, CLOUD_HIGH, THIN_CIRRUS), data)):
                    log.debug("Raster data contains clouds")
                    ax2.set_title(f"*******CLOUD ALERT ******")
                plot.show(src, ax=ax2, cmap="terrain")
                # fig, ax = plt.subplots(figsize=(15, 15))
                # plot.show(src, ax=ax)
                # show_hist(
                #     src, bins=50, lw=0.0, stacked=False, alpha=0.3,
                #     histtype='stepfilled', title="Histogram")
                # show_hist(src, bins=50, histtype='stepfilled',
                #                   lw=0.0, stacked=False, alpha=0.3, ax=ax)
                # plt.show()
                # plot.show(src, ax=ax2, cmap="terrain")

            # gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
            # fig2 = plt.figure()

            # ax = fig.add_subplot(gs2[0])
            #
            # product = filtered_products_df.iloc[n]
            #
            # dt = parser.parse(product.generationdate)
            #
            # # ax.set_title(f"{product.title}:\n{dt.day}/{dt.month}/{dt.year}", fontsize = 10, wrap=True )
            # ax.set_title(f"{dt.day}/{dt.month}/{dt.year}", fontsize=10)
            #
            # ax.axes.xaxis.set_visible(False)
            # ax.axes.yaxis.set_visible(False)
            #
            # with rasterio.open(true_colour_rasters[n], "r") as src:
            #     plot.show(src, ax=ax, cmap="terrain")
            # with rasterio.open(true_colour_rasters[n], "r") as src:
            #     ax = fig.add_subplot(gs2[1])

            # ax.set_title("cloud", fontsize=10)
            #
            # ax.axes.xaxis.set_visible(False)
            # ax.axes.yaxis.set_visible(False)
            # plot.show(src, ax=ax, cmap="terrain")

            # fig.add_subplot(gs[n])

        # plt.savefig(f"{FARM_SUMMARIES_DIRECTORY}/{farm_name}.jpg")

        if verify_images:
            plt.show()

    def get_all_farms_raster_for_band(self, raster_path):
        if not os.path.exists(raster_path):
            return None

        raster_file_path = Path(raster_path).with_suffix(".tif")
        raster_file_path = f"{raster_file_path.parent.parent.parent}/IMG_DATA_CROPPED/all_farms/{raster_file_path.parent.name}/{raster_file_path.name}"
        return raster_file_path

    def get_farm_raster_from_product_raster_path(self, field_id: str, raster_path: str) -> str:
        """
        Given a farm name and a band path from an original product, construct a path to the farm raster for this band
        :param field_id:
        :param raster_path:
        :return:
        """
        if not raster_path:
            return None

        if not os.path.exists(raster_path):
            return None

        raster_file_path = Path(raster_path).with_suffix(".tif")
        # Construct path to raster band that has been cropped for specified farm
        raster_file_path = (
            f"{raster_file_path.parent.parent.parent}/IMG_DATA_CROPPED/{field_id}/"
            f"{raster_file_path.parent.name}/{raster_file_path.name}"
        )
        return raster_file_path if os.path.exists(raster_file_path) else None

    def get_pixel_for_location_all_products(self, farm_index: int, band: str, x: float, y: float):
        """
        Given a farm index, get the farm's cloud free products. For each, get the pixel value in the specified band raster.
        :param farm_index:
        :param band:
        :param x:
        :param y:
        :return:
        """

        # Get cloud free products
        cloud_free_products = self.get_cloud_free_products_for_farm(band, self.get_farm_from_dataframe(farm_index))

        pixel_values = [
            self.get_pixel_for_location_for_specified_product_and_farm(
                product_band=product_band, farm_index=farm_index, band=band, x=x, y=y
            )
            for product_band in cloud_free_products[band]
        ]
        return pixel_values

    def get_pixel_for_location_for_specified_product_and_farm(
        self, product_band: str, farm_index: int, band: str, x: float, y: float
    ):
        """
        Given a farm and a band, get the pixel value for the specified location
        :param farm_index:
        :param band:
        :return:
        """

        # Get path for farm
        farm_raster = self.get_farm_raster_from_product_raster_path(
            self.get_farm_from_dataframe(farm_index).Name, product_band
        )
        if farm_raster:
            with rasterio.open(farm_raster) as src:
                # convert coordinate to raster projection
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xx, yy = transformer.transform(x, y)

                # get value from grid.
                pixel_values = list(src.sample([(xx, yy)]))[0]
                print(pixel_values)
                # Returns if coords are outwith the raster
                return pixel_values[0]
                # Alternative src.index(xx, yy)
                # r = src.read(1)
                # r[src.index(xx, yy)]
                # p_values = src.index(xx, yy)
                #
                #
                # # To sanity check
                # aff = src.transform
                # loc = rowcol(aff, xx, yy)
                #
                # # Get x and y of pixel at specified row and column
                # test = src.transform * loc
                # res = xy(transform=aff, rows=loc[0], cols=loc[1])
                # t = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
                # check = t.transform(*res)
                #
                # left, bottom, right, top = src.bounds
                # test = rowcol(aff, left, top)
                # print(test)

    def generate_band_histogram(self, product, field_id):
        def _open_band(band):
            with rasterio.open(band) as f:
                return f.read(1)

        arrs = [
            _open_band(self.get_farm_raster_from_product_raster_path(field_id, product[band]))
            for band in (BAND_2_10M, BAND_3_10M, BAND_4_10M, BAND_8_10M)
        ]

        sentinel_img = np.array(arrs, dtype=arrs[0].dtype)
        show(sentinel_img[0:3])
        rasterio.plot.show_hist(
            sentinel_img, bins=50, histtype="stepfilled", lw=0.0, stacked=False, alpha=0.3, title="10m bands"
        )
        log.debug("here")

    def generate_mean_ndvi(self, product, field_id):
        """
        Calculate the mean NDVI for a farm for the specified product. Save the results as a raster
        :param product:
        :return: product Series
        """

        red_path = product[BAND_4_10M]
        nir_path = product[BAND_8_10M]

        if red_path and nir_path:

            with rasterio.open(red_path) as red:
                red_band = red.read(1)
                out_meta = red.meta.copy()
            with rasterio.open(nir_path) as nir:
                nir_band = nir.read(1)

            # Allow division by zero
            np.seterr(divide="ignore", invalid="ignore")
            ndvi = (nir_band.astype(float) - red_band.astype(float)) / (
                nir_band.astype(float) + red_band.astype(float)
            )

            # Get the mean (discounting nan values)
            product[f"{field_id}_mean_ndvi"] = np.nanmean(ndvi)

            return product

            # ep.plot_bands(ndvi, cmap="YlGnBu", cols=1, title="ndvi", vmin=-1, vmax=1)

            # plt.imshow(ndvi)
            # plt.show()
            # vmin, vmax = np.nanpercentile(ndvi, (1,99))
            # # img_plt = plt.imshow(ndvi, cmap='gray', vmin=vmin, vmax=vmax)
            # plt.imshow(ndvi, cmap='Greens', vmin=vmin, vmax=vmax)
            # show(ndvi, cmap="Greens")

            # ndvi_raster_path = f"{Path(red_path).parent}/ndvi.tif"

            # out_meta.update(dtype=rasterio.float32, count=1)
            # with rasterio.open(
            #     ndvi_raster_path,
            #     "w",
            #     **out_meta,
            # ) as ndvi_out:
            #     ndvi_out.write(ndvi, 1)

            # show(ndvi, cmap="Greens")

            # Verify
            # with rasterio.open(ndvi_raster_path, "r") as src:
            #     fig, ax = plt.subplots(figsize=(15, 15))
            #     show(src, ax=ax, cmap="Greens")
            #     plt.show()

        return product

    def generate_band_means_at_soil_test_date(self):
        """
        For each farm, get the mean of each band and store in Farm Series
        Results are then persisted to filesystem
        :return:
        """

        def _calculate_means(farm_details: Series):
            """
            Get the mean value of each band and store in farm_details series
            :param farm_details:
            :return:
            """
            cloud_free_products_df = self.load_farm_cloud_free_products_df(farm_details)
            survey_date = farm_details["date_of_survey"]

            # Ignore the few fields where we don't have a date
            if pd.isnull(survey_date):
                return farm_details

            # Copy the date so we can easily access it later
            cloud_free_products_df["gen_date"] = cloud_free_products_df["generationdate"]

            # Convert and set date as index
            cloud_free_products_df["generationdate"] = pd.to_datetime(cloud_free_products_df["generationdate"])
            cloud_free_products_df = cloud_free_products_df.set_index("generationdate")

            field_id = farm_details["field_id"]
            # ****************  WARNING *************
            # Only add field_ids_with_clouds if you understand what is going on here.
            #
            # The idea behind this is assuming you have run the perform_analysis endpoint, examine the plots of all
            # the farms we are going to analyse.  The logic to automatically remove clouds isn't perfect and some cloudy
            # fields get through. If you specify their ids here, we look for the next nearest product and drop this from
            # the farm's list of cloud free products.
            # Running this repeatedly with the same ids will keep removing available products and could mess up the dataset.
            # Run it once, check the plots and if all have been fixed, set field_ids_with_clouds = []. If there are some
            # which haven't, keep their ids in the list, run again to select the next nearest product again, check the plot and repeat.
            # Simple as that ;)
            # *************************************
            # field_ids_with_clouds = ('660', '659', '821', '819', '787', '820', '823', 'G044', 'G066', 'G067', 'G082')
            field_ids_with_clouds = []

            if field_id not in field_ids_with_clouds:
                # Get the nearest product
                nearest_product_index = cloud_free_products_df.index.get_indexer([survey_date], method="nearest")[0]
                nearest_product = cloud_free_products_df.iloc[nearest_product_index]
            else:
                # Get next nearest product as manual inspection showed clouds

                # Get what we thought was cloud free
                nearest_uuid = farm_details["nearest_product_uuid"]
                # Remove the cloud covered product
                cloud_free_products_df = cloud_free_products_df[cloud_free_products_df.uuid != nearest_uuid]

                # Get next nearest
                nearest_product_index = cloud_free_products_df.index.get_indexer([survey_date], method="nearest")[0]
                nearest_product = cloud_free_products_df.iloc[nearest_product_index]

                # Update this farms' list of cloud free products
                original_cloud_free_products = farm_details["cloud_free_products"]
                original_cloud_free_products.remove(nearest_uuid)
                farm_details["cloud_free_products"] = original_cloud_free_products

                # Update the geojson of cloud free products
                cloud_free_products_df.to_file(
                    self.get_farm_cloud_free_products_df_path(farm_details), driver="GeoJSON"
                )

            def _mean_from_band(band: str):
                """
                Get the mean from the specified band
                :param band:
                :return:
                """
                with rasterio.open(band) as f:
                    return np.nanmean(f.read(1))

            def _assign_band_mean_to_farm(band: str):
                """
                Update the farm_details series with the mean value from the specified band
                :param band:
                :return:
                """
                farm_details[band] = _mean_from_band(nearest_product[band])

            # Add bands to farm_details
            list(map(_assign_band_mean_to_farm, ALL_BANDS))

            # Add mean ndvi
            red = farm_details[BAND_4_10M]
            nir = farm_details[BAND_8_10M]

            np.seterr(divide="ignore", invalid="ignore")
            ndvi = (nir - red) / (nir + red)

            farm_details["mean_ndvi"] = ndvi

            # ndwi
            # https://en.wikipedia.org/wiki/Normalized_difference_water_index
            green = farm_details[BAND_3_20M]
            nir = farm_details[BAND_8A_20M]

            ndwi = (green - nir) / (green + nir)

            farm_details["mean_ndwi"] = ndwi

            # Add the details of the product we used in case we have to inspect it in future
            farm_details["nearest_product_uuid"] = nearest_product["uuid"]
            farm_details["nearest_product_generationdate"] = nearest_product["gen_date"]

            return farm_details

        self.farm_bounds_32643 = self.farm_bounds_32643.apply(_calculate_means, axis=1)

        # Save the updated farms list
        self.save_farms_with_valid_products()

    def _plot_rgb_for_fields_chosen_for_analysis(self):
        """
        Create a grid of the rgb images of all the fields we have selected for analysis. This is important
        as we have to manually check there are no cloudy images - some sneak through the automatic detection.
        NOTE: If you notice any cloudy fields, take a note of the id, look at generate_band_means_at_soil_test_date
        and amend the field_ids_with_clouds list with the ids. Run generate_band_means_at_soil_test_date on it's own,
        then call this function to generate another plot of all farms. If some farms are still cloudy, repeat. If all
        fields are cloud free, remove the ids from field_ids_with_clouds so we don't accidentally remove any products
        that are don't have any issues
        """
        df = self.get_farm_bounds_as_pandas_df_for_analysis()
        number_of_raster = len(df)

        cols = 4
        rows = int(math.ceil(number_of_raster / cols))

        gs = gridspec.GridSpec(rows, cols, wspace=0.01)

        fig = plt.figure(figsize=(24, 100))
        fig.suptitle(f"Farms", fontsize=40)

        def _plot(farm):
            cloud_free_products_df = self.load_farm_cloud_free_products_df(farm)
            match = cloud_free_products_df[cloud_free_products_df["uuid"] == farm["nearest_product_uuid"]]
            index = farm.name
            ax = fig.add_subplot(gs[index])
            ax.set_title(f"{farm['field_id']}", fontsize=10)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            with rasterio.open(match[BAND_TCI_10M].iloc[0], "r") as src:
                plot.show(src, ax=ax, cmap="terrain")

            return farm

        df.apply(_plot, axis=1)

        plt.savefig(f"{ANALYSIS_RESULTS_DIR}/chosen_products.jpg")

        plt.show()

    def _correlation_plots(self, soil_test_columns: list, field: str):
        """
        Plot correlation matrices
        :param soil_test_columns:
        :param field:
        :return:
        """

        # Get fresh dataframe
        df = self.get_farm_bounds_as_pandas_df_for_analysis()

        # Discount rows that have 0 for test result as no results were supplied
        df = df[df[field] > 0]

        extra_columns = list(set(soil_test_columns) - set([field]))
        df.drop(extra_columns, axis=1, inplace=True)

        sns.lmplot(x=field, y=MEAN_NDWI, data=df)
        sns.lmplot(x=MEAN_NDVI, y=MEAN_NDWI, data=df)

        # Pearsons coefficient by default
        cormat = df.corr()
        r = round(cormat, 2)
        sns.set(rc={"figure.figsize": (25, 15)})
        sns.heatmap(r, annot=True, vmax=1, vmin=-1, center=0, cmap="vlag")
        plt.show()

        # Remove top half to make it easier to read
        mask = np.triu(np.ones_like(r, dtype=bool))
        sns.heatmap(r, annot=True, vmax=1, vmin=-1, center=0, cmap="vlag", mask=mask)

        plt.savefig(f"{ANALYSIS_RESULTS_DIR}/{field}_correlation.png")
        plt.show()
        corr_pairs = r.unstack()
        sorted_pairs = corr_pairs.sort_values(kind="quicksort")
        log.debug(sorted_pairs)

        negative_pairs = sorted_pairs[sorted_pairs < 0]
        log.debug(negative_pairs)

        strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

        log.debug(strong_pairs)

        stats.pearsonr(df[field], df[MEAN_NDVI])
        stats.pearsonr(df[MEAN_NDWI], df[MEAN_NDVI])

        sns.pairplot(df[[BAND_8_10M, BAND_2_10M, BAND_4_10M, BAND_3_10M, CARBON_RATING]], diag_kind="kde")

    def _test_linear_svc_classifier_with_different_band_combinations_inputs(self):
        """
        Test LinearSVC classifier with different combinations of the 10 and 20m bands
        Scores are saved to the farm_summaries_analysis directory in csv and plots
        """

        # Only use a subset of bands as we want to test the model with all combinations
        # If we used all the band means available, we would have over 16 millions combinations!

        # 4096 combinations with this list
        test_bands = [
            BAND_8_10M,
            BAND_2_10M,
            BAND_4_10M,
            BAND_3_10M,
            BAND_3_20M,
            BAND_4_20M,
            BAND_5_20M,
            BAND_6_20M,
            BAND_7_20M,
            BAND_8A_20M,
            BAND_11_20M,
            BAND_12_20M,
        ]

        # test_bands = [
        #     BAND_8_10M,
        #     BAND_2_10M,
        # ]

        def _all_subsets(bands: list):
            """
            Get all combinations of the supplied bands
            :param bands:
            :return:
            """
            # Taken from https://stackoverflow.com/a/5898031
            return chain(*map(lambda x: combinations(bands, x), range(0, len(bands) + 1)))

        def _plot_results(results, soil_test_field):
            bands_scores = pd.DataFrame.from_dict(results, orient="index")
            bands_scores.to_csv(f"{ANALYSIS_RESULTS_DIR}/{soil_test_field}_10_20_bands.csv")
            fig, axs = plt.subplots(figsize=(15, 15))
            axs.set_xlabel("Band combinations")
            axs.set_ylabel("Score")
            axs.tick_params(axis="both", labelsize=20)
            bands_scores.plot.bar(ax=axs)
            plt.rcParams.update({"font.size": 10})
            plt.xticks(rotation=45, ha="right")
            fig.suptitle(f"band scores comparison: {soil_test_field}")
            plt.show()

            # Display the same info in a boxplot
            model_scores_df_transposed = bands_scores.transpose()
            fig, axs = plt.subplots(figsize=(15, 15))
            axs.set_xlabel("Classifier model")
            axs.set_ylabel("Score")
            fig.suptitle(f"band scores comparison  {soil_test_field}")
            plt.xticks(rotation=45, ha="right")
            plt.rcParams.update({"font.size": 10})
            plt.boxplot(model_scores_df_transposed, labels=[r for r in results.keys()], showmeans=True)
            plt.savefig(f"{ANALYSIS_RESULTS_DIR}/{soil_test_field}_10_20_bands.jpg")
            plt.show()

        def _score_band_combinations(soil_test_field):
            """
            Test all combinations of test_bands. Add the results to a dict so we can analyse

            """
            results = {}

            # Get the dataframe we want to work with
            fields_df = self.get_farm_bounds_as_pandas_df_for_analysis()

            # Filter out results with test values of 0 as these are records where no results were supplied
            fields_df = fields_df[fields_df[soil_test_field] > 0]

            # Iterate through all combinations of supplied bands
            for subset in _all_subsets(test_bands):
                log.info(subset)
                if subset:

                    # Test this subset of bands
                    x_train, x_test, y_train, y_test = train_test_split(
                        fields_df[[band for band in subset]],
                        fields_df[soil_test_field],
                        train_size=0.75,
                        shuffle=True,
                    )
                    classifier = LinearSVC(max_iter=20000)
                    classifier = classifier.fit(x_train, y_train)

                    y_pred = classifier.predict(x_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    kf = StratifiedKFold(n_splits=5, shuffle=True)
                    scores_stratified_KFold = cross_val_score(classifier, x_train, y_train, cv=kf)

                    # Can use cross_val_score to get averages
                    scores_cross_val = cross_val_score(classifier, x_train, y_train, cv=5)

                    scores_dict = {}
                    scores_dict["scores_stratified_KFold"] = scores_stratified_KFold.mean()
                    scores_dict["scores_cross_val"] = scores_cross_val.mean()
                    scores_dict["accuracy_score"] = accuracy

                    results[" ".join(subset)] = scores_dict

            _plot_results(results, soil_test_field)

        # Iterate over each of the test fields, score on different band combinations and plot results
        [
            _score_band_combinations(soil_test_field)
            for soil_test_field in (CARBON_RATING, NITROGEN_RATING, POTASSIUM_RATING, PHOSPHORUS_RATING)
        ]

    def _plot_best_performing_linear_svc_classifier_with_different_band_combinations_inputs(self):
        """
        Plot the 10 best performing band combinations for each soil test category
        :return:
        """

        def _plot_10_best_performing_combinations(soil_test_field):
            df = pd.read_csv(f"{ANALYSIS_RESULTS_DIR}/{soil_test_field}_10_20_bands.csv", index_col=0)

            df["mean"] = df.mean(axis=1)
            df = df.sort_values("mean", ascending=False).head(10)
            # NOTE - you may want to save this dataframe to csv for further analysis

            fig, axs = plt.subplots(figsize=(15, 15))
            axs.set_xlabel("Band combinations")
            axs.set_ylabel("Score")
            axs.tick_params(axis="both", labelsize=10)
            df.plot.bar(ax=axs)
            plt.rcParams.update({"font.size": 10})
            plt.xticks(rotation=45, ha="right")
            fig.suptitle(f"10 best performing band scores comparison for {soil_test_field}")
            plt.savefig(f"{ANALYSIS_RESULTS_DIR}/{soil_test_field}_10_20_top_performing_bands_bar.jpg")
            plt.show()

            model_scores_df_transposed = df.transpose()
            fig, axs = plt.subplots(figsize=(15, 15))
            axs.set_xlabel("Band combinations")
            axs.set_ylabel("Score")
            fig.suptitle(f"10 best performing band scores comparison for {soil_test_field}")
            plt.xticks(rotation=45, ha="right")
            plt.rcParams.update({"font.size": 10})
            plt.boxplot(
                model_scores_df_transposed, labels=[r for r in model_scores_df_transposed.columns], showmeans=True
            )
            plt.savefig(f"{ANALYSIS_RESULTS_DIR}/{soil_test_field}_10_20_top_performing_bands_box.jpg")
            plt.show()

        [
            _plot_10_best_performing_combinations(soil_test_field)
            for soil_test_field in (CARBON_RATING, NITROGEN_RATING, POTASSIUM_RATING, PHOSPHORUS_RATING)
        ]

    def test_different_classifiers(self):
        """
        Iterate through and test a number of classifiers. We get 3 different scores which are added to results
        which is then converted to a dataframe and plotted
        """

        results = {}

        # Get the dataframe we want to work with
        fields_df = self.get_farm_bounds_as_pandas_df_for_analysis()

        # test_bands = [MEAN_NDWI, MEAN_NDVI]

        # test_bands = list(set(ALL_BANDS) - set([BAND_SCL_20M, BAND_SCL_60M, BAND_TCI_10M, BAND_TCI_20M]))

        # Just use a subset of all bands for now
        test_bands = [
            BAND_8_10M,
            BAND_2_10M,
            BAND_4_10M,
            BAND_3_10M,
        ]
        # Specify band - look at carbon rating initially. Remove ratings of 0 as these are records with no results
        fields_df = fields_df[fields_df[CARBON_RATING] > 0]

        # Split into test and train. Looking at carbon for the moment
        x_train, x_test, y_train, y_test = train_test_split(
            fields_df[[band for band in test_bands]], fields_df[CARBON_RATING], train_size=0.75, shuffle=True
        )

        C = 10
        kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

        # Linear SVC -> KNeighbours -> SVC
        # (see https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
        classifiers = {
            "LinearSVC": LinearSVC(),
            "LinearSVC modified": LinearSVC(C=2, class_weight="balanced"),
            "SVC (Linear kernel)": SVC(kernel="linear", C=1, probability=True, random_state=0),
            "SVC (Linear kernel 2)": SVC(kernel="linear", C=10, gamma=10, probability=True, random_state=0),
            "KNeighbors": KNeighborsClassifier(),
            "SVC (Linear kernel)": SVC(kernel="linear", C=C, probability=True, random_state=0),
            "SVC (rbf kernel)": SVC(kernel="rbf", C=C, probability=True, random_state=0),
            "L1 logistic": LogisticRegression(
                C=C, penalty="l1", solver="saga", multi_class="multinomial", max_iter=10000
            ),
            "L2 logistic (Multinomial)": LogisticRegression(
                C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=10000
            ),
            "L2 logistic (OvR)": LogisticRegression(
                C=C, penalty="l2", solver="saga", multi_class="ovr", max_iter=10000
            ),
            "logistic": LogisticRegression(solver="liblinear", random_state=0),
            # "GPC": GaussianProcessClassifier(kernel),
            " DecisionTreeClassifier": DecisionTreeClassifier(),
            " DecisionTreeClassifier (entropy)": DecisionTreeClassifier(criterion="entropy", max_depth=3),
            " RandomForest Classifier": RandomForestClassifier(n_estimators=100),
            # "SDG": SGDClassifier(
            #     max_iter=1000, tol=0.01
            # ),  # Stocastic models can offer different results each time they are run. Their behaviour incorporates elements for randomness.
            # see https://machinelearningmastery.com/different-results-each-time-in-machine-learning/
        }

        for index, (name, classifier) in enumerate(classifiers.items()):

            # We score each classifier in 3 different ways out of interest
            scores_dict = {}

            kf = StratifiedKFold(n_splits=5, shuffle=True)
            scores_stratified_KFold = cross_val_score(classifier, x_train, y_train, cv=kf)
            # Can use cross_val_score to get averages
            scores_cross_val = cross_val_score(classifier, x_train, y_train, cv=5)
            log.info(
                name
                + " %0.2f accuracy with a standard deviation of %0.2f"
                % (scores_cross_val.mean(), scores_cross_val.std())
            )
            classifier = classifier.fit(x_train, y_train)

            log.info(f"{name} coefficient of determination (train): {classifier.score(x_train, y_train)}")
            log.info(f"{name} coefficient of determination (test): {classifier.score(x_test, y_test)}")
            y_pred = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            # Maybe we should use the average from scores here
            results[name] = accuracy
            log.info("Accuracy for %s: %0.1f%% " % (name, accuracy * 100))

            cm = confusion_matrix(y_test, y_pred)
            log.info(cm)
            c = plot_confusion_matrix(classifier, x_test, y_test, cmap="GnBu")
            c.ax_.set_title(f"{name}, Accuracy {accuracy}")
            plt.show()

            scores_dict["scores_stratified_KFold"] = scores_stratified_KFold.mean()
            scores_dict["scores_cross_val"] = scores_cross_val.mean()
            scores_dict["accuracy_score"] = accuracy
            results[name] = scores_dict

        # Plot the scores for each in bar charts
        model_scores_df = pd.DataFrame.from_dict(results, orient="index")
        fig, axs = plt.subplots(figsize=(15, 15))
        axs.set_xlabel("Classifier model")
        axs.set_ylabel("Score")
        axs.tick_params(axis="both", labelsize=20)
        model_scores_df.plot.bar(ax=axs)
        plt.rcParams.update({"font.size": 20})
        plt.xticks(rotation=45, ha="right")
        fig.suptitle("Model scores comparison")
        plt.show()

        # Display the same info in a boxplot
        model_scores_df_transposed = model_scores_df.transpose()
        fig, axs = plt.subplots(figsize=(15, 15))
        axs.set_xlabel("Classifier model")
        axs.set_ylabel("Score")
        fig.suptitle("Model scores comparison")
        plt.xticks(rotation=45, ha="right")
        plt.boxplot(model_scores_df_transposed, labels=[r for r in classifiers.keys()], showmeans=True)
        plt.savefig(f"{ANALYSIS_RESULTS_DIR}/classifier_comparison_carbon.jpg")
        plt.show()

    def perform_analysis(self):
        """
        Work in progress - perform various analysis to see if we can find any relationships between band means and
        soil test results
        """

        # Plot a summary image containing RGB images of all of our fields
        # self._plot_rgb_for_fields_chosen_for_analysis()

        # Plot correlation matrices for each test field
        # Have to treat each individually as we have results for some and not others
        # soil_test_columns = [NITROGEN_RATING, PHOSPHORUS_RATING, POTASSIUM_RATING, CARBON_RATING]
        # [self._correlation_plots(soil_test_columns, field) for field in soil_test_columns]

        # Test list of classifiers with 10m bands to see if there is any relationship with carbon soil results.
        # TODO - check other soil test results and also try other band combinations
        # self.test_different_classifiers()

        # Focus on LinearSVC. Pass all combinations of the 10 and 20m bands to this model, store and plot accuracy scores
        # for each of the soil test categories
        # self._test_linear_svc_classifier_with_different_band_combinations_inputs()

        # Plot the best performing band combinations (based on output from _test_linear_svc_classifier_with_different_band_combinations_inputs())
        self._plot_best_performing_linear_svc_classifier_with_different_band_combinations_inputs()

    def get_farm_bounds_as_pandas_df_for_analysis(self) -> DataFrame:
        """
        Convert to normal dataframe rather than Geopandas as strange issues plotting sometimes
        Remove some bands that are not required for analysis
        :return:
        """
        # Convert to normal dataframe rather than Geopandas as strange issues plotting sometimes
        fields_df = pd.DataFrame(self.farm_bounds_32643)
        # Get rid of bands we don't need for analysis
        fields_df.drop(
            [BAND_SCL_20M, BAND_SCL_60M, "farm_id(from_platform)", BAND_TCI_10M, BAND_TCI_20M], axis=1, inplace=True
        )
        # Drop any rows that don't have band values
        fields_df = fields_df[fields_df[BAND_8_10M].notna()]
        fields_df.to_csv(f"{FARM_SUMMARIES_DIRECTORY}/farms.csv")
        return fields_df

    def generate_mean_ndwi(self, product):
        """
        Calculate the mean NDWI(Normalised Difference Water Index) for a farm for the specified product and add result to product
        :param product:
        :return:
        """

        # FIXME This is incorrect
        # See https://en.wikipedia.org/wiki/Normalized_difference_water_index
        # We should be using band 8A (864nm) and band 11 (1610nm)
        # or band 8A (864nm) and band 12 (2200nm)

        # Don't think we can do this at present until we resample so the bands are at the same resolution

        green_path = product[BAND_3_10M]
        nir_path = product[BAND_8_10M]

        if green_path and nir_path:

            with rasterio.open(green_path) as red:
                green_band = red.read(1)
                out_meta = red.meta.copy()
            with rasterio.open(nir_path) as nir:
                nir_band = nir.read(1)

            # Allow division by zero
            np.seterr(divide="ignore", invalid="ignore")
            # Calculate NDVI
            # ndvi = (b4.astype(float) - b3.astype(float)) / (b4 + b3)
            # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
            # Index = (NIR - MIR)/ (NIR + MIR) using Sentinel-2 Band 8 (NIR) and Band 12 (MIR).
            ndwi = (green_band.astype(float) - nir_band.astype(float)) / (
                green_band.astype(float) + nir_band.astype(float)
            )
            # Get the mean (discounting nan values)
            product["mean_ndwi"] = np.nanmean(ndwi)
            return product

            # plt.imshow(ndvi)
            # plt.show()
            # vmin, vmax = np.nanpercentile(ndvi, (1,99))
            # # img_plt = plt.imshow(ndvi, cmap='gray', vmin=vmin, vmax=vmax)
            # plt.imshow(ndvi, cmap='Greens', vmin=vmin, vmax=vmax)
            # show(ndvi, cmap="Greens")

            # out_raster_path = f"{Path(green_path).parent}/ndwi.tif"
            #
            # out_meta.update(dtype=rasterio.float32, count=1)
            # with rasterio.open(
            #     out_raster_path,
            #     "w",
            #     **out_meta,
            # ) as out:
            #     out.write(ndwi, 1)

            # show(ndvi, cmap="Greens")

            # Verify
            # with rasterio.open(ndvi_raster_path, "r") as src:
            #     fig, ax = plt.subplots(figsize=(15, 15))
            #     show(src, ax=ax, cmap="Greens")
            #     plt.show()

        return product

    def load_farm_cloud_free_products_df(self, farm_details: Series):
        """
        Load the specified farm products which should be in summaries directory saved as geojson
        :param farm_details:
        :return: GeoDataFrame
        """

        farm_products_path = self.get_farm_cloud_free_products_df_path(farm_details)
        if not os.path.exists(farm_products_path):
            sys.exit(
                f"Exiting as unable to find {farm_products_path}. Please run script with --crop-individual-farms "
                f"to generate geojson list of cloud free products for each farm"
            )
        return gpd.read_file(farm_products_path)

    def get_farm_cloud_free_products_df_path(self, farm_details: Series):
        """
        Get path to store cloud free products for specified farm
        :param farm_details:
        :return:
        """

        field_id = farm_details["field_id"]
        return f"{FARM_SUMMARIES_DIRECTORY}/{field_id}/{field_id}_cloud_free_products.geojson"

    def plot_mean_ndvi(self):
        def _plot_ndvi(farm_details):
            cloud_free_products_df = self.load_farm_cloud_free_products_df(farm_details)

            df = pd.DataFrame(cloud_free_products_df)
            df.plot(x="generationdate", y="mean_ndvi")
            plt.show()

            cloud_free_products_df["generationdate"] = pd.to_datetime(cloud_free_products_df["generationdate"])
            cloud_free_products_df.set_index("generationdate", inplace=True)
            cloud_free_products_df["mean_ndvi"].plot()

            plt.show()

            fig, axs = plt.subplots(figsize=(12, 4))
            # cloud_free_products_df["mean_ndvi"].plot.area(ax=axs)
            cloud_free_products_df["mean_ndvi"].plot(ax=axs, x="A", y="B")
            axs.set_ylabel("Reflectance")
            fig.suptitle(f"{farm_details['field_id']}:{farm_details['farm_name']}")
            plt.show()

        self.farm_bounds_32643.apply(_plot_ndvi, axis=1)

    def apply_raster_analysis_function_single_farm(self, farm_details: Series, analysis_func):
        """
        Generic function to apply the specified analysis function for each farm in suitable products
        :param farm_df_index:
        :param analysis_func:
        :return:
        """
        # farm_details = self.get_farm_from_dataframe(farm_df_index)
        cloud_free_products_df = self.load_farm_cloud_free_products_df(farm_details)

        field_id = farm_details["field_id"]
        updated = cloud_free_products_df.apply(analysis_func, field_id=field_id, axis=1)

        # Save results
        updated.to_file(self.get_farm_cloud_free_products_df_path(farm_details), driver="GeoJSON")

        # Add to master products list
        df = updated[["uuid", f"{field_id}_mean_ndvi"]]
        self.products_df = pd.merge(self.products_df, df, on="uuid", how="left")
        self.save_products_df_to_geojson()

    def apply_raster_analysis_function_all_farms(self, analysis_func):
        """
        Apply the specified analysis function to all farms
        :type analysis_func: the analysis to be performed for each farm on cloud free products

        """
        self.farm_bounds_32643.apply(
            self.apply_raster_analysis_function_single_farm, analysis_func=analysis_func, axis=1
        )

    def generate_all_farms_ndvi_rasters(self):
        """
        Generate ndvi rasters for all farms in appropriate products
        :return:
        """

        [
            self.apply_raster_analysis_function_single_farm(farm_index, self.generate_mean_ndvi)
            for farm_index in range(len(self.farm_bounds_32643))
        ]

    def generate_all_farms_band_histograms(self):

        [
            self.apply_raster_analysis_function_single_farm(farm_index, self.generate_band_histogram)
            for farm_index in range(len(self.farm_bounds_32643))
        ]

    def generate_all_farms_ndwi_rasters(self):
        """
        Generate ndwi rasters for all farms in appropriate products
        :return:
        """

        [
            self.apply_raster_analysis_function_single_farm(farm_index, self.generate_mean_ndwi)
            for farm_index in range(len(self.farm_bounds_32643))
        ]

    def generate_cloud_free_farm_product_lists(self, force_recreate=False):
        """
        Generate a list of valid products for each farm. We can then add metrics such as mean ndvi etc. Save as geojson
        """

        def _generate_product_list(farm_details):
            field_id = farm_details["field_id"]
            farm_name_dir = f"{FARM_SUMMARIES_DIRECTORY}/{field_id}"
            os.makedirs(farm_name_dir, exist_ok=True)

            farm_products_path = self.get_farm_cloud_free_products_df_path(farm_details)

            if not os.path.exists(farm_products_path) or force_recreate:

                if "cloud_free_products" not in farm_details or force_recreate:
                    # This adds list of cloud free products to farms df
                    farm_details = self.set_cloud_free_products(farm_details)

                cloud_free_products_df = self.get_cloud_free_products_for_farm(BAND_4_10M, farm_details)

                def fix_band_paths(product):
                    def _update_df_bands(band):
                        product[band] = self.get_farm_raster_from_product_raster_path(field_id, product[band])

                    [_update_df_bands(band) for band in ALL_BANDS]
                    return product

                cloud_free_products_df = cloud_free_products_df.apply(fix_band_paths, axis=1)
                cloud_free_products_df.to_file(farm_products_path, driver="GeoJSON")

            return farm_details

        # Save geojson of valid products for each field
        self.farm_bounds_32643 = self.farm_bounds_32643.apply(_generate_product_list, axis=1)
        self.save_farms_with_valid_products()


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option(
    "--download",
    "-d",
    is_flag=True,
    help="Download Sentinel 2 data. Note the downloader can be unreliable and you may have to kill and restart the script repeatedly",
)
@click.option(
    "--crop-individual-farms",
    "-ci",
    is_flag=True,
    help="Crop Sentinel 2 rasters, filter clouds and calculate band means",
)
@click.option("--farm_summaries", "-fs", is_flag=True, help="Generate summary jpegs for specified bands over time")
@click.option("--farm_analysis", "-fa", is_flag=True, help="Perform analysis on farms dataframe.  In progress")
@click.option(
    "--sentinel_date_range",
    required=True,
    type=(str, str),
    default=("20210401", "20220430"),
    help='Specify the date window to get sentinel data. Default is ("20210401", "20220401").'
    " Has to be combined with -d flag to start download",
)
def main(download, crop_individual_farms, sentinel_date_range, farm_summaries, farm_analysis):
    """
    Download and process Sentinel 2 rasters.

    If you wish to download (-d), please ensure you set "SENTINEL_USER" and "SENTINEL_PASSWORD"
    environment variables. An account can be created at https://scihub.copernicus.eu/dhus/#/self-registration

    :param farm_summaries:
    :param crop_individual_farms:
    :param download:
    :param sentinel_date_range:
    """
    data_handler = DataHandler(sentinel_date_range)

    if download:
        data_handler.download_sentinel_products()

    if crop_individual_farms:
        data_handler.crop_rasters_to_individual_fields_bbox()
        data_handler.generate_cloud_free_farm_product_lists(force_recreate=True)
        data_handler.generate_band_means_at_soil_test_date()

    if farm_summaries:
        data_handler.generate_individual_farm_bands_summary(filter_clouds=True)

    if farm_analysis:
        data_handler.perform_analysis()

    log.info("Done")


if __name__ == "__main__":
    main()
