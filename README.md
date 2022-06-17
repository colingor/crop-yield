# Crop Yield

This script provides a means to download Sentinel 2 data for the specified farm regions, crop the images for each field and check
which are cloud covered. It loads the soil test data for each field and gets the nearest cloud free images to the date the soil
tests were conducted. The mean of each band of the selected Sentinel 2 product are calculated and added to the final farms.csv
dataframe. This dataframe forms the basis of further analysis.

Once the above steps have been completed, the analysis functionality can be run to examine the data. Note that farms.csv has already
been generated, using the farms from MASTER_DOCUMENT_V1_240522.xlsx. If you wish to analyse new farms, you will have to run the download step above. If you are
happy to analyse the existing list, you can proceed without downloading any Sentinel 2 products and use the previously genertated farms.csv

Additionally, it can also generate summary RGB images for each field over time. This depends on the user having already downloaded Sentinel 2 products (using the -d flag)
Note that the code ships with farm summaries for the existing list of farms

## Getting Started

* Create a virtual environment:
    ```bash
    python3 -m venv ~/<path_on_your_machine>/<chosen_venv_name>/
    ```
* Activate the virtual environment:
  ```bash
  source ~/<path_on_your_machine>/<chosen_venv_name>/bin/activate
  ```
* Install python requirements
    ```bash
    pip install -r requirements.txt
  ```

## Running the script

To see the available options, run with the help flag:
```bash
python crop_yield.py --help
```
Resulting in:

```
Usage: crop_yield.py [OPTIONS]

  Download and process Sentinel 2 rasters.

  If you wish to download (-d), please ensure you set "SENTINEL_USER" and
  "SENTINEL_PASSWORD" environment variables. An account can be created at
  https://scihub.copernicus.eu/dhus/#/self-registration

  :param farm_summaries: :param crop_individual_farms: :param download: :param
  sentinel_date_range:

Options:
  -d, --download                  Download Sentinel 2 data. Note the
                                  downloader can be unreliable and you may
                                  have to kill and restart the script
                                  repeatedly
  -ci, --crop-individual-farms    Crop Sentinel 2 rasters, filter clouds and
                                  calculate band means
  -fs, --farm_summaries           Generate summary jpegs for specified bands
                                  over time
  -fa, --farm_analysis            Perform analysis on farms dataframe.  In
                                  progress
  --sentinel_date_range <TEXT TEXT>...
                                  Specify the date window to get sentinel
                                  data. Default is ("20210401", "20220401").
                                  Has to be combined with -d flag to start
                                  download  [required]
  --help                          Show this message and exit.
```

**NOTE: If you wish to download Sentinel 2 data, you have to create an account and specify your
credentials as specified above**

If you only wish to run the analysis on the existing farm list, you don't have to download the Sentinel 2 data as the code 
ships with the generated list of farms, band means and soil test results in farms.csv.  You would only have 
to download the Sentinel data if you added new fields and wished to add them to the analysis


## Analysis

The -fa flag calls the perform_analysis function. ** You should look at the code in this function before running it.** It contains a 
number of experimental data analysis steps to see if we could find any links between band means and test results. It is likely
you would want to comment/uncomment various analysis as you see fit.


