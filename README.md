# Sand and Gravel Mapping

This repository provides the code and data that accompanied our paper titled Mapping Construction Grade Sand: Stepping Stones Towards
Sustainable Development chosen as a poster for COMPASS 2023 and a paper at KDD / Fragile Earth Workshop 2023.

## Data

There are 2 levels of data aggregation that we performed:

1. Aggregated geocoordinates and grain size information from various academic sources
This dataset can be found in a [Google Sheet here](https://docs.google.com/spreadsheets/d/13nF_pJ02Bd70cDJamuKbvZIkIdJ-kOI4O3Cx9K7Wzos/edit?usp=sharing). Each sheet contains data from a separate source.
  
2. The positions from the dataset are fine-tuned, along with a timestamp (if the SGR desposits are not available on the original date), and a set of spectral bands are  grabbed, object based image analysis (OBIA) is conducted and the median value of those bands are stored. These are available in `data/labels`

  1. The base dataset is: `gt-bands.xlsx`, which contains all S-1 and S-2 median band values for the cluster identified using SNIC for each sample
  2. Other modfied dataset that could be of use are `gt-bands-resampled-s10-dw.xlsx` that also has the [Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) (DW) class probability layers as bands. Other variations include `gt-bands-resampled-s15-dw.xlsx` or `gt-bands-resampled-s15.xlsx` which are resampled versions of the base dataset with different superpixel parameters for the SNIC algorithm, with or without the DW bands.


## Entry points
