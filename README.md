# GoodForest - DTY Fall2024 - P1

## Satellite Imagery Analysis for Forest Management

### Description
This project aims to provide a model based on AI for forest management using Sentinel-2 satellite imagery. Detecting early signs of trees dieback, parasites, and other diseases is crucial for the preservation of forests. The model will be able to detect these signs and provide a report to the forest managers.

### Installation
To prevent any conflict with the dependencies, we recommend using a virtual environment with Python 3.9.

To install the project, you need to clone the repository and install the required packages.

After having cloned the repository, create a virtual environment:

```bash
git clone
cd goodofrest-fall2024p1
python -m venv .venv
source .venv/bin/activate
```

Then, you can either install `goodforest_lib` as a library defined in [`setup.py`](setup.py) or install the dependencies directly to use the command line interface (CLI). If you want to install the library while editing the code, you can use the `-e` option.

```bash
pip install . # Install the library
# or
pip install -e . # Install the library in editable mode
# or
pip install -r requirements.txt # Install the dependencies
```

### Usage

#### Library

The `goodforest_lib` library provides several modules to load, preprocess, and visualize the data. You can use these modules in your own scripts or notebooks.

To see some use cases of the library, you can check the [`notebooks`](./notebooks) folder. In these notebooks, we provide examples of how to use the library to load the data, preprocess it, and visualize it.

#### Command Line Interface (CLI)

All the available modules are accessible with the command line interface (CLI). To access the CLI, run the following command:

```bash
python -m goodforest-cli
```

The CLI provides the following commands:

- `load-data`: This command allows you to load the data from the Sentinel-2 satellite imagery.
    - `S2-train`: This subcommand allows you to load the Sentinel-2 data from Google Earth Engine API to train a model
    - `S2-predict`: This subcommand allows you to load the Sentinel-2 data from Google Earth Engine API to predict with a model
    - `S2-fordead`: This subcommand allows you to load the Sentinel-2 data from Google Earth Engine API to use them in fordead
- `preprocess-data`: This command allows you to preprocess the data.
    - `labelize`: This subcommand allows you to labelize the data thanks to the annotated dataset.
    - `cubify`: This subcommand allows you to cubify the data.
    - `filter`: This subcommand allows you to filter the cubes with few vegetation.
- `visualize`: This command allows you to visualize the data.
    - `tif`: This subcommand allows you to visualize the tif files of Sentinel2 images.
    - `h5-cube`: This subcommand allows you to visualize the cubes from a pkl files.
- `utils`: This command allows you to use the utility functions.
    - `download_from_gcs`: This subcommand allows you to download the data from Google Cloud Storage.
    - `upload_to_gcs`: This subcommand allows you to upload the data to Google Cloud Storage.


To get more information about the commands and subcommands, you can use the `--help` or `-h` option:

```bash
python -m goodforest-cli <command> <subcommand> --help
```

### Methodology

The project is divided in several modules, each one with a specific purpose. For each module, we provide a README file with a detailed description of the subcommands and the methodology used.

Moreover, all constants used in the project are stored in the [`goodforest-cli/config/constants.py`](./goodforest-cli/config/constants.py) file. You can modify these constants to adapt the project to your needs or simply change the default values as arguments in the CLI.


### Support
For any questions or issues, please contact us at the email addresses below.

### Authors and acknowledgment
Main authors and contributors of the project.

| Name | Role | Email Address |
|------|--------|------|
| Alexandre Faure | Data Scientist | [alexandre.faure@student-cs.fr](mailto:alexandre.faure@student-cs.fr) |
| Lucien Auriol-Delpla | Data Scientist | [lucien.delpla@student-cs.fr](mailto:lucien.delpla@student-cs.fr) |
| Thibaut Mazuir | Data Scientist | [thibaut.mazuir@student-cs.fr](mailto:thibaut.mazuir@student-cs.fr) |
| Mayara Ayat | Data Scientist | [mayara.ayat@student-cs.fr](mailto:mayara.ayat@student-cs.fr) |
| Théo Rubenach | Technical Coach | [theo.rubenach@illuin.tech](mailto:theo.rubenach@illuin.tech) |

### License
To be defined

### Project status

This is the first version of this project on the 31th of October 2024.