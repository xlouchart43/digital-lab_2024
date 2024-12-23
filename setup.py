from setuptools import find_packages, setup

setup(
    name="goodforest_lib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "altair==5.4.1",
        "blinker==1.8.2",
        "contextily==1.6.2",
        "geedim==1.8.1",
        "geemap==0.34.4",
        "geopandas==1.0.1",
        "gitpython==3.1.43",
        "google==3.0.0",
        "h5py==3.12.1",
        "opencv-python==4.10.0.84",
        "pyarrow==17.0.0",
        "pydeck==0.9.1",
        "rich==13.9.2",
        "tensorboard==2.17.1",
        "toml==0.10.2",
        "torch==2.4.1",
    ],
)
