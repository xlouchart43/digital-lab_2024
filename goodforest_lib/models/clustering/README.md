# Clustering and Anomaly Detection
This module contains the implementation of the clustering and anomaly detection algorithms.

## Clustering
The clustering module contains the implementation of the following clustering methods:
- Clustering with K-Means on overlapping grids
- Clustering with Louvain Modularity Optimization on graphs

### K-Means on Overlapping Grids
The `get_clusters` function in the `zone_segmentation/create_cluster.py` retrieves clusters from historical data. It takes in the following parameters:
- `images`: An array of images to get the clusters from.
- `dates`: An array of dates corresponding to the historical data.
- `bands`: A list of bands to fetch.
- `before_date`: The date before which to fetch the data.
- `cluster_period_duration_in_days`: The duration of the period to cluster (default is 3 months).

The function first calculates the `after_date` by subtracting the `cluster_period_duration_in_days` from the `before_date`. It then calls the `get_historical_data_with_dates` function to retrieve the historical data and dates within the specified period.

Next, the function prepares the data by converting any NaN values to zeros and obtaining the zones with overlapping SWIR (Short-Wave Infrared) data using the `get_overlapped_swir_zones` function.

The `clustering_zones_step` function is then called to perform clustering on the zones with SWIR data. This function takes in a list of tuples, where each tuple contains a zone (as a numpy array) and its starting coordinates. It returns a dictionary of clusters for each zone, a dictionary of centroids for each zone, and a dictionary of labels for each zone.

Finally, the `get_outlier_input` function is called to obtain outlier detection input from the clusters and pixel dictionary of the historical data. This function divides the data into grids and retrieves the corresponding clusters for each grid.

The `get_clusters` function returns the partitions (outlier detection input) and the dates of the clustering.

Overall, this code implements the K-Means clustering algorithm on overlapping grids to retrieve clusters from historical data for further analysis and anomaly detection.

The detailed implementation of the K-Means clustering algorithm on overlapping grids can be found in the `zone_segmentation/clustering_process.py` and `prepare_data` files.

### Louvain Modularity Optimization on Graphs
The `build_pixel_graph` function in the `graph_segmentation/create_graph.py` file builds a graph from the historical data. It takes in the following parameter:
- `historical_data`: The historical data to build the graph from.

The function first calculates the height and width of the historical data. It then initializes an empty graph using the `nx.Graph()` function.

Next, the function computes the weights for the vertical, horizontal, and diagonal edges of the graph using the `get_weights_from_arrays` function. This function calculates the absolute difference between two arrays, normalizes the difference, and applies a correction function to the normalized difference.

The function then iterates over each pixel in the historical data and adds the corresponding edges to the graph. Vertical edges are added between each pixel and its below neighbor, horizontal edges are added between each pixel and its right neighbor, and diagonal edges are added between each pixel and its bottom-right and top-right neighbors.

The resulting graph represents the relationships between pixels in the historical data based on their weights. This graph can be used for further analysis and clustering using the Louvain Modularity Optimization algorithm.

For a detailed implementation of the Louvain Modularity Optimization algorithm on graphs, please refer to the `graph_segmentation/louvain_clustering.py` file.



## Anomaly Detection
The `anomaly_detection.py` module is responsible for detecting anomalies in the historical data. It contains the implementation of the `predict_dieback` function, which predicts dieback based on the historical data and clustering of the zones. The function takes in parameters such as images, dates, bands, before_date, clusters, and detection_period_duration_in_days. It calculates the after_date, retrieves the historical data within the specified period, performs anomaly detection using the `anomaly_detection` function, and returns the predicted anomalies, degrees of the anomalies, and dates of detection.

The `anomaly_detection` function detects anomalies in given zones based on SWIR values or other bands and clustering. It takes in parameters such as historical_data, global_clusters, threshold_factor, threshold_count, and zone_size. The function detects anomalies using max-std-based adaptive thresholds and returns the detected anomalies and degrees of the anomalies.

The `detect_zone_anomalies_with_max_std` function detects anomalies in a zone by monitoring deviations from updated cluster centroids using max-std-based adaptive thresholds. It takes in parameters such as zone_size, detection_swir, global_clusters, threshold_factor, and threshold_count. The function returns a list of detected anomalies and their per-date deviation distances.

The `outlier_detect` function detects outliers in the clusters using the max std threshold method. It takes in parameters such as clusters, historical_data, and threshold_factor. The function returns a list of detected outliers and their per-date deviation distances.

The `compute_max_std_threshold` function computes a single adaptive threshold based on the maximum standard deviation of the SWIR values across all dates. It takes in parameters such as cluster_time_series and threshold_factor. The function returns the adaptive threshold.

The `detect_deviation_with_max_std` function detects if a pixel deviates from the updated centroid of its cluster using a max-std-based adaptive threshold. It takes in parameters such as pixel_data, cluster_time_series, updated_centroid, and threshold_factor. The function returns a boolean indicating if a deviation was detected, the deviation per date, and the adaptive threshold.


## Usage
To use the clustering and anomaly detection modules, you can follow the steps in the demo notebook at the root of this repository `notebooks/anomaly-detection_clustering.ipynb`. This notebook demonstrates how to retrieve clusters from historical data using K-Means on overlapping grids or Louvain Modularity Optimization on graphs, and how to detect anomalies in the historical data using the clustering information.

You can also refer to the individual module files for detailed implementations of the clustering and anomaly detection algorithms.




