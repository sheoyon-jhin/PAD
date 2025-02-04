<h1 align='center'> Precursor-of-Anomaly Detection for Irregular Time Series <br>(KDD 2023)<br>
    [<a href="https://arxiv.org/abs/2306.15489">paper</a>] </h1>
Anomaly detection is an important field that aims to identify unexpected patterns or data points, and it is closely related to many real-world problems, particularly to applications in finance, manufacturing, cyber security, and so on. While anomaly detection has been studied extensively in various fields, detecting future anomalies before they occur remains an unexplored territory. In this paper, we present a novel type of anomaly detection, called Precursor-of-Anomaly (PoA) detection. Unlike conventional anomaly detection, which focuses on determining whether a given time series observation is an anomaly or not, PoA detection aims to detect future anomalies before they happen. To solve both problems at the same time, we present a neural controlled differential equation-based neural network and its multi-task learning algorithm. We conduct experiments using 17 baselines and 3 datasets, including regular and irregular time series, and demonstrate that our presented method outperforms the baselines in almost all cases. Our ablation studies also indicate that the multitasking training method significantly enhances the overall performance for both anomaly and PoA detection.
<p align="center">
  <img align="middle" src="./PAD.png" alt="PAD"/> 
</p>
Comparison between the conventional anomaly detection and our proposed the precursor-of-anomaly (PoA) detection. In PoA, we predict whether the next window will contain any abnormal observation before it happens, which is much more challenging than the anomaly detection.

### create conda environments
```
conda env create --file PAD.yml
```

### activate conda 
```
conda activate PAD
```


### activate conda 
```
sh swat.sh
```
```
@inproceedings{10.1145/3580305.3599469,
author = {Jhin, Sheo Yon and Lee, Jaehoon and Park, Noseong},
title = {Precursor-of-Anomaly Detection for Irregular Time Series},
year = {2023},
isbn = {9798400701030},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3580305.3599469},
doi = {10.1145/3580305.3599469},
booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {917–929},
numpages = {13},
keywords = {neural controlled differential equations, precursor-of-anomaly detection, time-series anomaly detection, anomaly detection},
location = {Long Beach, CA, USA},
series = {KDD '23}
}
```
