Item-based collaborative filtering recommender system
=====================================================

Requirements
------------
- pyspark
- numpy

Project structure
------------------
- `predictor.py`: script with prediction function
- `metrics.py`: performance metrics implementation
- `data_transform.py`: script for string indexing and creating new csv file
- `validation.py`: cross-validation script
- `utils.py`: some utils for the algorithm

Cross-validation performance metrics results
---------------------------

| Sub <br> dataset |    MSE    |     RMSE     |     nDCG     |     Gini     |
|-----------------:|:---------:|:------------:|:------------:|:------------:|
| 1                | 1.579     | 3.471        | 0.6632       | 0.241        |
| 2                | 1.345     | 3.036        | 0.202        | 0.225        |
| 3                | 2.012     | 5.030        | 0.490        | 0.107        |
| 4                | 1.199     | 1.021        | 0.810        | 0.273        |
| 5                | 0.922     | 1.048        | 0.613        | 0.328        |
