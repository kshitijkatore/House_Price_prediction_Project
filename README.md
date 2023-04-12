
## Boston-House Price Prediction ->LinearRegression

Thise project motive to Explore some LinearRegression model using the House price prediction datasets.


## ⌛Installation

Install some Libraries

```bash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```
    
## API Reference

#### Get all items

### [Numpy Reference](https://numpy.org/doc/stable/reference/c-api/array.html)
### [Pandas](https://pandas.pydata.org/docs/reference/index.html)
### [matplotlib](https://matplotlib.org/stable/index.html)
### [Seaborn](https://seaborn.pydata.org/api.html)
### [sklearn](https://scikit-learn.org/stable/modules/classes.html)


#### Steps for model creation
* 1.Gather and clean the data: Collect the data that you will use for training and testing your model. This data may be in various formats such as CSV, Excel, or a database. You will need to clean the data by removing missing values, removing outliers, and transforming the data as necessary.

* 2.Split the data into training and testing sets: Split your data into two sets: one for training the model and another for testing the model. The typical split is 80% for training and 20% for testing, but this can vary depending on the size of the dataset.

* 3.Feature selection: Select the features or independent variables that will be used to train the model. It's important to choose features that are relevant to the problem you are trying to solve.

* 4.Scaling and normalization: Scale and normalize the data to ensure that each feature has a similar range and distribution. This step is important because linear regression is sensitive to the scale of the features.

* 5.Train the model: Use the training dataset to train the linear regression model. During training, the model will learn the coefficients for each feature that will be used to make predictions.

* 6.Evaluate the model: Use the testing dataset to evaluate the performance of the model. You can use metrics such as mean squared error, R-squared, and others to evaluate the performance of the model.

* 7.Tune the model: If the model's performance is not satisfactory, you can tune the hyperparameters such as the learning rate or regularization parameter to improve the model's performance.

* 8.Deploy the model: Once the model has been trained and tested, you can deploy it to make predictions on new data.
## 📝Description

* Thise projects based on LinearRegression algorithm.
* In thise projects i have using different-different types of LinearRegression model.
  * 1.LinearRegression()
Using sklearn machine learning libraries.

## 📊Datasets
* ### [Download Datasets](https://drive.google.com/drive/folders/1v-vQum2yW81vRJG0JVNhzEh0H5SJnv6J)
* Download the datasets for costom training

## 🎯Inference Demo
### Using LinearRegression
```
from sklearn.linear_model import LinearRegression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
```
## 🕮 Please go through [House_Price_prediction.docx](http//.grfefe) more info.