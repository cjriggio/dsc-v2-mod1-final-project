# Kings County Housing Data Analysis

### Outline of Jupyter Notebook
###### 1. Importing and Cleaning the Data
###### 2. Dealing with Categorical Data
###### 3. Test Train Split and Normalization
###### 4. Run the Regression and Results
###### 5. Takeaways


In this first data science project I take an in depth look at housing data in Kings County, Washington and try to build a regression model in order to answer which features best indicate price.

The data set includes both continuous and categorical data with over 21,000 data points for each column. Outliers and null values were either eliminated or dealt with accordingly before going on to check for correlation. The categorical data was binned and transformed into dummie variables while the continuous data (in many cases) was normalized through the use of log transformation. Finally the data was split for testing and training and ran in a multiple linear regression model.


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
```


After loading all the libraries we need lets take a look at the correlation matrix

![correlation](https://github.com/cjriggio/dsc-v2-mod1-final-project/blob/master/Correlation%20Matrix.png)

According to the heatmap price seemes to be highly correlated with sqfr_living(0.69) and grade (0.68)


![Regressions](https://github.com/cjriggio/dsc-v2-mod1-final-project/blob/master/partial_regression.png)

+ (top left) The Y and Fitted vs X plot displays the true values of price and the predictions as well as confidence intervals in the form of lines  
+ (top right) The residuals appear to be distributed in a fairly random fashion
+ (bottom left) The partial regression plot shows a slightly positive correlation but is not very linear.
+ (bottom right) The component and component-plus-residual plot is a good indicator of the nature of the relationship

```python
prediction = reg_model.predict(X_test)
from sklearn.metrics import mean_squared_error
print("RMSE:", np.sqrt(mean_squared_error(y_test, prediction)))
prediction = reg_model.predict(X_test)
from sklearn.metrics import mean_squared_error
print("RMSE:", np.sqrt(mean_squared_error(y_test, prediction)))
```
RMSE: 0.0680852

### Takeaways
+ After a great deal of trial and error using log transformation on not only the continuous features but the dependent variable price ended up improving my model a great deal.
+ Pandas and Sklern are two powerful libraries with an amazing amount of functionality that I've only begun to explore
+ Data cleaning, binning and creating dummie variables for categorical datas well as the normalization of continuous values through logarithmic transformation are absolutely vital parts of building regression models. This involves a great deal of trial and error and is by far the most time consuming part of the process.
