# Chapter 2: Simple Linear Regression


In this chapter, we will introduce simple linear regression
- it models simple linear relationship between one response variable and one feature of an explanatory variable
- we will discuss how to fit model via a toy problem as it rarely applicable to real-world problems 


## 2.1 Intro to simple linear regression


### 2.1.1 A tutorial for simple linear regression

Let's model relationship between pizza price ~ pizza size, salary ~ year of experience, or hourse price ~ house size.

Tutorial **1-home-price.ipynb**
- import modules
- plot original data
- linear regression
- replot data with fitting parameters
- save model parameters

![](./1-home-price-predicted.png)

Simple linear regression assumes that a linear relationship exists between response variable and explanatory variable.
- it models this relationship with a linear surface called a **hyperplane**
- a hyperplane is a subspace that has one dimension less than ambient space that contains it
- in simple linear regression, there is one dimension for response variable and another dimension for explanatory variable, for a total of two dimensions
- regression hyperplane has one dimension; a hyperplane with one dimension is a line (green line in figure)

The `LinearRegression` class is an `estimator`, which predict a value based on observed data.
- In scikit-learn, all estimators implement `fit` and `predict` methods.
	- `fit` is used to learn parameters of a model
	- `predict` is used to predict value of a response variable for an explanatory variable using learned parameters.

The `fit` method of `LinearRegression` learns parameters of a linear model **y = w\*x + b** for simple linear regression.
- *x* is explanatory variable, *y* is predicted value of response variable
- coefficient *w* and intercept *b* are parameter of models that are learned by learning algorithm, sometimes they are called **weight** and **bias**

Using training data to learn model parameters for simple linear regression that produce best fitting model is called **ordinary least squares (OLS)** or **linear least squares**.
- we can conduct learning using scikit learn library
- ==we can also using analytical method to obtain model parameters==
- ==we will learn approaches for approximating model parameters that are suitable for larger datasets==


### 2.1.2 Evaluating fitness of model with a cost function

Why the obtained model parameter is the best parameters? If we have several sets of models, how can we assess which set of model parameters produced the best-fitting regression line?

![](./1-home-price-trial-models.png)

A **cost function** (**loss function**) is used to measure error of a model.
- differences between predicted values by trial model parameters and observed data in training set are called **residuals** (**training errors**)
- differences between predicted values and observed data in test data are called **prediction errors** (**test errors**)
- **residuals** are indicated by vertical lines between points for training instances and regression hyperplane in following plot

![](./1-home-price-cost-func.png)

The best predictor can be achieved by minimizing sum of all residuals.
- that is, our model fits if it predicted value for response variables are close to observed values for all training examples
- this measure of model's fitness is called **residual sum of squares (RSS)** cost function
- RSS is calculated with equation $SS_{res} = \sum_{i=1}^n (y_i - f(x_i))^2$, where $y_i$ are observed values and $f(x_i)$ are predicted values


### 2.1.3 Solving OLS for simple linear regression

Simple linear regression is given by equation **y = w\*x + b** and that our goal is to solve OLS to get values of *w* and *b* to minimize cost function. To do so, we need to will calculate **variance of x** and **covariance of x and y**.

**Variance** is a measure of how far a set of values are spread out.
- if all numbers in dataset are equal, variance of this dataset is zero
- a small variance indicates that numbers are near mean of data set
- a dataset containing numbers that are far from mean and from each other will have a large variance
- variance can be calculated using equation $var(x) = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1}$
	- $\bar{x}$ is the mean of x
	- $x_i$ is value of x for $i^{th}$ training instance
	- $n$ is number of training instances
	- **why substract 1?**
		- we substract 1 from number of training instances when calculating sample variance
		- this technique is called **Bessel's correction**
		- it corrects bias in estimation of population variance from a sample
	- variance can also be calculated from Numpy

**Covariance** is a measure of how much two variables change together.
- if two variables increase together, their covariance is positive.
- if one variable tends to increase while other decreases, their covariance is negative
- if there is no linear relationship between two variables, their covariance will be equal to zero; they are linearly uncorrelated but not necessarily independent
- covariance can be calculated using formula $cov(x,y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n-1}$
	- $\bar{y}$ is the mean of y
	- $y_i$ is value of $i^{th}$ training instance
	- Covariance can also be calculated from Numpy

With variance of explanatory variable and covariance of response and explanatory variables
- *w* can be obtained using equation $w = \frac{cov(x,y)}{var(x)}$
- *b* can be obtained using formula $b = \bar{y} - w*\bar{x}$
- obtained values can be compared with model parameters `model.coef_` and `model.intercept_`


## 2.2 Evaluating model

How can we assess whether our model is a good representation of real relationship?
- we can evaluate y value predictor using a measure called **R-squared**.
- known as coefficient of determination, **R-squared** measures how close the data are to a regression line.
- several methods to calculate R-squared
	- in simple linear regression, **R-squared = square of Pearson product-moment correlation coefficient (PPMCC)** (Pearson's r value)
	- using this method, R-squared must be a positive number between 0 and 1
	- other methods, including method used by scikit-learn, R-squared can be negative if model performs extremely poorly
	- R-squared in particular is sensitive to outliers, and can spuriously increase when features are added to model
- We follow method used by scikit-learn to calculate R-squared.
	- $R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = \frac{\sum_{i=1}^n(y_i - f(x_i))^2}{\sum_{i=1}^n(y_i - \bar{y})^2}$
	- `score` method of `LinearRegression` returns model's R-squared value, `model.score(x,y)`

:::danger
:::
