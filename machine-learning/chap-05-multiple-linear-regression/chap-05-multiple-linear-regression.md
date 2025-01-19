# Chapter 5: From Simple to Multiple Linear Regression


- In [Chap. 2, Simple Linear Regression](../chap-02-simple-linear-regression/chap-02-simple-linear-regression.md), we used simple linear regression to model relationship between a single explanatory variable and a continuous response variable.
- In [Chap. 3, Classification and Regression with K-Nearest Neighbors](../chap-03-classification-logistic-regression-with-kNN/chap-03-classification-logistic-regression-with-kNN.md), we introduced kNN and trained classifiers and regressors that use more than one explanatory variable to make predictions.
- In this chapter, we discuss a multiple linear regression, a generalization of simple linear regression that regresses a continuous response variable onto multiple features.
	- we first analytically solve values of parameters that minimize RSS cost function.
	- we then introduce a powerful learning algorithm that can estimate values of parameters that minimize a variety of cost functions, called gradient descent.
	- we then discuss polynomial regression, another special case of multiple linear regression, and learn why increasing model’s complexity can increase risk that it fails to generalize.


## 5.1 Multiple linear regression

In previous chapter, we introduced a simple linear regression model. How to improve it?
- consider factors affecting house's price, no. of rooms, which can be 2nd explanatory variable
- we use a multiple explanatory variables called **multiple linear regression** given by $y = b+ w_1*x_1 + w_2*x_2 + ... + w_n*x_n$
- this model for linear regression can be written in vector notation $Y = WX$
	- which is equivalent to multiple simple linear regression
	- y is a column vector of values of response variables for training examples
	- β/w is a column vector of values of model's parameters
	- X, sometimes called **design matrix**, is an *m \* n* dimensional matrix of values of explanatory variables for training examples
		- m is number of training examples
		- n is number of features

We update training and test data to include number of rooms and age of house
- learning algorithm must estimate values of three parameters: coefficients for two features and intercept term
- we directly solve model parameters using scikit-learn **1-house-price-train.ipynb**

![](./1-house-price-multivarient-linear-regression.png)





:::danger
:::
