# Chapter 2: Simple Linear Regression


In this chapter, we will introduce simple linear regression
- it models simple linear relationship between one response variable and one feature of an explanatory variable
- we will discuss how to fit model via a toy problem as it rarely applicable to real-world problems 


## 2.1 Intro to simple linear regression


### 2.1.1 A tutorial for simple linear regression

Let's model relationship between pizza price ~ pizza size, salary ~ year of experience, or hourse price ~ house size.

The code example in **1-home-price** jupyter notebook.
- import modules
- plot original data
- linear regression
- replot data with fitting parameters
- save model parameters

![](./1-house-price-1.png)

Simple linear regression assumes that a linear relationship exists between response
variable and explanatory variable.
- it models this relationship with a linear surface called a **hyperplane**.
- A hyperplane is a subspace that has one dimension less than ambient space that contains it.
- In simple linear regression, there is one dimension for response
variable and another dimension for explanatory variable, for a total of two dimensions.
- The regression hyperplane thus has one dimension; a hyperplane with one dimension is a
line (green line in figure).

The `LinearRegression` class is an `estimator`, which predict a value based on
observed data.
- In scikit-learn, all estimators implement `fit` and `predict` methods.
	- `fit` is used to learn parameters of a model
	- `predict` is used to predict value of a response variable for an explanatory variable using learned parameters.

The `fit` method of `LinearRegression` learns parameters of a linear model **y = w*x + b** for simple linear regression.
- **x** is explanatory variable, and **y** is predicted value of response variable
- coefficient **w** and intercept **b** are parameter of models that are learned by learning algorithm, sometimes they are called **weight** and **bias**

Using training data to learn model parameters for simple linear regression that produce best fitting model is called **ordinary least squares (OLS)** or **linear least squares**.
- we can conduct learning using scikit learn library
- ==we can also using analytical method to obtain model parameters==
- ==we will learn approaches for approximating model parameters that are suitable for larger datasets==


### 2.1.2 Evaluating fitness of model with a cost function

Why the obtained model parameter is the best parameters? If we have several sets of models (in figure), how can we assess which set of model parameters produced the best-fitting regression line?

![](./1-house-price-2.png)

A **cost function** (**loss function**) is used to measure error of a model.
- The differences between predicted values by trial model parameters and observed data in training set are called **residuals** (**training errors**).
- The differences between predicted values and observed data in test data are called **prediction errors** (**test errors**).
- **residuals** are indicated by vertical lines between points for training instances and regression hyperplane in following plot:

![](./1-house-price-3.png)

The best predictor can be achieved by minimizing the sum of all residuals. That
is, our model fits if itt predicted value for response variables are close to observed values for all training examples. This measure of model's fitness is called **residual sum of squares (RSS)** cost function. RSS is calculated with equation $SS_{res} = \sum_{i=1}^n (y_i - f(x_i))^2$, where $y_i$ are observed values and $f(x_i)$ are predicted values









:::danger
:::
