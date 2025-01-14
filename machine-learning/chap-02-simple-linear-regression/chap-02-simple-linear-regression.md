# Chapter 2: Simple Linear Regression


In this chapter, we will introduce simple linear regression
- it models simple linear relationship between one response variable and one feature of an explanatory variable
- we will discuss how to fit model via a toy problem as it rarely applicable to real-world problems 


## 2.1 Intro to simple linear regression

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




:::danger
:::
