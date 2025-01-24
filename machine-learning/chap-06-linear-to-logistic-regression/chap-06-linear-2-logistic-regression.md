# Chapter 6: From Linear to Logistic Regression


Previously, we discussed simple, multiple, and polynomial linear regression. These models are special cases of **generalized linear model**, a flexible framework that requires fewer assumptions than ordinary linear regression.
- here we discuss these assumptions as they relate to **logistic regression** (another special case of generalized linear model).
	- logistic regression is used for classification tasks
	- recall that goal of classification tasks is to induce a function that maps an observation to its associated class or label
- a learning algorithm must use pairs of feature vectors and their corresponding labels to induce values of mapping function's parameters that produce best classifier, as measured by some performance metric
	- in binary classification, classifier must assign instances to one of two classes
	- in multi-class classification, classifier must assign one of several labels to each instance
	- in multi-label classification, classifier must assign a subset of labels to each instance
- in this chapter, **we work on several classification problems using logistic regression**
	- discuss performance measures for classification task
	- apply some feature extraction techniques in [Chap. 4: Feature Extraction](../chap-04-feature-extraction/).


## 6.1 Binary classification with logistic regression

Ordinary linear regression assumes that response variable is normally distributed.
- **normal distribution** (or **Gaussian distribution**) is a function that describes probability that an observation will have a value between any two real numbers
	- normally distributed data is symmetrical
	- half values are greater than mean and half are less than mean
	- many natural phenomena are approximately normally distributed, *i.e.*,
		- height of people is normally distributed, most people are of average height, a few are tall, and a few are short
	- response variable for some problems is not normally distributed, *i.e.*,
		- a coin toss can result in two outcomes: heads or tails
- **bernoulli distribution** describes probability distribution of a random variable that can take positive case with probability *P* or negative case with probability *1-P*.
	- if response variable represents a probability, it must be constrained to [0, 1]


