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




