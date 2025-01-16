# Chapter 3: Classification and Logistic Regression with k-Nearest Neighbors


In this chapter, we will introduce `k-Nearest Neighbors (KNN)` algorithm for classification and logistic regression tasks.


## 3.1 k-Nearest Neighbors

kNN is a simple model for regression and classification tasks.
- The titular neighbors are representations of training instances in a **metric space**, which is a feature space in which contains distances between all members in a data set.
- These neighbors are used to estimate value of response variable for a test instance.
- The **hyperparameter k** specifies how many neighbors can be used in estimation.
	- A hyperparameter is a parameter that controls how algorithm learns;
	- hyperparameters are not estimated from training data and are sometimes set manually.
	- The final k neighbors selected are those that are nearest to test instance, as measured by some distance function.

For classification tasks, a set of tuples of feature vectors and class labels comprise training set.
- kNN is a capable of binary, multi-class, and multi-label classification.
- The simplest kNN classifiers use mode of kNN labels to classify test instances, but other strategies can be used.
- `k` is often set to an odd number to prevent ties.
- In regression tasks, feature vectors are each associated with a response variable that takes a realvalued scalar instead of a label. The prediction is mean or weighted mean of kNN response variables.


## 3.2 Lazy learning and non-parametric models

kNN is a **lazy learner** (known as **instance-based learners**), it simply store training dataset with little or no processing.
- In contrast to **eager learners** (such as **simple linear regression**), kNN does not estimate parameters of a model that generalizes training data during a training phase.
- Lazy learning has advantages and disadvantages.
	- Training an eager learner is often computationally costly, but prediction with resulting model is often inexpensive.
		- For simple linear regression, prediction consists only of multiplying learned coefficient by feature, and adding learned intercept parameter.
	- A lazy learner can predict almost immediately, but making predictions can be costly.
		- In simple implementation of kNN, prediction requires calculating distances between a test instance and all training instances.

In contrast to most of other models that we will discuss, **kNN is a non-parametric model**.
- A parametric model uses a fixed number of parameters, or coefficients, to define the model that summarizes the data.
	- The number of parameters is independent of number of training instances.
- Non-parametric may seem to be a misnomer, as it does not mean that the model has no parameters;
	- Non-parametric means that number of parameters of model is not fixed, and may grow with number of training instances.
	- Non-parametric models can be useful when training data is abundant and you have little prior knowledge about relationship between response and explanatory variables.
		- kNN makes only one assumption: instances that are near each other are likely to have similar values of response variable.
		- This flexibility provided by non-parametric models is not always desirable;
	- A model that makes assumptions about the relationship can be useful if training data is scarce or if you know relationship.


## 3.3 Classification with kNN

Tutorial **1-height-weight-gender** using a person's height and weight (two explanatory variables) to predict gender (response variable).
- This is called **binary classification** because response variable can take one of two labels.
- kNN is not limited to two features as kNN algorithm can use an arbitrary number of features, but more than three features cannot be visualized.
- Male denoted by red O and female denoted by green markers.

![](1-height-weight-gender-plot.png)


### 3.3.1 Prediction using Euclidean distance

Let's use **Euclidean distance** to predict whether a person with a given height (155 cm) and weight (70 kg) is a man or a woman. 
- First define distance measure.
	- **Euclidean distance** = the straight distance between points in a Euclidean space. 
	- Euclidean distance in a two-dimensional space is given by formula $d(p,q) = d(q,p) = \sqrt{(q_1-p_1)^2 + (q_2-p_2)^2}$
	- Cal distance between test point to all other points
- Set $k=3$ and select 3 (marked as `x` in figure) nearest training instances
	- blue O is test point
	- 2 neighbors are female and 1 is male. We therefore predict that test instance is female


### 3.3.2 Implementation of a kNN classifier using scikit-learn

Implement a kNN classifier using scikit-learn
- 1st use `LabelBinarizer` to convert labels (famele and male) to integers as our labels are strings 
	- `LabelBinarizer` implements **transformer interface**, which consists of `fit`, `transform`, and `fit_transform` methods.
	- `fit` method prepares transformer; here it creates a mapping from label strings to integers.
	- `transform` method applies mapping to input labels.
	- `fit_transform` is convenient to call `fit` and `transform`.
		- A transformer should be fit only on training set.
		- Independently fitting and transforming the training and testing sets could result in inconsistent mappings from labels to integers; here male might be mapped to 1 in training set and 0 in testing set.
		- Fitting on entire dataset should also be avoided because for some transformers it will leak information about testing set into model. This advantage won't be available in production, so performance measures on test set may be optimistic.
- 2nd initialize `KNeighborsClassifier`
	- Even though kNN is a lazy learner, it still implements estimator interface
	- call `fit` and `predict` just as we did with simple linear regression object
- 3rd use our fit `LabelBinarizer` to reverse transformation and return a string label

Validate the kNN classifier via predictions for a test set
- 4 test points and got 3 right prediction
- Reason: Recall from *Chapter 1*, there are two types of errors in binary classification tasks: **false positives** and **false negatives**.
- There are many performance measures for classifiers
	- some measures may be more appropriate than others depending on consequences of types of errors in application.
	- we will **assess our classifier using several common performance measures, including accuracy, precision, and recall**.
		- Accuracy is proportion of test instances that were classified correctly, here it is 75%.
		- Precision is proportion of test instances that were predicted to be positive that are truly positive. here positive class is male (male=1). The assignment of male and female to positive and negative classes is arbitrary, and could be reversed. Our classifier predicted that one of test instances is positive class. This instance is truly positive class, so classifier's precision is 100%:
		- Recall is proportion of truly positive test instances that were predicted to be positive. Our classifier predicted that one of two truly positive test instances is positive. Its recall is therefore 50%.
	- Sometimes it is useful to summarize precision and recall with a single statistic, called **F1 score or F1 measure**. 
		- F1 score is harmonic mean of precision and recall
		- Note that arithmetic mean of precision and recall scores is upper bound of F1 score. F1 score penalizes classifiers more as difference between their precision and recall scores increases. 
	- **Matthews correlation coefficient (MCC)** is an alternative to F1 score for measuring performance of binary classifiers.
		- A perfect classifier's MCC is 1.
		- A trivial classifier that predicts randomly will score 0.
		- A perfectly wrong classifier will score -1.
		- MCC is useful even when proportions of classes in test set is severely imbalanced.
- scikit-learn provides a `classification_report` function that reports precision, recall, and F1 score


## 3.4 Regression with kNN


### 3.4.1 kNN for a regression task

> Tutorial `2-height-gender-weight.ipynb`

Using kNN for a regression task to predict weight from height and gender
- instantiate and fit `KNeighborsRegressor`, and use it to predict weights
- in dataset, gender is coded as a binary-valued feature (1 for male and 0 for female), while height values range from 155 to 191 (this is a problem and will discuss how it can be ameliorated).
- adopt two performance measures for regression tasks--**MeanAbsolute Error (MAE)** and **Mean Squared Error (MSE)**:
	- **MAE** is average of absolute values of errors of predictions, $MAE = \frac{1}{n} \sum_{i=0}^{n-1} | y_i - \hat{y_i} |$
	- **MSD** is average of squares of errors of predictions, $MSE = \frac{1}{n} \sum_{i=0}^{n-1} ( y_i - \hat{y_i} )^2$

```
Coefficient of determination: 0.6290565226735438
Mean absolute error (MAE): 8.333333333333336
Mean squared error (MAE): 95.8888888888889
```

It is important that regression performance measures disregard directions of errors; otherwise, errors of a regressor that under- and over-predicts equally would cancel out.
- MSE and MAE accomplish this by squaring errors and taking absolute values of errors.
- MSE penalizes outliers more than MAE; squaring a large error makes it contribute disproportionately more to total error.
	- This may be desirable in some problems, but MSE is often preferred even when it is not, as MSE has useful mathematical properties.
- Note that for ordinary linear regression, such as simple linear regression problem, minimizes square root of MSE.


### 3.4.2 Scaling features

> Tutorial `2-height-gender-weight.ipynb`

Many learning algorithms work better when features take similar ranges of values.
- In this regression task, we used two features
	- a binary-valued feature representing gender
	- a continuous-valued feature representing height in cm.
- Consider a dataset in which a man is 170 cm tall and a woman is 160 cm tall.
	- which instance is closer to a man who is 164 cm tall?
	- For weight prediction problem, we probably believe that query is closer to male instance; a 6 cm difference in height is less important to predicting weight than difference between gender.
	- If we represent height in mm, query instance is closer to 1600 mm tall female.
	- If we represent height in m, query instance is closer to 1.7 meter tall male.
	- If we represent heights in micrometers, height feature would dominate distance function even more.

![]()

scikit-learn's `StandardScaler` is a transformer that scales features so that they have unit variance.
- It first centers features by subtracting mean of each feature from each instance's value of feature.
- It then scales features by dividing each instance's value of feature by standard deviation of feature.
- Data that has zero mean and unit variance is **standardized**.
	- Like `LabelBinarizer`, `StandardScaler` implements transformer interface.
- We standardize previous problem's features, fit regressor, and compare performances of two models
	```
	Coefficient of determination: 0.6706425961745109
	Mean absolute error (MAE): 7.583333333333336
	Mean squared error (MAE): 85.13888888888893
	```
- The model performs better on standardized data.
	- The feature representing gender contributes more to distance between instances and allows model to make better predictions.

![]()


:::danger
:::
