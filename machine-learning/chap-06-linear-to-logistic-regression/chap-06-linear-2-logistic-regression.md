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

Linear regression assumes that a constant change in value of a feature results in a constant change in value of response variable, an assumption that cannot hold if value of response variable represents a probability.
- generalized linear models remove this assumption by relating a linear combination of features to response variable using a link function.
	- we have already used a link function in [Chap. 2: Simple Linear Regression](../chap-02-simple-linear-regression/chap-02-simple-linear-regression.md)
	- ordinary linear regression is a special case of generalized linear model that links a linear combination of features to a normally distributed response variable using identity function
	- we can use a different link function to relate a linear combination of features to a response variable that is not normally distributed

In logistic regression, response variable describes probability that outcome is positive case. 
- if response variable is equal to or exceeds a discrimination threshold, positive class is predicted; otherwise, negative class is predicted
- response variable is modeled as a function of a linear combination of features using **logistic function**
	- logistic function always returns a value between 0 and 1: $F(t)=\frac{1}{1+e^{-t}}$
	- following plot of value of logistic function for range [-6, 6]
	- for logistic regression, *t=* a linear combination of explanatory variables as $F(t)=\frac{1}{1+e^{-(b+wx)}}$
	- - **logit function** is inverse of logistic function
		- it links $F(x)$ back to a linear combination of features $g(x) = ln\frac{F(x)}{1-F(x)} = b + wx$
		- model's parameters can be estimated using a variety of learning algorithms, including gradient descent

![](./0-plot-logistic-func.png)


## 6.2 Spam filtering

Our first problem is a modern version of canonical binary classification problem: spam filtering.
- we will classify spam and ham sms messages rather than e-mail
- we will extract tf-idf features from messages using techniques shown in previous chapters, and classify messages using logistic regression
- we use *SMS Spam Collection Data Set* from *UCI Machine Learning Repository*.
	- download dataset from [HERE](http://archive.ics.uci.edu/dataset/228/sms+spam+collection)

==**Tutorial: 1-sms-spam-collection.ipynb**==

First explore dataset and calculate basic summary statistics using pandas
- `df.head()`
- `df[df[0] == 'spam'][0].count()`
	- each row comprises a binary label and a text message
	- dataset contains 5574 instances; 4825 messages are *ham* and remaining 747 are *spam*

We make some predictions using scikit-learn's `LogisticRegresion` class.
- 1st split dataset into training and test sets. default `train_test_split` assigns 75% samples to training
- 2nd create a `TfidfVectorizer`,which combines `CountVectorizer` and `TfidfTransformer`
	- we fit it with training messages and transform both training and test messages
- 3rd create an instance of `LogisticRegression` and train the model
	- `LogisticRegression` implements `fit` and `predict` methods
- linear regression performs bad
- below we discuss some performance metrics to evaluate binary classifiers


### 6.2.1 Binary classification performance metrics

A variety of metrics exist for evaluating performance of binary classifiers against trusted labels.
- most common metrics are accuracy, precision, recall, F1 measure, and ROC AUC score.
- all depend on concept of true positives, true negatives, false positives, and false negatives
- positive and negative refer to classes
- true and false denote whether predicted class is same as true class
	- for SMS spam classifier
	- a true positive prediction is when classifier correctly predicts that a message is spam
	- a true negative prediction is when classifier correctly predicts that a message is ham
	- a prediction that a ham message is spam is a false positive prediction
	- a spam message incorrectly classified as ham is a false negative prediction
- a **confusion matrix**, or contingency table, to visualize true and false positives and negatives
	- rows of matrix are true classes of instances, columns are predicted classes of instances

![](./1-sms-spam-collection-confusion-matrix.png)


### 6.2.2 Accuracy

Accuracy measures fraction of classifier's predictions that are correct.
- `LogisticRegression.score` predicts scores labels for a test set using accuracy
	- `model.score(x_test, y_test))` is value of truly predicted values in diagonal cells over all instances
	- `cross_val_score(model, x_train, y_train, cv=5)`
	- `cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")`
		- default is `scoring="accuracy"`
	- `cross_val_score(model, x_train, y_train, cv=5, scoring="balanced_accuracy")`
	- use `cross_validate` to run both `accuracy` and `balanced_accuracy`
- accuracy measures overall correctness of classifier, it does not distinguish between false positive errors and false negative errors
	- some applications may be more sensitive to false negatives than false positives
	-  accuracy is not an informative metric if proportions of classes are skewed in population
	- *i.e.*, a classifier that predicts whether or not credit card transactions are fraudulent may be more sensitive to false negatives than to false positives
	- to promote customer satisfaction, credit card company may prefer to risk verifying legitimate transactions rather than risk ignoring a fraudulent transaction
	- because most transactions are legitimate, accuracy is not an appropriate metric for this problem
- a classifier that always predicts that transactions are legitimate could have a high accuracy score but may not be useful
- for these reasons, classifiers are often evaluated using **precision and recall**


### 6.2.3 Precision and recall

- **Precision** is fraction of positive predictions that are correct
	- in SMS spam classifier, precision is fraction of messages classified as spam that are actually spam
- **Recall** is fraction of truly positive instances that classifier recognized, sometimes called sensitivity in medical domains
	- a recall score of 1 indicates that classifier did not make any false negative predictions
	- for SMS spam classifier, recall is fraction of truly spam messages that were classified as spam
- individually, precision and recall are seldom informative; they are both incomplete views of a classifier's performance
	- both precision and recall can fail to distinguish classifiers that perform well from certain types of classifiers that perform poorly
	- a trivial classifier could easily achieve a perfect recall score by predicting positive for every instance
- *i.e.*, assume that a test set contains 10 positive examples and 10 negative examples
	- a classifier that predicts positive for every example will achieve a recall of 1
	- a classifier that predicts negative for every example, or one that makes only false positive and true negative predictions, will achieve a recall score of 0
	- similarly, a classifier that predicts that only a single instance is positive and happens to be correct will achieve perfect precision
- we calculate SMS classifier's precision and recall
	- classifier's precision showes that almost messages that it predicted as spam were actually spam
	- its recall is lower, indicating that it incorrectly classified approximately 32% of spam messages as ham


### 6.2.4 Calculating F1, F0.5, and F2 measures

**F1 measure is harmonic mean of precision and recall scores**
- it penalizes classifiers with imbalanced precision and recall scores, like trivial classifier that always predicts positive class
	- a model with perfect precision and recall scores will achieve an F1 score of 1
	- a model with a perfect precision score and a recall score of 0 will achieve an F1 score of 0
- we compute our classifier's F1 score
	- models are sometimes evaluated using the F0.5 and F2 scores
	- F0.5 and F2 scores bias precision over recall and recall over precision, respectively




:::danger
:::
