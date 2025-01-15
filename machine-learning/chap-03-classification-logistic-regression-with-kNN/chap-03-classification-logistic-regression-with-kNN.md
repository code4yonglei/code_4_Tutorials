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






:::danger
:::
