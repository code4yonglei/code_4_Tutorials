# Chapter 11: Perceptron to Support Vector Machines

Previous Chap., we introduced perceptron and described why it cannot effectively classify linearly inseparable data.
- recall that we encountered a similar problem in discussion on multiple linear regression
	- we examined a dataset in which response variable was not linearly related to explanatory variables
	- to improve model accuracy, we introduced a special case of multiple linear regression called polynomial regression
	- we created synthetic combinations of features, and were able to model a linear relationship between response variable and features in higher dimensional feature space
	- while this method of increasing dimensions of feature space may seem like a promising technique to use when approximating nonlinear functions with linear models, it suffers from two related problems
		- 1st is a computational problem; computing mapped features and working with larger vectors requires more computing power
		- 2nd problem pertains to generalization; increasing dimensions of feature representation exacerbates curse of dimensionality
	- learning from high-dimensional feature representations requires exponentially more training data to avoid overfitting

This Chap., we discuss a powerful discriminative model for classification and regression, called **support vector machine (SVM)**.
- 1st revisit mapping features to higher dimensional spaces
- 2nd discuss how SVMs mitigate computation and generalization problems encountered when learning from data mapped to highdimensional spaces
	- entire books are devoted to describing SVMs, and describing optimization algorithms used to train SVMs requires more advanced math than we have used in previous chapters
	- instead of working through simple examples in previous chapters, we try to develop an intuition of how SVMs work in order to apply them effectively with scikit-learn


## 11.1 Kernels and kernel trick

Recall that perceptron separates instances of positive class from instances of negative class using a hyperplane as a decision boundary.
- decision boundary is given by formula: $f(x) = <w, x> + b$
- predictions are made using following function $h(x) = sign(f(x))$
- while proof is beyond scope of this chapter, we write model differently
	- expression we used previously is **primal** form
	- following expression of model is called the **dual** form
		- $f(x) = <w,x>+b = \sum \alpha_i y_i <x_i, x> + b$
	- **important difference between primal and dual forms** is that primal form computes inner product of model parameters and test instance's feature vector
	- while dual form computes inner product of training instance's and test instance's feature vector
	- later we will exploit this property of dual form to work with linearly inseparable classes

1st, we must formalize our definition of mapping features to higher dimensional spaces.
- in *polynomial regression* section, we mapped features to a higher dimensional space in which they were linearly related to response variable
- mapping increased number of features by creating quadratic terms from combinations of original features
- these synthetic features allowed us to express a nonlinear function with a linear model
- generally, a mapping is given by $x \rightarrow \phi(x)$ and $\phi: R^d \rightarrow R^D$
- left plot in following figure shows original feature space of a linearly inseparable dataset
- right plot on shows that data is linearly separable after mapping to a higher dimensional space

![](./fig-01-high-dimension.png)

We return to dual form of decision boundary and observation that feature vectors appear only inside of a dot product.
- we can map data to a higher dimensional space by applying mapping to feature vectors
	- $f(x) = \sum \alpha_i y_i <x_i, x> + b$
	- $f(x) = \sum \alpha_i y_i <\phi(x_i), \phi(x)> + b$
- this mapping allows us to express more complex models, but it introduces computation and generalization problems
- mapping feature vectors and computing their dot products can require a prohibitively large amount of processing power
- in 2nd equation that while we mapped feature vectors to a higher dimensional space, feature vectors still only appear as a dot product
	- dot product is a scalar; we do not require mapped feature vectors once this scalar has been computed
	- if we can use a different method to produce same scalar as dot product of mapped vectors, we can avoid costly work of explicitly computing dot product and mapping feature vectors
	- there is a method called **kernel trick**
		- a kernel is a function that, given original feature vectors, returns same value as dot product of its corresponding mapped feature vectors
		- a kernel does not explicitly map feature vectors to a higher dimensional space or calculate dot product of mapped vectors
		- a kernel produces same value through a different series of operations that can often be computed more efficiently
		- a kernel is defined more formally in $K(x,z) = <\phi(x), \phi(z)>$
- we demonstrate how kernels work
	- suppose we have two feature vectors, $x = (x_1, x_2)$ and $z = (z_1, z_2)$
	- in model we wish to map feature vectors to a higher dimensional space using transformation $\phi(x) = x^2$
	- dot product of mapped, normalized feature vectors is equivalent to $<\phi(x), \phi(z)> = <(x_1^2, x_2^2, \sqrt 2 x_1x_2), (z_1^2, z_2^2, \sqrt2 z_1z_2)>$
- kernel given by following formula produces same value as dot product of mapped feature vectors
	- $K(x,z) = <x,z>^2 = (x_1z_1 + x_2z_2)^2 = x_1^2z_1^2 + 2x_1z_1x_2z_2 + x_2^2z_2^2$
	- $K(x,z) = <\phi(x), \phi(z)>$
- we plug in values for feature vectors to make this example more concrete
	- $x = (4,9)$ and $z=(3,3)$
	- $K(x,z) = 1521$
	- $<\Phi(x), \Phi(z)> = <(4^2, 9^2, \sqrt2*4*9), (3^2, 3^2, \sqrt2*3*3)> = 1521$
- kernel $K(x,z)$ produced same value as dot product $\phi(x),\phi(z)$ of mapped feature vectors, but it never explicitly mapped feature vectors to higher dimensional space and required fewer arithmetic operations
	- this example used only 2D feature vectors
- datasets with even a modest number of features can result in mapped feature spaces with massive dimensions
- scikit-learn provides several commonly used kernels, including polynomial, sigmoid, Gaussian, and linear kernels
	- polynomialkernels are given by equation $K(x, x')=(\gamma<x-x'>+r)^k$
		- quadratic kernels, or polynomial kernels with $k=2$, are commonly used in natural language processing
	- sigmoid kernel is given by equation $K(x, x')=tanh(\gamma<x-x'>+r)$
		- $\gamma$ and $r$ are hyperparameters can be tuned through cross-validation
	- Gaussian kernel is a good 1st choice for problems requiring nonlinear models
		- it is a **radial basis function**
		- a decision boundary that is a hyperplane in mapped feature space is similar to a decision boundary that is a hypersphere in original space
		- feature space produced by Gaussian kernel can have an infinite number of dimensions, a feat that would be impossible otherwise
		- Gaussian kernel is given by equation $K(x, x')=exp(-\gamma |x-x'|^2)$

It is always important to scale features when using SVMs, but feature scaling is especially important when using Gaussian kernel.
- choosing a kernel can be challenging
- ideally, a kernel measure similarity between instances in a way that is useful to task
- while kernels are commonly used with SVMs, they can also be used with any model that can be expressed in terms of dot product of two feature vectors, including logistic regression, perceptrons, and **principal component analysis (PCA)**
- in next section, we address 2nd problem caused by mapping to high-dimensional feature spaces: **generalization**


## 11.2 Maximum margin classification and support vectors

Following figure depicts instances from two linearly separable classes and three possible decision boundaries.
- all decision boundaries separate training instances of positive class from training instances of negative class, and a perceptron can learn any of them
- which one is most likely to perform best on test data?, from visualization
	- dotted decision boundary seems to be best
	- solid decision boundary is near many of positive instances
		- test set could contain a positive instance that has a slightly smaller value for 1st explanatory variable, $x_1$
		- this instance would be classified incorrectly
	- dashed decision boundary is farther away from most of training instances
		- however, it is near one of positive instances and one of negative instances

:::danger
:::
