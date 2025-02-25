# Chapter 10: Perceptron

We have discussed linear models such as multiple linear regression and logistic regression.
- here we introduce another linear model for binary classification tasks called **perceptron**
- while perceptron is seldom used today, understanding it and its limitations is important in order to understand models in following chapters


## 10.1 Perceptron


### 10.1.1 Neuron and perceptron

Invented by Frank Rosenblatt at Cornell Aeronautical Laboratory in late 1950s, development of perceptron was originally motivated by efforts to simulate human brain.
- a brain is composed of cells called **neurons** that process information
- connections between neurons are called **synapses**, through which information is transmitted
- human brain has been estimated to be composed of as many as 100 billion neurons and 100 trillion synapses
- as illustrated below, main components of a neuron are dendrites, a body, and an axon
	- dendrites receive electrical signals from other neurons
	- signals are processed in neuron's body, which then sends a signal through axon to another neuron

![](./fig-01-neuron-perceptron.png)

An individual neuron can be thought of as a computational unit that processes one or more inputs to produce an output.
- a perceptron functions analogously to a neuron, it accepts one or more inputs, processes them, and returns an output
- a model of just one of hundreds of billions of neurons in human brain will be of limited use, that is true
- a single perceptron is incapable of approximating many functions
- however, we will still discuss perceptrons for two reasons
	- 1st, perceptrons are capable of online learning
		- learning algorithm can update model's parameters using a single training instance rather than entire batch of training instances
		- online learning is useful for learning from training sets that are too large to be represented in memory
	- 2nd, understanding perceptron and its limitations is necessary for understanding some powerful models in subsequent chapters, including support vector machines and artificial neural networks

Perceptrons are commonly visualized using a diagram shown above.
- circles labeled $x_1$, $x_2$, and $x_m$ are input units, and each input unit represents one feature
- perceptrons frequently use an additional input unit that represents a constant bias term, but this input unit is usually omitted from diagrams
- circle in center is a computational unit, or neuron's body
- edges (arrows) connecting input units to computational unit are analogous to dendrites
- each edge (arow) is associated with a parameter, or weight
	- parameters can be interpreted easily
	- a feature that is correlated with positive class will have a positive weight
	- a feature that is correlated with negative class will have a negative weight
- edge directed away from computational unit returns output (right circle), and can be thought of as axon


### 10.1.2 Activation functions

Perceptron classifies instances by processing a linear combination of features and model parameters using an activation function:
$$
y = \phi (\sum_{i=1}^n w_i x_i + b)
$$
- $w_i$ are model's parameters, $b$ is a constant bias term, $\phi$ is activation function
- a linear combination of parameters and inputs is sometimes called **preactivation**
- several different activation functions are commonly used
	- Rosenblatt's original perceptron used Heaviside step function
	- Heaviside step function is also called unit step function as $g(x) = 1 (x>0) and 0 (others)$
		- x is weighted combination of features
		- if weighted sum of features and bias term > 0, activation function returns 1 and perceptron predicts that instance is positive class
		- otherwise, function returns 0 and perceptron predicts that instance is negative class
	- another common activation function is logistic sigmoid $g(x) = \frac{1}{1+e^{-x}}$
		- $x$ is weighted sum of inputs
		- unlike unit step function, logistic sigmoid is differentiable
		- this difference will become important when discuss artificial neural networks


### 10.1.3 Perceptron learning algorithm

Perceptron learning algorithm begins by setting weights to zero, or to small random values, and then predicts class for a training instance.
- perceptron is an error-driven learning algorithm
	- if prediction is correct, algorithm continues to next instance
	- if incorrect, algorithm updates weights
	- formally, update rule is given by $w_i(t+1) = w_i(t) + \alpha(d_j-y_j(t))x_{j,i}$
	- for each training instance, parameter value for each feature is incremented by $\alpha(d_j-y_j(t))x_{j,i}$
		- $d_j$ is true class for instance $j$, $y_j(t)$ is predicted class for instance $j$
		- $x_{j,i}$ is value of ith feature for instance $j$
		- $\alpha$ is a hyperparameter that controls learning rate
	- if prediction is correct, $d_j-y_j(t)=0$, and $\alpha(d_j-y_j(t))x_{j,i} = 0$
		- that is, if prediction is correct, weight is not updated
		- if prediction is incorrect, we compute $d_j-y_j(t)$, value of feature, and learning rate
		- then add product (which may be negative) to weight

This update rule is similar to update rule for gradient descent in that weights are adjusted towards classifying instance correctly and size of update is controlled by a learning rate.
- each pass through training instances is called an **epoch**
- learning algorithm has converged when it completes an epoch without misclassifying any instances
- learning algorithm is not guaranteed to converge
	- later will discuss linearly inseparable datasets for which convergence is impossible
	- for this reason, learning algorithm also requires a hyperparameter that specifies maximum number of epochs that can be completed before algorithm terminates


### 10.1.4 Binary classification with perceptron

We work through a toy classification problem to separate adult cats from kittens
- only two explanatory variables are available in dataset: proportion of day when animal was asleep and proportion of day when animal was grumpy
- training data consists of following four instances
- scatter plot shown below confirms that they are linearly separable

| Instance | Proportion of day sleeping | Proportion of day grumpy | Kitten or adult |
| :-: | :-: |:-: | :-: |
| 1 | 0.2 | 0.1 | Kitten |
| 2 | 0.4 | 0.6 | Kitten |
| 3 | 0.5 | 0.2 | Kitten |
| 4 | 0.7 | 0.9 | Adult  |

![](./1-kitten-adult-cat-data.png)

Goal is to train a perceptron that can classify animals using two real-valued features.
- we represent kittens with positive class and adult cats with negative class
- perceptron has three input units
	- $x_1$ is input unit for bias term
	- $x_2$ and $x_3$ are input units for two features
- perceptron's computational unit uses unit step activation function
- here we set maximum number of training epochs to 10
- if algorithm does not converge within epochs, it will stop and return current values of weights
- for simplicity, we set learning rate to 1, and all weights to 0 initially
- in 5 epochs, we got results as shown below

![](./1-kitten-adult-cat-epoch.png)


### 10.1.5 Document classification with perceptron

Like other estimators, `Perceptron` class implements `fit` and `predict` methods and hyperparameters are specified through its constructor.
- `Perceptron` also implements a `partial_fit` method, allowing classifier to be trained incrementally

**Tutorial: 2-rec-sports.ipynb**

Here, we train a perceptron to classify documents from 20 Newsgroups dataset.
- dataset consists of approximately 20000 documents sampled from 20 Usenet newsgroups
- dataset is commonly used in document classification and clustering experiments
	- scikit-learn even provides a convenience function for downloading and reading dataset
- we train a perceptron to classify documents from three newsgroups *rec.sports.hockey*, *rec.sports.baseball*, and *rec.auto*
- perceptron is capable of multiclass classification
	- it use one-versus-all strategy to train a classifier for each class in training data
- we represent documents as tf-idfweighted bags-of-words
- `partial_fit` method can be used in conjunction with `HashingVectorizer` to train from large to streaming data in a memory-constrained setting
	- 1st download and read dataset using `fetch_20newsgroups` function
		- consistent with other built-in datasets, function returns an object with *data*, *target*, and *target_names* attributes
	- we specify that documents' headers, footers, and quotes should be removed
	- each newsgroup used different formatting conventions in headers and footers
		- retaining them makes classifying documents artificially easy
	- we produce tf-idf vectors using `TfifdVectorizer`, train perceptron, and evaluate it on test set
	- without hyperparameter optimization, perceptron's average precision, recall, and F1 score are 0.84


## 10.2 Limitations of perceptron

Perceptron uses a hyperplane to separate positive and negative classes.
- a simple example of a classification problem that is linearly inseparable is logical exclusive disjunction, or *XOR* 
- output of XOR is 1 when one input = 1 and another = 0, otherwise output = 0
- inputs and outputs of *XOR* are plotted in two dimensions below
- when *XOR* outputs 1, instance is marked with a circle
- when *XOR* outputs 0, instance is marked with a diamond
- it is impossible to separate circles from diamonds using a single straight line

![](./fig-02-XOR.png)

Suppose that instances are pegs on a board.
- if you were to stretch a rubber band around both positive instances, and stretch a second rubber band around both negative instances, bands would intersect in middle of board.
- rubber bands represent **convex hulls**, or envelope that contains all points within set and all points along any line connecting a pair points within set
- feature representations are more likely to be linearly separable in higher dimensional spaces than lower dimensional spaces
- *i.e.*, text classification problems tend to be linearly separable when high dimensional representations such as bag-of-words are used

In following two chapters, we discuss techniques that can be used to model linearly inseparable data.
- 1st technique, called **kernelization**, projects linearly inseparable data to a higher dimensional space in which it is linearly separable.
	- kernelization can be used in many models, including perceptrons, but it is particularly associated with support vector machines
- 2nd technique creates a directed graph of perceptrons
	- the resulting model, called an **Artificial Neural Network (ANN)**, is a universal function approximator


## 10.3 Summary

Inspired by neurons, perceptron is linear model for binary classification.
- perceptron classifies instances by processing a linear combination of features and weights with an activation function
- while a perceptron with a logistic sigmoid activation function is same model as logistic regression, perceptron learns its weights using an online, error-driven algorithm
- like other linear classifiers, perceptron separates instances of positive and negative classes using a hyperplane
- some datasets are not linearly separable
	- that is, no possible hyperplane can classify all instances correctly
	- in following chapters, we will discuss two models that can be used with linearly inseparable data
		- ANN, which creates a universal function approximator from a graph of perceptrons
		- SVM, which projects data onto a higher dimensional space in which it is linearly separable

:::danger
:::
