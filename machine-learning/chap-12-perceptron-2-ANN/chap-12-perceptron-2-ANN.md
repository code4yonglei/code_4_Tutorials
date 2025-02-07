# Chapter 12: From Perceptron to Artificial Neural Networks

- In [Chap. 10](../chap-10-perceptron/chap-10-perceptron.md) we introduced perceptron, a linear model for binary classification
	- we learned that perceptron is not a universal function approximator
	- its decision boundary must be a hyperplane
- In [Chap. 11](../chap-11-perceptron-2-SVM/chap-11-perceptron-2-SVM.md) we introduced SVM, which addresses some perceptron's limitations by using kernels to efficiently map feature representations to a higher dimensional space in which classes may be linearly separable.
- In this chapter, we discuss ANNs, powerful nonlinear models for supervised and unsupervised tasks that use a different strategy to overcome perceptron's limitations.
	- if perceptron is analogous to a neuron, an ANN is analogous to a brain
	- as billions of neurons with trillions of synapses comprise a human brain, an ANN is a directed graph of artificial neurons
		- graph's edges are weighted, and these weights are model parameters that must be learned

This chapter provide an overview of structure and training of small, feed-forward artificial neural networks.
- scikit-learn implements neural networks for classification, regression, and feature extraction
	- however these implementations are suitable for only small networks
- training neural networks is computationally expensive
	- in practice, most neural networks are trained using GPU with thousands of parallel processing cores
	- scikit-learn does not support GPUs, and it is not likely to do so in near future
- training neural networks is better served by purpose-built libraries such as Caffe, TensorFlow, and Keras than generalpurpose machine learning libraries such as scikit-learn
	- we will not use scikit-learn to train deep **Convolutional Neural Networks (CNN)** for object recognition or recurrent networks for speech recognition, understanding workings of small networks is an important prerequisite for these tasks


## 12.1 Nonlinear decision boundaries

![](./fig-01-XOR.png)

Recall that while some Boolean functions such as AND, OR, and NAND can be approximated by perceptron, the linearly inseparable function XOR cannot
- we review XOR in detail to develop an intuition of power of ANN
- in contrast to AND (outputs 1 when both inputs = 1) and OR ( outputs 1 when at least one input = 1), output of XOR is 1 when exactly one of its inputs = 1
- we view XOR as outputting 1 when two conditions are true
	- 1st is that at least one input must be = 1; this is same condition that OR tests
	- 2nd is that inputs cannot both = 1; NAND tests this condition
- we can produce same output as XOR by processing input with both OR and NAND, and then verifying that outputs of both functions are equal to 1 using AND
	- that is, functions OR, NAND, and AND can be composed to produce same output as XOR

| A | B | A AND B | A NAND B | A OR B | A XOR B |
|:-:|:-:|:-------:|:--------:|:------:|:-------:|
| 0 | 0 | 0       | 1        | 0      | 0       |
| 0 | 1 | 0       | 1        | 1      | 1       |
| 1 | 0 | 0       | 1        | 1      | 1       |
| 1 | 1 | 1       | 0        | 1      | 0       |

Above figure provides truth tables for XOR, OR, AND, and NAND for inputs A and B.
- we can verify that inputting output of OR and NAND to AND produces same output as inputting A and B to XOR

| A | B | A OR B | A NAND B | (A OR B) AND (A NAND B) |
|:-:|:-:|:------:|:--------:|:-----------------------:|
| 0 | 0 | 0      | 1        | 0                       |
| 0 | 1 | 1      | 1        | 1                       |
| 1 | 0 | 1      | 1        | 1                       |
| 1 | 1 | 1      | 0        | 0                       |

Instead of trying to represent XOR with a single perceptron, we will build an ANN from multiple artificial neurons that each approximate a linear function.
- each instance's feature representation will be input to two neurons
- one neuron will represent NAND and the other will represent OR
- the output of these neurons will be received by a third neuron that represents AND to test that both of XOR's conditions are true


## 12.2 Feed-forward and feedback ANNs

ANNs are described by three components.
- 1st is model's architecture, or topology, which describes types of neuron and structure of connections between them.
- 2nd we have activation functions used by artificial neurons
- 3rd component is learning algorithm that finds optimal values of weights

There are two main types of ANN.
- **feed-forward neural networks** are most common type and are defined by their directed acyclic graphs
	- information travels in one direction only, towards output layer
	- commonly used to learn a function to map an input to an output
	- *i.e.*, a feed-forward net can be used to recognize objects in a photo or predict likelihood that a subscriber of a SaaS product will churn
- **feedback neural networks (or recurrent neural networks)**, contain cycles
	- feedback cycles can represent an internal state for network that can cause network's behavior to change over time based on its input
	- temporal behavior of feedback neural networks make them suitable for processing sequences of inputs
	- feedback neural nets have been used to translate documents between languages and automatically transcribe speech
	- because feedback neural networks are not implemented in scikit-learn, we will limit our discussion to only feed-forward neural networks


## 12.3 Multi-layer perceptrons

Multi-layer perceptron is a simple ANN and its name, however, is a misnomer.
- a multilayer perceptron is not a single perceptron with multiple layers, but rather multiple layers
of artificial neurons that resemble perceptrons
- multi-layer perceptrons have three or more layers of artificial neurons that form a directed, acyclic graph
- generally, each layer is fully connected to subsequent layer
	- output (or activation) of each artificial neuron in a layer is an input to every artificial neuron in next layer
	- features are input through input layer
- simple neurons in input layer are connected to at least one Hidden layer
	- hidden layers represents latent variables
	- these cannot be observed in training data
	- hidden neurons in these layers are often called hidden units
	- finally last hidden layer is connected to an output layer
	- activations of output layer are predicted values of response variable
- following diagram depicts the architecture of a multilayer perceptron with three layers
	- note: input layer is not included in count of a network's layers, but it is counted in `MLPClassifier.n_layers_`

Recall from [Chap. 10](../chap-10-perceptron/chap-10-perceptron.md) that a perceptron has one or more binary inputs, one
binary output, and a Heaviside step activation function.
- a small change to a perceptron's weights may have no effect on its output, or it may cause its output to flip from 1 to 0 or vice versa
- this make it difficult to understand how network's performance changes as we change its weights
- as such, we build our MLP from a different type of neuron
- a **sigmoid neuron** has one or more real-valued inputs and one real-valued outputs, and it uses a sigmoid activation function
	- it allows us to understand how changes to inputs affect output








:::danger
:::
