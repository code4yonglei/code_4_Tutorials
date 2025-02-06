# Chapter 12: From Perceptron to Artificial Neural Networks

- In [Chap. 10](xxx) we introduced perceptron, a linear model for binary classification
	- we learned that perceptron is not a universal function approximator
	- its decision boundary must be a hyperplane
- In [Chap. 11](xxx) we introduced SVM, which addresses some perceptron's limitations by using kernels to efficiently map feature representations to a higher dimensional space in which classes may be linearly separable.
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






:::danger
:::
