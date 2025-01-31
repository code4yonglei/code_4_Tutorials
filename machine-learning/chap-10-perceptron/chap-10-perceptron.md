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

![](./fig-01-neutron-perceptron.png)

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



:::danger
:::
