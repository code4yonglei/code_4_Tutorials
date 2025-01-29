# Chapter 7: Naive Bayes


Previous chapters introduced two models for classification tasks: kNN and logistic regression.
- here introduce another family of classifiers called Naive Bayes
- named for its use of Bayes' theorem and for its naive assumption that all features are conditionally independent of each other given response variable, Naive Bayes is 1st generative model to discuss
	- 1st introduce Bayes' theorem
	- 2nd compare generative and discriminative models
		- we will discuss Naive Bayes and its assumptions and examine its common variants
		- we will fit a model using scikit-learn


## 7.1 Bayes' theorem

**Bayes' theorem** is a formula for calculating probability of an event using prior
knowledge of related conditions.
- theorem was discovered by Thomas Bayes in 18th century
- Bayes never published his work; his notes were edited and published posthumously by mathematician Richard Price
- Bayes' theorem is given by formula $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ 
	- A and B are events
	- $P(A)$ is probability of observing event A, and $P(B)$ is the probability of observing event B
	- $P(A|B)$ is conditional probability of observing A given that B was observed
- in classification tasks, our goal is to map features of explanatory variables to a discrete response variable; we must find most likely label, A, given features, B

We work through an example
- assume that a patient exhibits a symptom of a particular disease, and that a doctor administers a test for that disease
- this test has been found to have 99% recall and 98% specificity
	- **specificity** measures true negative rate, or proportion of truly negative instances that were predicted to be negative 
	- specificity and recall are often used to evaluate medical tests
	- **recall is sometimes called sensitivity**
		- 99% recall means that 99% patients who truly have disease were predicted to have it
		- 98% specificity means that 98% patients who truly do not have disease were predicted not to have it
- also assume that disease is rare, probability that a person in population has it is only 0.2%
- if a patient's test result is positive, what is probability that she actually has disease?
- what is conditional probability of having disease, A, given a positive test result, B?









:::danger
:::
