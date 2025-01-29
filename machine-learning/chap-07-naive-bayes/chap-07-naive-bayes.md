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

We can solve this using Bayes' theorem if we know values of terms $P(A)$, $P(B)$, and $P(B|A)$. - $P(A)$ is probability of having disease, which we know to be 0.2%
- $P(B|A)$, or probability of a positive test result given that patient has disease, is test's recall, 0.99
- $P(B)$, probability of a positive test result, it equals to sum of probabilities of true and false positive results
	- $P(positive) = P(positive|disease)P(disease) + P(positive|not-disease)P(not-disease)$
	- here $not-disease$ is a single symbol, not a difference
- probability of a positive test result given that patient has disease = test's recall, 0.99
- probability of this outcome is product of test's recall and probability of having disease, 0.002
- probability of a positive test result given that patient does not have disease is complement of test's specificity, or 0.02
- probability of this outcome is product of complement of test's specificity, 0.02, and complement of probability of having disease, 0.998
- $P(positive) = 0.99*0.002 + 0.02*0.998 = 0.022$

Following is Bayes' theorem re-written in terms of our events:
- $P(disease|positive) = \frac{P(positive|disease)P(disease)}{P(positive|disease)P(disease) + P(positive|not-disease)P(not-disease)}$
- we solve for all terms, and can now solve for conditional probability of
having disease given a positive test result
- $P(disease|positive) = \frac{0.99*0.002}{0.99*0.002+0.02*0.998} = 0.09$
- probability that a patient who tests positive truly has disease is less than 10%
- this seems incorrect
- test's recall and specificity were 99% and 98%, respectively; it is not intuitive that a patient who tests positive is much more likely not to have disease
- while test's specificity and recall are similar, false positives are much more frequent than false negatives because probability of having disease is very small
- in a population of 1000 patients, we expect only 2 to have disease
- with 99% recall, we should expect test to correctly detect these two patients
- however, we should also expect test to incorrectly predict that almost 20 other patients have disease
- only 9% of 22 positive predictions are true positives






:::danger
:::
