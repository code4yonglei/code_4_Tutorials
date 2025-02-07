# Chapter 13: K-Means

Previous chapters, we discussed supervised learning tasks, examined algorithms for regression and classification that learned from labeled training data.

This chapter, we introduce our first unsupervised learning task: clustering.
- clustering is used to find groups of similar observations within a set of unlabeled data
- we discuss K-means clustering algorithm, apply it to an image compression problem, and learn to measure its performance
- we work through a semi-supervised learning problem that combines clustering with classification


## 13.1 Clustering

Recall from [Chap. 1](xxx), goal of unsupervised learning is to discover hidden structures or patterns in unlabeled training data.
- **clustering**, or **cluster analysis**, is task of grouping observations so that members of same group, or cluster, are more similar to each other by some metric than they are to members of other clusters
- as with supervised learning, we represent an observation as an n-dimensional vector
- *i.e.*, assume that your training data consists of samples plotted in following figure (same size, color and shape)
- clustering might produce following four groups indicated by different shapes and colors
- clustering is commonly used to explore a dataset
	- social networks can be clustered to identify communities and to suggest missing connections between people
	- in biology, clustering is used to find groups of genes with similar expression patterns
	- recommendation systems sometimes employ clustering to identify products or media that might appeal to a user
	- in marketing, clustering is used to find segments of similar consumers
	- In following sections, we work through an example of using K-means algorithm to cluster a dataset






:::danger
:::
