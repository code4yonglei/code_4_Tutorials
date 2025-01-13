
# Chapter 1: Fundamentals of Machine Learning


In this chapter, we will
- review fundamental concepts in machine learning 
- will compare supervised and unsupervised learning
- discuss uses of training, testing, and validation data
- describe applications of machine learning
- discuss performance measures that can be used to assess machine learning systems
- introduce scikit-learn, and install tools required in subsequent chapters


## 1.1 Defining machine learning

**Machine learning** is the design and study of software artifacts that use past experience to
inform future decisions, that is, machine learning is the study of programs that learn from data. The fundamental goal of machine learning is to generalize, or to induce an unknown rule from examples of rule's application.


## 1.2 Learning from experience

Machine learning systems are often described as learning from experience either with or without supervision from humans, and is generally categorized into three main types based on nature of learning process and kind of data available.
- **Supervised Learning**
    - In supervised learning, **model is trained on a labeled dataset**, where each input is paired with its corresponding output. The model learns to map inputs to outputs by minimizing prediction errors.
    - Applications:
        - Regression (e.g., predicting house prices, stock prices)
        - Classification (e.g., email spam detection, image recognition)
    - Example Algorithms
        - Linear regression, logistic regression
        - support vector machines (SVMs)
        - decision trees
        - neural networks
- **Unsupervised Learning**
    - In unsupervised learning, **model works with unlabeled data**. It tries to find hidden patterns or intrinsic structures within the data without specific guidance on what to predict.
    - Applications:
        - Clustering (e.g., customer segmentation, grouping similar products)
        - Dimensionality reduction (e.g., Principal Component Analysis for feature reduction)
    - Example Algorithms:
        - K-means clustering
        - hierarchical clustering
        - autoencoders, t-SNE
- **Reinforcement Learning**
    - In reinforcement learning, **an agent learns by interacting with an environment**. It takes actions to maximize cumulative rewards based on feedback (rewards or penalties) from environment.
    - Unlike supervised learning, reinforcement learning programs do not learn from labeled pairs of inputs and outputs. Instead, they receive feedback for their decisions, but errors are not explicitly corrected. For example, a reinforcement learning program that is learning to play a side-scrolling video game like Super Mario Bros may receive a reward when it completes a level or exceeds a certain score, and a punishment when it loses a life. However, this supervised feedback is not associated with specific decisions to run, avoid Goombas, or pick up fire flowers.
    - Applications:
        - Game AI (*e.g.*, AlphaGo)
        - Robotics (*e.g.*, teaching robots to navigate or manipulate objects)
        - Autonomous systems (e.g., self-driving cars)
    - Example Algorithms:
        - Q-learning
        - deep Q-networks (DQN)
        - policy gradient methods

Supervised learning and unsupervised learning can be thought of as occupying opposite
ends of a spectrum. In addition, there are emerging categories and techniques:
- **Semi-Supervised Learning**: Combines a small amount of labeled data with a large amount of unlabeled data to improve learning efficiency.
- **Self-Supervised Learning**: A subset of unsupervised learning where the system generates its own labels from raw data, often used in NLP and computer vision.
- **Online Learning**: Models learn incrementally as new data becomes available, often used in real-time systems.
- **Federated Learning**: Training models across decentralized devices while maintaining data privacy.

> catetories of machine learning? super, rein, semi-super, unsuper

For supervised learning program, **output is referred to response variable** (it has other names, like "dependent variables", "regressands", "criterion variables", "measured variables", "responding variables", "explained variables", "outcome variables", "experimental variables", "labels", and "output variables"). **Input variables as features**, and **phenomena they represent as explanatory variables** (other names include "predictors", "regressors", "controlled variables", and "exposure variables").


## 1.3 Machine learning tasks

Two of the most common supervised machine learning tasks are **classification** and
**regression**.
- In classification tasks, the program must learn to predict discrete values for one or more response variables from one or more features. That is, the program must predict most probable category, class, or label for new observations. Applications of classification include predicting whether a stock's price will rise or fall, or deciding whether a news article belongs to the politics or leisure sections.
- In regression problems, the program must predict the values of one more or continuous response variables from one or more features. Examples of regression problems include predicting the sales revenue for a new product, or predicting the salary for a job based on its description. Like classification, regression problems require supervised learning.

For unsupervised learning tasks
- a common  is to discover groups of related observations, called clusters, within the dataset. This task, called **clustering**, assigns observations into groups such that observations within a groups are more similar to each other based on some similarity measure than they are to observations in other groups. Clustering is often used to explore a dataset. For example, given a collection of movie reviews, a clustering algorithm might discover the sets of positive and negative reviews. The system will not be able to label the clusters as positive or negative; without supervision, it will only have knowledge that the grouped observations are similar to each other by some measure. A common application of clustering is discovering segments of customers within a market for a product. By understanding what attributes are common to particular groups of customers, marketers can decide what aspects of their campaigns to emphasize. Clustering is also used by internet radio services; given a collection of songs, a clustering algorithm might be able to group the songs according to their genres. Using different similarity measures, the same clustering algorithm might group the songs by their keys, or by the instruments they contain.
- **Dimensionality reduction** is another task that is commonly accomplished using unsupervised learning. Some problems may contain thousands or millions of features, which can be computationally costly to work with. Additionally, the program's ability to generalize may be reduced if some of the features capture noise or are irrelevant to the underlying relationship. Dimensionality reduction is the process of discovering the features that account for the greatest changes in the response variable. Dimensionality reduction can also be used to visualize data. It is easy to visualize a regression problem such as predicting the price of a home from its size; the size of the home can be plotted on the graph's x axis, and the price of the home can be plotted on the y axis. It is similarly easy to visualize the housing price regression problem when a second feature is added; the number of bathrooms in the house could be plotted on the z axis, for instance. A problem with thousands of features, however, becomes impossible to visualize.


## 1.4 Training data, testing data, and validation data

The collection of examples that comprise supervised experience is called a **training set**, and that used to assess the performance of a program is called a **test set**.

A program that generalizes well will be able to effectively perform a task with new data. In
contrast, a program that memorizes the training data by learning an overly-complex model
could predict the values of response variable for the training set accurately, but will fail
to predict value of response variable for new examples. Memorizing the training set
is called **overfitting**. Balancing generalization and memorization is a problem to many machine learning algorithms, which can be solved via **regularization**.

In addition to training and test data, a third set of observations, called a **validation or hold-out set**, is sometimes required. **Validation set is used to tune variables called hyperparameters** that control how algorithm learns from training data. Validation set should not be used to estimate real-world performance because program has been tuned to learn from training data in a way that optimizes its score on validation data.

==It is common to partition a single set of supervised observations into training, validation, and test sets==.

When training data is scarce, a practice called **cross-validation** can be used to train and validate a model on same data.
- In cross-validation, training data is partitioned. The model is trained using all but one of partitions, and tested on remaining partition.
- The partitions are then rotated several times so that the model is trained and evaluated on all data.
- The following diagram depicts cross validation with 5 partitions folds.
    - Original dataset is partitioned into 5 folds of equal size labeled Fold 1 through Fold 5.
    - Initially the model is trained on partitions Fold 2-5, and tested on F1
    - In next iteration (split), model is trained on partitions F1, F3-5, and tested on F2
    - Partitions are rotated until models have been trained and tested on all partitions. 
- Cross-validation provides a more accurate estimate of model's performance than testing a single partition of data.

![](./cross-validation.png)










:::danger
:::
