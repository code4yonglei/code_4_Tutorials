
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









:::danger
:::
