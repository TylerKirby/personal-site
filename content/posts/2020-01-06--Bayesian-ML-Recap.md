---
title: What I Learned in Bayesian Machine Learning
date: "2020-01-06"
template: "post"
draft: true
slug: "/posts/bayesian-ml-recap"
category: "Machine Learning"
# tags:
#   - "Theory of Machine Learning"
description: "A review of what I found most interesting from my Bayesian ML class"
---
The Bayesian machine learning course I've recently completed has probably has the most profound impact on my knowledge of data science of any course I have taken in graduate school so far. Bayesian statistics in general seems to be more intuitive to me than Frequentist's interpretations and the core algorithms presented in the course have many useful properties. In this post, I'll highlight what I think are the key ideas in the field and how they could be used for everyday data science.

## The Bayesian Method
All of the key results in Bayesian machine learning can be derived from first principles using the Bayesian method. In its simplest formulation, the method has three primary steps:
1. Use you prior beliefs in the beginning to address the problem
2. Act on this prior
3. Observe the results and update your beliefs

Using what we know about the world to solve new problems is the foundation of the Bayesian method. In fact it is the very essence of Bayesian statistics and its prime point of difference with Frequentists methods. The use of a prior falls naturally out of the Bayesian definition of probability. Whereas the Frequentists define probability as the **relative frequency of occurrence of an event**, Bayesians define probability as the **degree of belief**. Therefore Bayesians can assign nonzero probabilities to events that have not occurred. For example, a Bayesian may very well say that the probability of life on mars is 0.01 but a Frequentist will be at a loss since there has been no occurence of the event to establish the probability. The fundemental difference between Bayesian and Frequentist statistics is this philosophical distinction. Personally, I believe that the Bayesian interpretation of probability is more useful and practical though I admit that it does invite a large amount of subjectivity into the analysis.

## Updating Our Beliefs
A key step in the Bayesian method is to update our current beliefs with new evidence we come across. Consider the biased coin problem. We have a simple two-sided coin that potentially has some bias, meaning that each side may not be equally likely to occur. Say we suspect that flipping the coin will result in heads 60% of the time. We want to structure our beliefs about the coin in such a way that we can incorporate our prior knowledge and our current understanding of the problem. To do so, we use Bayes' Theorem:
$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$
We call the conditional probability $P(A|B)$ of the event we're interested in $A$ given the event $B$ our **posterior belief**. The **likelihood** is $P(B|A)$ and the **prior belief** is $P(A)$. $P(B)$ is the **normalization constant** to ensure that our probabilities are valid.

Note that we can use the Law of Total Probabilities to find our normalization constant once we know the likelihood and prior:
$$
P(B) = \sum_i P(B|A_i)P(A_i)
$$

Our coin flipping problem can be thought of as a series of Bernuoulli trials where the two possible outcomes are heads and tails. The most appropriate probability distribution then to model the likelihood would be the Binomial distribution:
$$
Pr(x|n, \theta) = {n \choose x} \theta^x (1-\theta)^{n-x}
$$

Here $n$ is the number of trials and $\theta$ is the parameter representing the bias of the coin. We want to place a prior belief on $\theta$. This belief will need to be a continuous probability distribution since $\theta$ can be any real number on $[0, 1]$. Often times when selecting a prior belief we should consider **conjugacy**, i.e. distributions that have a special relationship with the likelihood distribution. Since the Beta distribution is a **conjugate prior** for the Binomial distribution, it is a good choice for this problem.
$$
p(\theta| \alpha, \beta) = \frac{1}{B(\alpha, \beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}
$$
where $B(\alpha, \beta) = \int^1_0 \theta^{\alpha-1}(1-\theta)^{\beta-1} d\theta$.

Now with our likelihood and prior defined, we can formulate our posterior:
$$
p(\theta|x, n, \alpha, \beta) = \frac{Pr(x|n, \theta)p(\theta|\alpha, \beta)}{\int Pr(x|n, \theta)p(\theta|\alpha, \beta) d\theta}
$$

Note that we integrate over $\theta$ rather than sum since our prior is a continuous distribution.

The rest is simply algebra. After reducing the expression we arrive at the following result: $p(\theta|x, n, \alpha, \beta) = \text{Beta}(\alpha+x, \beta+n-x)$.

Bayesian models like the one above are intuitive and practical and frankly deserve a dedicated blog post. The key takeaway for me however is that once we understand the general Bayesian method, we can apply it to any problem where we are trying to find a solution under uncertainty.