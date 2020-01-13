---
title: Introduction to Bayesian Machine Learning
date: "2020-01-06"
template: "post"
draft: true
slug: "/posts/bayesian-ml-recap"
category: "Machine Learning"
# tags:
#   - "Theory of Machine Learning"
description: "A tutorial on probabilistic learning"
---
Many elementary machine learning algorithms are taught from a Frequentist point of view where the primary goal is risk minimization achieved by fitting a set of parameters. Bayesian machine learning provides an alternative view that focuses more on probability. We want to maximize the probability of certain outcomes and minimize those of other outcomes. But even more so than this, Bayesian machine learning is often concerned with the problem of uncertainty: how do we know what we know and when do we know that we do not know? In a world where complex decision making is increasingly delegated to machines, it is critical that we give them the ability to recognize when they do not have enough information to sufficiently address a problem. In this post, I will survey the most important ideas in Bayesian machine learning and show where they can be most readily applied.

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

Bayesian models like the one above are intuitive and practical and frankly deserve a dedicated blog post. The key takeaway however is that once we understand the general Bayesian method, we can apply it to any problem where we are trying to find a solution under uncertainty.

## Hypothesis Testing, Credible Intervals, and Decision Theory
Now that we have estabilished the Bayesian framework, we can reconstruct important statistical instruments. Let's begin with hypothesis testing. In the Frequentists context, we may only ask if we can accept a null hypothesis $H_0$. Note that we cannot make use of degrees of belief in that statement. For a Frequentist, it is meaningless to ask what is the probability of the null hypothesis being true. For example, consider parameter estimation. We have some parameter $\theta$ and would like to test its value. We can do so by partitioning the parameter space $\Theta$ into two disjoint sets $\Theta_0$ and $\Theta_1$. Our hypothesis is that either $\theta \in \Theta_0$ or $\theta \in \Theta_1$. A Bayesian approach frames this question differently. Rather than asking if the parameter is in a set or not, we ask what is the probability that the parameter could be found on some interval. In essence we are asking what is the probability of a parameter value given the data we have $Pr(\theta|D)$. Since our posterior belief is just a probability measure, we may reason with it similarly. Thus if we wanted to compute the probability that $\theta = 0.3$, then we would simply compute the following:
$$
Pr(\theta = 0.3|D) = \int^{0.3}_{0.3}p(\theta|D)d\theta
$$
Here in lies the power and elegance of the the Bayesian method: posterior beliefs are simply probability measures and we may treat them as such. We can easily see how the above could be extended to support the function of confidence intervals. Suppose we wanted to know the probability of a parameter lying within a specific interval, for example $\theta \in [0.3, 0.5]$. We would simply compute $Pr(\theta \in [0.3, 0.5]) = \int^{0.5}_{0.3}p(\theta|D)d\theta$. In a Bayesian setting, this statistic is our **credible interval** which differs from the notion of confidence intervals in interesting and important ways.

Often in practice we are not computing statistics independent of any particular problem: we are trying to assist in decision making where risks and rewards need to be weighed before taking an action. Let us consider parameter estimation more fully now. We have some parameter $\theta$ for which we can produce an estimate $\hat{\theta}$. The risk of selecting a particular $\hat{\theta}$ can be measured with some loss function $L(\theta, \hat{\theta})$. Notice that decision theory begins to introduce the machinary of statistical learning. We can compute the **posterior expected loss** of estimate $\hat{\theta}$ to determine the quality of that estimate by using the expectation of the loss conditioned on the data:
$$
\mathbb{E}[L(\theta, \hat{\theta})|D] = \int L(\theta, \hat{\theta}) p(\theta|D) d \theta
$$
Minizing the risk is achieved by selecting the estimate the yields the lowest possible loss. This becomes the fundamental project of Bayesian machine learning and is achieved only by computing a posterior like the one above.