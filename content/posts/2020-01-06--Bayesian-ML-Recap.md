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

Using what we know about the world to solve new problems is the foundation of the Bayesian method. In fact it is the very essence of Bayesian statistics and its prime point of difference with Frequentists methods. The use of a prior falls naturally out of the Bayesian definition of probability. 