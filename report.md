---
title: |
    Automated Ensemble Learning\
    \vfill
    \large{Project Report for Natural Computing}\
    Radboud University Nijmegen\
    November 1, 2017
abstract: |
    \newpage
    \thispagestyle{empty}
    \mbox{}
    \newpage
author:
    - Authored by Bob de Ruiter
links-as-notes: true

---

\newpage

# Introduction

In popular perception, the phrase "automated machine learning" might seem like a pleonasm. Machine learning is often seen as the field where computers are given the ability to automatically learn, without requiring explicit programming. However, this definition does not paint the entire picture: to successfully apply machine learning techniques, practitioners often first have to iteratively write a program in which the data is pre-processed, features are engineered and selected, and hyperparameters are optimized. In some cases, some form of prediction post-processing, such as thresholding, is also applied.

Besides programming the machine learning pipeline, the main other obstacle keeping laypeople from using machine learning is algorithm selection. Different machine learning models make different assumptions about the data, and some understanding of an algorithm is required to determine whether its usage is appropriate.

The field of automated machine learning strives to automate both pipeline creation and algorithm selection. In most implementations, many pipelines are iteratively created and evaluated, often using internal cross-validation, for this purpose. Even though these models could very well be complementary in their nature, many automated machine learning libraries, such as [TPOT](https://github.com/EpistasisLab/tpot) [@tpot] and [auto_ml](https://github.com/ClimbsRocks/auto_ml), only return the best-performing pipeline. By contrast, [auto-sklearn](https://github.com/automl/auto-sklearn) [@autosklearn] employs ensemble selection [@caruana2004ensemble] to create a relatively small ensemble from the large list of optimized models. An important thing to note about this method is that individual pipelines are optimized to perform well on their own, not as a complementary part of an ensemble. While this appears to work well combined with auto-sklearn's Bayesian optimization, this approach does not seem to translate to TPOT, which is based on Genetic Programming (GP). According to [Olson, one of the creators of the TPOT](https://github.com/EpistasisLab/tpot/issues/479), ensembling the final population of a TPOT run does not increase performance, nor does ensembling the performance-complexity pareto front. The reason stated by Olson is that TPOT pipelines are optimized to perform well by themselves. Considering that auto-sklearn does get a significant performance boost from the same type of ensembling, this problem is either specific to GP or the result of a different implementation detail.

Notwithstanding the generalizability of his theory, Olson proposes quite a few interesting alternative ways of automatically training an ensemble in the same discussion on TPOT's bug tracker, some of which were implemented. His initial proposal (hereafter referred to as Olson1), paraphrased below, was to implement ensembling as a boosting procedure.

> 1. Generate an initial population, P0
> 1. Evaluate each individual in P0 on K folds of the dataset as normal
> 1. Put the best pipeline from P0 into stacked ensemble M
> 1. Generate the next population, P1, using the fitness scores obtained in step 2
> 1. For each individual in P1, make a copy of stacked ensemble M and add the individual to it
> 1. Evaluate the copies on the dataset
> 1. Put the P1 pipeline corresponding to the highest stacked ensemble score into stacked ensemble M
> 1. Generate the next population using the fitness scores obtained in step 6, i.e. the stacked ensemble scores
> 1. Go back to step 5 until the stopping criterion occurs
> 1. Use stacked ensemble M for predictions

The obvious advantage of this is that the base estimators are trained to cooperate with other base estimators. In Olson's implementation, the main disadvantage was speed, especially in later generations. If TPOT's pipelines are generated in g generations using K-fold cross-validation, a population size of P, and an offspring size of O, this procedure will require `P + K x O x g!` pipeline evaluations, whereas the default TPOT algorithm only requires `P + K x O x g` evaluations.

A possible improvement of this method (henceforth referred to as Ruiter1), not discussed by Olson, would be to evaluate the models based on their individual internal cross-validation score except on every nth generation. On every nth generation, all pipelines, both the new offspring those pre-existing in the population, are evaluated in the context of the existing ensemble as described above. Only during these generations, the best pipeline is taken and put into the voting classifier. This method reduces the final ensemble size n times, decreasing the overall run time by a much larger factor. This method still weeds out pipelines that do not work well with the existing pipelines in the ensemble, but it also ensures that the pipelines added to the ensemble are grounded in the dataset, and are not just overfitted on the mistakes made by previous pipelines.

Olson, however, heads in an entirely new direction, where an ensemble is created from the latest population every generation. In a proposal inspired by the feature engineering wrapper FEW [@few] (later referred to as Olson2), he suggests to take the entire population (including the offspring) and stack their predictions in a feature matrix; to fit a linear meta-estimator on the feature matrix; and to use the linear meta-estimator coefficients as the fitness of each pipeline. This is clearly much faster than the implementation described above: even if predictions are not cached, the only decrease in speed comes from fitting the meta-estimator and from re-evaluating the pre-existing population every generation. One downside is that this algorithm cannot easily be implemented using TPOT's current public API. TPOT's architecture assumes that all pipelines can be evaluated independently and that pipelines that already existed in a previous generation do not have to be re-evaluated every generation. Other downsides are that this algorithm does not take the user-provided scoring function into account at all, and that the ensemble size is fixed to the population size plus the offspring size.

In this paper, I set out to implement Olson2, and I attempt to modify the proposed algorithm such that it takes the user-provided scoring function into account and the ensemble size can be limited. Even though this proposal, with some modifications, could be implemented in many automated machine learning libraries, I did end up implementing it in TPOT. First and foremost, this was because TPOT's genetic-programming-based approach was a better thematic fit for the Natural Computing course, even though I expected the changes to this part of TPOT to be minimal. Second, there is some evidence that TPOT is a top-performing automated machine learning library. On a benchmark on 10 bioinformatics datasets, TPOT performed best in a majority of the cases when compared to [Recipe](https://github.com/RecipeML/Recipe) and auto-sklearn, although the differences were fairly small [@recipe]. On a different comparative benchmark of auto-sklearn, [Auto-WEKA](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/), and hyperopt-sklearn on 16 datasets, auto-sklearn, in turn, achieved top performance on a majority of the datasets. For the other prominent automated machine learning libraries, including [auto_ml](https://github.com/ClimbsRocks/auto_ml), [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html), [devol](https://github.com/joeddav/devol), [MLBox](https://github.com/AxeldeRomblay/MLBox), and [Xcessiv](https://github.com/reiinakano/xcessiv), I was unable to find any high-quality comparative benchmarks.

Initially, I attempted to implement the Olson2 algorithm in a way that would allow it to be used in later versions of TPOT without requiring any form of merger, provided that the TPOT internals remain roughly the same. My initial approach was to add a `rescore` parameter to TPOT, which, if set, re-evaluated all individuals every generation, regardless of if they already existed in previous generations, and to subclass the TPOT base class, overriding the evaluation method. To this end, I submitted a [pull request](https://github.com/EpistasisLab/tpot/pull/704), but retaining the original behavior of TPOT was more difficult than anticipated, hence why the final implementation is a fork of TPOT. The subclassing approach to pipeline evaluation did make it into the final product.

# Implementation

The ensembling functionality was implemented in a fork of TPOT, which is [publicly available](https://github.com/bopjesvla/tpot).

Every generation, the training data is partitioned into multiple train-test splits according to a user-provided cross-validation object, which defaults to five-fold cross-validation.

For each fold, all pipelines in the population are fitted on the internal train set. The fitted pipelines are then used to predict the labels of the test set. Fitting and predicting are both performed in parallel. All predictions are stacked into a feature matrix. A copy of the user-provided meta-estimator was then trained to predict the true labels given the pipeline predictions.

The coefficients used for external predictions were computed by taking the average coefficient for each pipeline over all folds. Since the meta-estimators are linear models, this is equivalent to averaging the predictions of all meta-estimators.

The per-pipeline averages were also used as the pipelines' fitness scores.

something about failed filtering

# Evaluation

We compared 

In the experiment, a Lasso regression model was used as a meta-estimator. The intercept was frozen at zero and coefficients were forced to remain positive.

. In the experiment, we use the TPOT default,

Wilcoxon Signed-Ranks Test as multi

> Hypothesis 1a: the larger the number of instances in the dataset, the higher the performance of ensemble TPOT compared to regular TPOT.

> Hypothesis 1b: the larger the number of instances in the dataset, the higher the performance of regular TPOT compared to XGBoost.


1. Every generation, take the entire TPOT population and stack the outputs into a feature matrix
1. Fit a regularized (Lasso, preferably) linear model on the feature matrix
1. Use the linear model coefficients as the fitness of each pipeline

This 

...

Another reason machine learning is not entirely automated is that no single machine learning algorithm 

# Discussion

The fact that XGBoost consistently outperformed the TPOT-based approaches may be a sign of 

sped up by caching predictions

https://github.com/EpistasisLab/tpot/issues/335

ensembling pareto front: fit on same data, too similar https://github.com/EpistasisLab/tpot/issues/656

per-generation tree ensembling: too slow, since every node is an entire pipeline, so by generation 100 you are evaluating an ensemble of 100 pipelines for every member of the population

https://github.com/EpistasisLab/tpot/issues/479

It seemed to work fine in my tests, although it gets much, much slower as the generations pass because, e.g., by generation 100 every pipeline is being evaluated in a VotingClassifier with 99 other pipelines. The only reasonable solution seems to be to store the predictions of each "best" pipeline from every generation, and manually ensemble those predictions with the new predictions from the pipelines in the current generation.

only add to model when plateau?

use the scoring function to select for the best n models, ensemble those

# Bibliography
