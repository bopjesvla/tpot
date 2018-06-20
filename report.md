---
title: |
    Automated Ensemble Learning\
    \vfill
    \large{Project Report for Natural Computing}\
    Radboud University Nijmegen\
    June 17, 2018
abstract: |
    Automated machine learning library TPOT was modified to create an ensemble of machine learning pipelines optimized to perform well together. However, when benchmarked on 46 datasets, the unmodified version of TPOT outperformed the modified version. The unmodified version of TPOT also performed significantly better than the XGBoost baseline, while the modified version did not.
    \thispagestyle{empty}
    \newpage
    \thispagestyle{empty}
    \mbox{}
    \newpage
author:
    - Authored by Bob de Ruiter
links-as-notes: true
header-includes: |
    \usepackage{placeins}
---

\newpage

# Introduction

To the general public, the phrase "automated machine learning" might seem like a pleonasm. Machine learning is often seen as the field where computers are given the ability to automatically learn, without requiring explicit programming. However, this definition does not paint the entire picture: to successfully apply machine learning techniques, practitioners often first have to iteratively write a program in which the data is pre-processed, features are engineered and selected, and hyperparameters are optimized. In some cases, some form of prediction post-processing, such as thresholding, is also applied.

Besides programming the machine learning pipeline, the main other obstacle keeping laypeople from using machine learning is algorithm selection. Different machine learning models make different assumptions about the data, and some understanding of an algorithm is required to determine whether its usage is appropriate.

The field of automated machine learning strives to automate both pipeline creation and algorithm selection. In most implementations, many candidate pipelines are iteratively created, evaluated and modified, often using internal cross-validation. Even though these generated models could very well be complementary in their nature, many automated machine learning libraries, such as [TPOT](https://github.com/EpistasisLab/tpot) [@tpot] and [auto_ml](https://github.com/ClimbsRocks/auto_ml), only return the best-performing pipeline. By contrast, [auto-sklearn](https://github.com/automl/auto-sklearn) [@autosklearn] employs ensemble selection [@caruana2004ensemble] to create a relatively small ensemble from the large list of optimized models. An important thing to note about this method is that individual pipelines are optimized to perform well on their own, not as a complementary part of an ensemble. While this appears to work well combined with auto-sklearn's Bayesian optimization, this approach does not seem to translate to TPOT, which is based on Genetic Programming (GP). According to Olson, one of the creators of the TPOT, [ensembling the final population of a TPOT run does not increase performance, nor does ensembling the performance-complexity pareto front](https://github.com/EpistasisLab/tpot/issues/479). The reason stated by Olson is that TPOT pipelines are optimized to perform well by themselves. Considering that auto-sklearn does get a significant performance boost from the same type of ensembling, this problem is either specific to GP or the result of a different implementation detail.

Notwithstanding the generalizability of his theory, Olson proposes quite a few interesting alternative ways of automatically training an ensemble in the same discussion on TPOT's bug tracker, some of which were implemented. His initial proposal (later referred to as Olson1), paraphrased below, was to implement a boosting-like procedure.

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

The obvious advantage of this is that the base estimators are trained to cooperate with other base estimators. In Olson's implementation, the main disadvantage was speed, especially in later generations. If TPOT's pipelines are generated in g generations using an offspring size of O, a population size of P, and K-fold cross-validation, this procedure will require $P + K * O * g!$ pipeline evaluations, whereas the default TPOT algorithm only requires $P + K * O * g$ evaluations.

A possible improvement of Olson1, which is not discussed by Olson, would be to evaluate the models as normal, based on their individual internal cross-validation score, except on every nth generation. On every nth generation, all pipelines, both the new offspring those pre-existing in the population, are evaluated in the context of the existing ensemble as described above. Only during these generations, the best pipeline is taken and put into the voting classifier. This method reduces the final ensemble size n times, decreasing the overall run time by a much larger factor. This method still weeds out pipelines that do not work well with the existing pipelines in the ensemble, but it also ensures that the pipelines added to the ensemble are grounded in the dataset, and are not just overfitted on the mistakes made by previous pipelines.

Olson, however, heads in an entirely new direction, where an ensemble is created from the latest population every generation. In a proposal (later referred to as Olson2) inspired by the feature engineering wrapper FEW [@few], he suggests to take the entire population (including the offspring) and stack their predictions in a feature matrix; to fit a linear meta-estimator on the feature matrix; and to use the linear meta-estimator coefficients as the fitness of each pipeline. This is clearly much faster than the implementation described above: even if predictions are not cached, the only decrease in speed comes from fitting the meta-estimator and from re-evaluating the pre-existing population every generation. One downside is that this algorithm cannot easily be implemented using TPOT's current public API. TPOT's architecture assumes that all pipelines can be evaluated independently and that pipelines that already existed in a previous generation do not have to be re-evaluated every generation. Other downsides are that this algorithm does not take the user-provided scoring function into account at all, and that the ensemble size is fixed to the population size plus the offspring size.

In this paper, I set out to implement Olson2. Additionally, I attempt to modify the proposed algorithm such that the user-provided scoring function is taken into account and the ensemble size can be limited. Even though this proposal, with some modifications, could be implemented in many automated machine learning libraries, I did end up implementing it in TPOT. First and foremost, this was because TPOT's GP-based approach was a better thematic fit for the Natural Computing course, even though I expected the changes to this part of TPOT to be minimal. Second, there is some evidence that TPOT is a top-performing automated machine learning library. When benchmarked on 10 bioinformatics datasets, TPOT performed best in a majority of the cases when compared to [Recipe](https://github.com/RecipeML/Recipe) and auto-sklearn, although the differences were fairly small [@recipe]. On a different comparative benchmark of auto-sklearn, [Auto-WEKA](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/), and hyperopt-sklearn on 16 datasets, auto-sklearn, in turn, achieved top performance on a majority of the datasets. For the other prominent automated machine learning libraries, including [auto_ml](https://github.com/ClimbsRocks/auto_ml), [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html), [devol](https://github.com/joeddav/devol), [MLBox](https://github.com/AxeldeRomblay/MLBox), and [Xcessiv](https://github.com/reiinakano/xcessiv), I was unable to find any high-quality comparative benchmarks.

Initially, I attempted to implement the Olson2 algorithm in a way that would allow it to be used in later versions of TPOT without requiring any form of merger, provided that the TPOT internals remain roughly the same. My initial approach was to add a `rescore` parameter to TPOT, which, if set, re-evaluated all individuals every generation, regardless of if they already existed in previous generations, and to subclass the TPOT base class, overriding the evaluation method. To this end, I submitted a [pull request](https://github.com/EpistasisLab/tpot/pull/704), but retaining the original behavior of TPOT was more difficult than anticipated, hence why the final implementation is a fork of TPOT. The subclassing approach to pipeline evaluation did make it into the final product.

# Implementation

The ensembling functionality was implemented in a fork of TPOT, which is [publicly available](https://github.com/bopjesvla/tpot).

Every generation, the training data is partitioned into multiple train-test splits according to a user-provided cross-validation object, which defaults to five-fold cross-validation.

For each fold, all pipelines in the population are fitted on the internal train set. The fitted pipelines are then used to predict the labels of the test set. Fitting and predicting are both performed in parallel. All predictions are stacked into a feature matrix. A copy of the user-provided meta-estimator is then trained to predict the true labels given the pipeline predictions.

The coefficients used for external predictions were computed by taking the average meta-estimator coefficient for each pipeline over all folds. Since the meta-estimators are linear models, a prediction from a meta-estimator with averaged coefficients is equivalent to the average prediction of all meta-estimators.

The per-pipeline averages are also used as the pipelines' fitness scores.

To incorporate the user-provided scoring function and limit the ensemble size, a variant of the ensembling implementation was created where all pipelines are evaluated according to the scoring function, and only the best-scoring pipelines are added to the ensemble. Additional parameters were introduced for the number of estimators in the ensemble as well as the relative weight of the meta-estimator coefficient compared to the scoring function. The fitness for the pipelines added to the ensemble were computed by adding the weighted meta-estimator coefficient to the pipeline score. For the other pipelines, the individual score was used as the fitness.

During preliminary testing the variant implementation was found to behave erratically. In the majority of cases, the ensemble's predictions did not improve over time. Additionally, the implementation would sporadically and non-deterministically throw errors, which I was unable to trace down since TPOT silently swallows errors to prevent faulty pipelines from stopping the optimization process. As such, this implementation was not included in the final evaluation.

![Error on the validation set during a representative run of the TPOT variant using the scoring function (lower is better)](bad.png){width=400px}

# Benchmark

A comparative benchmark was run on 46 regression datasets: all Penn Machine Learning Benchmarks [@penn] apart from the Friedman datasets, the classification datasets, and the datasets with more than 100,000 samples. The unmodified version of TPOT, the ensemble version of TPOT, and XGBoost [@xgboost] were compared, along with a best-of-ensemble pipeline described below.

Each dataset was split into a train set, consisting of 75% of all data and a test set, consisting of 25% of all data.

Instances of both versions of TPOT were fit on each train set. Population size, offspring size and the number of generations were all set to 10. These parameters are all lower than recommended by the creators of TPOT; all values default to 100. Although practical considerations were the major reason for changing the number of generations, decreasing the population size and the offspring size was also thought to favor the ensemble version of TPOT, since the final meta-estimator has a number of base estimators equal to the population size added to the offspring size. To take an extreme, a meta-estimator with 200 complex base estimators is not likely to perform well on a 400-sample dataset. In any case, barring the possibility that the ensemble version of TPOT is initially slower to converge, performance differences at 10 generations are more likely than not directionally representative of performance differences at 100 generations.

To make sure TPOT did find a reasonable pipeline in 10 generations, an XGBoost regression model was also fit on each train set, without changing any of the hyperparameters. XGBoost was chosen as a baseline because it performs well on a wide variety of datasets and, like TPOT, handles mixed data well. Moreover, since TPOT is able to construct pipelines containing XGBoost regression models, a consistently superior performance from XGBoost would mean that either the pipeline space was not properly searched, or that the sample size cost of internal cross-validation proved too high.

Finally, the individual with the highest fitness score was tracked in each ensemble version of TPOT. This is the individual that dominated the ensemble the most in any generation. If the best-of-ensemble pipelines consistently outperform the ensembles, either stacking TPOT pipelines has no added value, or the ensemble size is too large.

A Lasso regression model was used as the meta-estimator for the ensemble version of TPOT. The meta-estimator's intercept was frozen to zero and the coefficients were forced to be positive. The unmodified version of TPOT used the Mean Squared Error as the scoring function.

Once a model was fitted, predictions on the test set were made and scored using the Mean Squared Error metric.

# Statistical Analysis

Six sign tests [@sign] were performed on the error scores of all pairs of predictors. A T-test was out of the question since its assumptions were clearly violated. A Wilcoxon Signed-Ranks Test, recommended by @multi for comparing classification performance across datasets, would also have been inappropriate since the absolute difference between mean squared errors can largely be attributed to the scale and distribution of the labels, which were not normalized as a part of the benchmark.

I hypothesized that the larger the number of instances in the dataset, the higher the performance of the ensemble variant of TPOT compared to regular TPOT would be, since more internal test data exists to calibrate the ensemble on, and the ensemble is less likely to overfit on large datasets. In a similar vein, I expected unmodified TPOT to excel more on large datasets when compared to XGBoost, since TPOT's decrease in effective sample size as a result of internal cross-validation hurts more on small datasets.

For each of these hypotheses, Spearman's rank order correlation between sample size and a binary variable signifying whether one method outperformed the other was computed.

# Results

Table \ref{signtest} shows the results of the sign tests performed on all pairs of predictors. A sign test is usually much weaker than a Wilcoxon Signed-Ranks Test, requiring one algorithm to outperform another on $ceil(N/2 + 1.96 * \sqrt{N/2})$ of $N$ datasets for a significant result. Since the models were tested on 45 datasets, this bar was set at 32.

\begin{table}
\begin{tabular}{|l|c|c|c|c|}
\hline
{} &  Ensemble &  BOE &  TPOT &  XGBoost \\
\hline
Ensemble &               0 &         27 &            13 &         27 \\
\hline
BOE &              18 &          0 &            10 &         21 \\
\hline
TPOT &              \textbf{32} &         \textbf{35} &             0 &         \textbf{34} \\
\hline
XGBoost      &              18 &         24 &            11 &          0 \\
\hline
\end{tabular}
\caption{The number of datasets the predictors in the left header performed better on than the predictors in the top header. Significant results are shown in bold. (TPOT = TPOT without any modifications, BOE = best-of-ensemble)}
\label{signtest}
\end{table}

Unmodified TPOT performed significantly better than all other methods. No other significant results were found.

\begin{table}
\begin{tabular}{lr}
\hline
{} &         Rank \\
\hline
Ensemble &  2.511111 \\
BOE      &  2.911111 \\
TPOT   &  1.755556 \\
XGBoost      &  2.822222 \\
\hline
\end{tabular}
\caption{Average rank of the different models (lower is better)}
\end{table}

No significant correlation between sample size and the relative performance of unmodified TPOT compared to XGBoost was found ($\rho$ = 0.289). Likewise, no significant correlation between sample size and the relative performance of the ensemble variant of TPOT compared to regular TPOT was found ($\rho$ = 0.115).

\begin{table}
\begin{tabular}{lrrr}
\hline
{} &   N &  TPOT > XGB &  Ensemble > TPOT \\
Sample size   &     &             &            \\
\hline
10 - 99       &   4 &    75.0\% &         25.0\% \\
100 - 999     &  21 &    61.9\% &         23.8\% \\
1000 - 9999   &  12 &    83.3\% &         25.0\% \\
10000 - 99999 &   8 &   100.0\% &         50.0\% \\
\hline
\end{tabular}
\caption{Relative performance of selected models. (N = number of datasets in the given range of sample sizes, TPOT > XGB = percentage of datasets on which TPOT performed better than XGBoost, Ensemble > TPOT = percentage of datasets on which ensemble TPOT performed better than unmodified TPOT)}
\end{table}

# Discussion

The fact that unmodified TPOT performed significantly better than XGBoost after only 110 pipeline evaluations shows that TPOT is a solid out-of-the-box regression model. Since TPOT writes the top machine learning pipeline to a human-readable Python file, it is, at the time of writing, probably the most transparent automated machine learning library; it was originally marketed a data science assistant, and it can still be used that way.

While TPOT's usability is great, the fact that modifications to TPOT code can silently fail makes development quite difficult. Since the code is littered with try-catch statements, I was unable to figure out where exactly the silencing occurred. It was unfortunate that my variant implementation of the Olson2 algorithm did not work, but if I were to choose again between implementing the Olson1 variant and implementing the Olson2 variant, I would pick the former. A large reason that human data scientists are excited about ensembling automated machine learning pipelines is that it works well when human data scientists do it. Therefore, it seems sensible to make the process of generating base estimators for the ensemble at least somewhat similar to the process humans create their base models. In my experience, human data scientists typically do not evaluate their base models in the context of a stacked ensemble right from the start of the process of pipeline modeling, possibly because optimizing a stacked ensemble in such a way is prone to overfitting. The more common workflow is to optimize the base estimators independently, perhaps sporadically checking if they work well together. The Olson1 variant captures this approach in spirit.

# Bibliography
