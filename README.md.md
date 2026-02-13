This report details the experimentation done in the context of multi-arm bandits.

Background:

This experiment was run on the basis of three main datasets: news\_articles.csv, train\_users.csv and test\_users.csv. The news articles featured 144186 data points with 6 features: link, headline, category, short description, authors and date. The user CSVs had 2000 datapoints with 32 different features, some of which are: age, income, clicks, purchase\_amounts, etc.

Approach

1.1 Data Preprocessing

After examination and analysis of the data, it was established that the data suffered from missing values and categorical data had to be encoded. The data was preprocessed by removing the user\_id feature due to irrelevance, filling missing values with the median and categorical features were encoded using ordinal encoding.

1.2 Classification

For classification, a few different models were experimented with, but due to having the best performance, XGBOOST was finalised and is in the final code. The parameters were initialised as: n\_estimators = 200, max\_depth = 6, learning\_rate = 0.1 and through XG-boost, the validation accuracy had a value of 0.9025

1.3 Multi-arm contextual Bandits

In order to build an effective contextual Bandit system, 3 RL strategies were considered: Epsilon Greedy, UCB and Softmax. This section will further explore each strategy and the outcome.

1.3.1 Epsilon Greedy

The Epsilon-Greedy algorithm balances exploration and exploitation by:

*   Choosing the best-known action with probability **1 − ε**
    
*   Exploring randomly with probability **ε**
    

In this experiment, different values of epsilon were experimented with, \[0.05, 0.1, 0.2, 0.3\] across all the contexts. 

As is observable, as the epsilon value reduces, the final average reward increases however, so does the convergence time of the algorithm.  When e= 0.05, there is an average reward value of 6.6 gained, while e=0.3 only has an average reward of approximately 4.8, although it converges before all the other experiments.

Another observation in these experiments is the reward distribution per news Category. As seen, there is a high variance in rewards across all categories. The Crime category, while being the most spread out, also has the highest reward and mean reward compared to the other categories

1.3.2 Upper Confidence Bound (UCB)

UCB selects actions based on:

*   Q(a): estimated reward
    
*   N(a): number of times action was chosen
    
*   C: exploration coefficient
    

In the case of the experiment, the C values used were \[0.5, 1, 2\]

In UCB, when C=1 or 2, the average reward converged to 7, whereas it is around 5.5 when C= 0.5. Also, when C=0.5, there is an initial local maximum before convergence. When examining the reward distribution per category, the box plots are still a bit wide, although they have fewer negative outliers than E-Greedy. Once again, the crime category has a higher average and the highest reward value in the graph. 

1.3.3 Softmax

Softmax dictates the exploration-exploitation strategy through the tweaking of the parameter “τ”.

High “τ” causes exploration and random like-behaviour where as a low “τ” causes a higher exploitation behaviour. In this experiment, “τ” was initialised to 1, thus allowing for consistent exploration.

Like the UCB strategy, the average reward over time converges to 7; however, this convergence is faster than that observed in UCB. There are also no observable fluctuations in the graph. By examining the box plots, it is also visible that the reward distributions are tighter than the other strategies, although there are several outliers present under the Tech category.

1.4 Strategy Selection

From the experimental results detailed above, Softmax with τ = 1 achieves the fastest convergence and highest stable average reward (~7), outperforming both UCB and ε-greedy. Hence, Softmax was selected as the optimal strategy for the recommendation system and further coded.

1.5 Recommendation System

Based on the insights gained from all of the experiments, the recommendation system was built with an XGBoost Classifier and a SoftMax strategy was used to dictate the RL algorithm.