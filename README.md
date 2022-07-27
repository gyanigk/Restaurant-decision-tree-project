# Restaurant-decision-tree-project
## Machine Learning Course project

# Introduction
- Decision trees are considered to be one of the most popular approaches for representing classifiers. Researchers from various disciplines such as statistics, machine learning, pattern recognition, and data mining considered the issue of growing a decision tree from available data.
- We used the Decision Tree for classification of numerical+textual Zomato restaurant dataset taken from Kaggle. The task was to implement the Decision Tree algorithms like ID3, CART, C4.5 with/without pruning to qualify the best result.  


# Dataset

- The basic idea of analyzing the Zomato dataset is to get a fair idea about the factors affecting the aggregate rating of each restaurant, establishment of different types of restaurant at different places, Bengaluru being one such city has more than 12,000 restaurants with restaurants serving dishes from all over the world.
- With such an overwhelming demand of restaurants it has therefore become important to study the demography of a location. What kind of a food is more popular in a locality. Do the entire locality loves vegetarian food. If yes then is that locality populated by a particular sect of people for eg. Jain, Marwaris, Gujaratis who are mostly vegetarian. These kind of analysis can be done using the data, by studying different factors.
----
##     [dataset_link](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
----
# Results

## Decision Tree (No Pruning)
<p align="center">
<img src=".\results\no_pruning_zomato.png">
</p>

----
## Decision Tree (Max_Depth=[1-49], highest 79.15% Acc)
<p align="center">
<img src=".\results\max_depth=50.png">
</p>

----
## Decision Tree at max_depth = 4 (Pre-pruning)
<p align="center">
<img src=".\results\depth5.png">
</p>

----

## Confusion Matrix
<p align="center">
<img src=".\results\finalcm.png">
</p>

----
- Best Accuracy at Max_Depth=32 ('entropy' criterion) is 0.79152.
<pre>
max_depth     accuracy
1            0.12792
2            0.182
3            0.1924
4            0.2068
5            0.22128
6            0.24464
7            0.25608
8            0.27456
9            0.30512
10            0.35104
11            0.40344
12            0.4724
13            0.53296
14            0.59128
15            0.64224
16            0.6788
17            0.71432
18            0.74608
19            0.7612
20            0.77208
21            0.77632
22            0.7812
23            0.78568
24            0.79024
25            0.79128
26            0.78632
27            0.78944
28            0.788
29            0.79016
30            0.7908
31            0.78856
32            0.79152 [Best]
33            0.79048
34            0.78776
35            0.79144
36            0.79064
37            0.78912
38            0.78992
39            0.79088
</pre>