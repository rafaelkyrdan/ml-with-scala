# Clustering. K-means

## Training a clustering model
MLlib's K-means provides random and K-means || initialization, with the 
default being K-means ||. As both of these initialization methods are based on 
random selection to some extent, each model training run will return a 
different result. K-means does not generally converge to a global optimum model, 
so performing multiple training runs and selecting the most optimal model 
from these runs is a common practice. 



## Input data
The MovieLens 100k dataset is a set of 100,000 data points related to ratings 
given by a set of users to a set of movies. You can download the dataset by 
[link](http://files.grouplens.org/datasets/movielens/ml-100k.zip)
