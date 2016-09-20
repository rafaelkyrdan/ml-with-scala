# Recommendation engine

## Explanation
This example uses Spark's MLlib library to train the model.

To train the model we use next agruments:
1. rank - This refers to the number of factors in our ALS model, that is,
the number of hidden features in our low-rank approximation matrices;
2. iterations - refers to the number of iterations to run;
3. lambda - controls the regularization of our model. The higher the value of lambda,
the more is the regularization applied.

Evaluation metrics are:
1. Mean Squared Error
2. Mean average precision

The Mean Squared Error (MSE) is a direct measure of the reconstruction error 
of the user-item rating matrix. It is also the objective function being minimized 
in certain models, specifically many matrix-factorization techniques, including ALS. 
It is defined as the sum of the squared errors divided by the number of observations.
The squared error, in turn, is the square of the difference between the predicted 
rating for a given user-item pair and the actual rating. It is common to use 
the Root Mean Squared Error (RMSE), which is just the square root of the MSE metric. 
This is somewhat more interpretable, as it is in the same units as the underlying 
data (that is, the ratings in this case). It is equivalent to the standard 
deviation of the differences between the predicted and actual ratings.

Mean average precision at K (MAPK) is the mean of the average precision 
at K (APK) metric across all instances in the dataset. APK is a measure of 
the average relevance scores of a set of the top-K documents presented in 
response to a query. For each query instance, we will compare the set of 
top-K results with the set of actual relevant documents (that is, a ground 
truth set of relevant documents for the query). In the APK metric, the order 
of the result set matters, in that, the APK score would be higher if the 
result documents are both relevant and the relevant documents are presented higher in the results.

Follow the comments in the code, step by step.

## Input data
The MovieLens 100k dataset is a set of 100,000 data points related to ratings 
given by a set of users to a set of movies. You can download the dataset by 
[link](http://files.grouplens.org/datasets/movielens/ml-100k.zip)

