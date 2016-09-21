# Classification

## Explanation
In the `App.scala` we train classification models. 
To compare the performance and use of different models,
we train a model using logistic regression, SVM, naïve Bayes, and a decision tree.
Run this example and check the output of 4 models. Only model trained
based on decision tree gave the right output for single data input.

Evaluating the performance of classification models
The prediction error for binary classification is the number of training 
examples that are misclassified, divided by the total number of examples. 
Similarly, accuracy is the number of correctly classified examples divided by the total examples.

We use area under the PR curve and the area under the ROC curve to compare 
models with differing parameter settings and even compare completely different models.

As you can see from output all 4 models showed not a very good result in 
terms of binary classi cation performance.
 
Follow the comments in the code.

## Input data
This example uses a dataset from a competition on Kaggle. 
The dataset was provided by StumbleUpon, and the problem relates to classifying 
whether a given web page is ephemeral (that is, short lived and will cease 
being popular soon) or evergreen (that is, persistently popular) on their 
web content recommendation pages.
[link](http://www.kaggle.com/c/stumbleupon/data)

