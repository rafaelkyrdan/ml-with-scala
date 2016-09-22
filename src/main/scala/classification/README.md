# Classification

## List
- App.scala
- AppEnhanced.scala
- ExtantedSet

## First attempt
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

## Enhanced version
Enhanced version(`AppEnhanced.scala`) includes the classification models
based on logistic regression, SVM(without naïve Bayes, and a decision tree). The only
difference is we make standardization because models make inherent assumptions 
about the distribution or scale of input data.
`App.scala` and `AppEnhanced.scala` files.

## Expanded feature set
This example uses expanded feature set, we added a category.
We create a Map(index, category), 1-of-k encoding of this categorical feature.
We achieved a boost to 66 percent of accuracy by adding the category feature into our model.

## Tuning model parameters(`AppWithTunedModelParams.scala`)
Many machine learning methods are iterative in nature, converging to a solution 
(the optimal weight vector that minimizes the chosen loss function) over 
a number of iteration steps. From output we can see that the number of iterations has minor impact.

In SGD, the step size parameter controls how far in the direction of the 
steepest gradient the algorithm takes a step when updating the model weight 
vector after each training example. The output shows that increasing the step 
size too much can begin to negatively impact performance.

Regularization can help avoid over-fitting of a model to training data by 
effectively penalizing model complexity. This can be done by adding a term 
to the loss function that acts to increase the loss as a function of the model weight vector.
From the output we can see, at low levels of regularization, there is not 
much impact in model performance. However, as we increase regularization, 
we can see the impact of under-fitting on our model evaluation.

We set a parameter called `maxDepth`, which controls the maximum depth of the tree and, 
thus, the complexity of the model. Deeper trees result in more complex models 
that will be able to fit the data better, check the output.

The lambda parameter for naïve Bayes controls additive smoothing, which 
handles the case when a class and feature value do not occur together in the dataset.
From the  output we can see that lambda has no impact in this case, since 
it will not be a problem if the combination of feature and class label not 
occurring together in the dataset.

## Input data
This example uses a dataset from a competition on Kaggle. 
The dataset was provided by StumbleUpon, and the problem relates to classifying 
whether a given web page is ephemeral (that is, short lived and will cease 
being popular soon) or evergreen (that is, persistently popular) on their 
web content recommendation pages.
[link](http://www.kaggle.com/c/stumbleupon/data)

