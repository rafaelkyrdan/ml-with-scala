package classification

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


/**
  * Classification
  * - logistic regression,
  * - SVM,
  * - naïve Bayes,
  * - decision tree.
  *
  */

object App {

  def main(args: Array[String]): Unit = {


    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Classification")
      .set("spark.app.id", "classification")

    val sc = new SparkContext(conf)

    try {

      // 1. Load and inspect the raw ratings dataset
      // Load the raw ratings data from a file
      val rawData = sc.textFile("./data/classification/train_noheader.tsv")
      val records = rawData.map(line => line.split("\t"))
      records.first.foreach(println)


      // 2. Cleaning, trimming and fixing the missing data
      val data = records.map { r =>
        val trimmed = r.map(_.replaceAll("\"", ""))
        val label = trimmed(r.size - 1).toInt
        val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      }
      data.cache
      val numData = data.count
      println(numData + " = count data")
      println("explore the first line in the cleaned data \n " + data.first)


      // 3. Convert dataset for naive Bayes
      // convert negative values to zeroes
      val nbData = records.map { r =>
        val trimmed = r.map(_.replaceAll("\"", ""))
        val label = trimmed(r.size - 1).toInt
        val features = trimmed.slice(4, r.size - 1)
          .map(d => if (d == "?") 0.0 else d.toDouble)
          .map(d => if (d < 0) 0.0 else d)
        LabeledPoint(label, Vectors.dense(features))
      }

      // 4. Train classification models
      // To compare the performance and use of different models,
      // we train a model using logistic regression, SVM, naïve Bayes, and a decision tree.
      // number iterations for logistic regression and SVM
      val numIterations = 10
      // the max depth tree for decision tree
      val maxTreeDepth = 5

      // 4.1 train logistic regression
      val lrModel = LogisticRegressionWithSGD.train(data, numIterations)

      // 4.2 train SVM model
      val svmModel = SVMWithSGD.train(data, numIterations)

      // note we use nbData here for the NaiveBayes model training
      // 4.3 train naive Bayes model
      val nbModel = NaiveBayes.train(nbData)

      // 4.4 train decision tree
      // set the mode, Algo, to Classification
      // set the Entropy impurity measure
      val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)


      // 5. Generating predictions

      // based on logistic regression
      val dataPoint = data.first
      // make prediction on a single data point
      val prediction = lrModel.predict(dataPoint.features)
      val trueLabel = dataPoint.label
      println("predictions based on logistic regression")
      println("compare truLabel " + trueLabel + " with prediction " + prediction)
      val predictions = lrModel.predict(data.map(lp => lp.features))
      println("predictions - ")
      predictions.take(5).foreach(println)

      // based on SVM model
      println("predictions based on SVM model")
      val prediction2 = svmModel.predict(dataPoint.features)
      println("compare truLabel " + trueLabel + " with prediction " + prediction2)
      val predictions2 = svmModel.predict(data.map(lp => lp.features))
      println("predictions - ")
      predictions2.take(5).foreach(println)

      // based on naive Bayes model
      println("predictions based on naive Bayes model")
      val prediction3 = nbModel.predict(dataPoint.features)
      println("compare truLabel " + trueLabel + " with prediction " + prediction3)
      val predictions3 = nbModel.predict(data.map(lp => lp.features))
      println("predictions - ")
      predictions3.take(5).foreach(println)

      // based on decision tree
      println("predictions based on decision tree")
      val prediction4 = dtModel.predict(dataPoint.features)
      println("compare truLabel " + trueLabel + " with prediction " + prediction4)
      val predictions4 = dtModel.predict(data.map(lp => lp.features))
      println("predictions - ")
      predictions4.take(5).foreach(println)

      // At this point compare 4 outputs
      // only decision tree gave the right output for single data input


      // 6. Evaluating the performance of classification models

      // calculating accuracy for logistic regression model
      val lrTotalCorrect = data.map { point =>
        if (lrModel.predict(point.features) == point.label) 1 else 0
      }.sum
      val lrAccuracy = lrTotalCorrect / data.count
      println("accuracy for logistic regression model " + lrAccuracy)

      // calculating accuracy for SVM model
      val svmTotalCorrect = data.map { point =>
        if (svmModel.predict(point.features) == point.label) 1 else 0
      }.sum
      val svmAccuracy = svmTotalCorrect / data.count
      println("accuracy for SVM model " + svmAccuracy)

      // calculating accuracy for naive Bayes model
      val nbTotalCorrect = data.map { point =>
        if (nbModel.predict(point.features) == point.label) 1 else 0
      }.sum
      val nbAccuracy = nbTotalCorrect / data.count
      println("accuracy for naive Bayes model " + nbAccuracy)

      // calculating accuracy for decision tree model
      val dtTotalCorrect = data.map { point =>
        val score = dtModel.predict(point.features)
        val predicted = if (score > 0.5) 1 else 0
        if (predicted == point.label) 1 else 0
      }.sum

      val dtAccuracy = dtTotalCorrect / data.count
      println("accuracy for decision tree model " + dtAccuracy)


      // 7. Compute area under PR and ROC curves for each model
      // generate binary classification metrics

      val metrics = Seq(lrModel, svmModel).map { model =>
        val scoreAndLabels = data.map { point =>
          (model.predict(point.features), point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
      }

      // again, we need to use the special nbData for the naive Bayes metrics
      val nbMetrics = Seq(nbModel).map{ model =>
        val scoreAndLabels = nbData.map { point =>
          val score = model.predict(point.features)
          (if (score > 0.5) 1.0 else 0.0, point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
      }

      // here we need to compute for decision tree separately since it does
      // not implement the ClassificationModel interface
      val dtMetrics = Seq(dtModel).map{ model =>
        val scoreAndLabels = data.map { point =>
          val score = model.predict(point.features)
          (if (score > 0.5) 1.0 else 0.0, point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
      }

      val allMetrics = metrics ++ nbMetrics ++ dtMetrics
      println("\nPR and ROC curves")
      allMetrics.foreach{ case (m, pr, roc) =>
        println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
      }

    } finally {
      // Always stop Spark Context explicitly
      sc.stop
    }
  }
}
