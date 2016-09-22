package classification

import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithSGD, NaiveBayes}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Example with tuned model parameters
  */

object AppWithTunedModelParams {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Tuned model params")
      .set("spark.app.id", "tuned-model-params")

    val sc = new SparkContext(conf)

    try {

      // 1. Load the raw ratings data from a file
      val rawData = sc.textFile("./data/classification/train_noheader.tsv")
      val records = rawData.map(line => line.split("\t"))

      // 2. Adding in the 'category' feature
      val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
      val numCategories = categories.size
      val dataCategories = records.map { r =>
        val trimmed = r.map(_.replaceAll("\"", ""))
        val label = trimmed(r.size - 1).toInt
        val categoryIdx = categories(r(3))
        val categoryFeatures = Array.ofDim[Double](numCategories)
        categoryFeatures(categoryIdx) = 1.0
        val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
        val features = categoryFeatures ++ otherFeatures
        LabeledPoint(label, Vectors.dense(features))
      }

      val data = records.map { r =>
        val trimmed = r.map(_.replaceAll("\"", ""))
        val label = trimmed(r.size - 1).toInt
        val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      }

      data.cache

      val dataNB = records.map { r =>
        val trimmed = r.map(_.replaceAll("\"", ""))
        val label = trimmed(r.size - 1).toInt
        val categoryIdx = categories(r(3))
        val categoryFeatures = Array.ofDim[Double](numCategories)
        categoryFeatures(categoryIdx) = 1.0
        LabeledPoint(label, Vectors.dense(categoryFeatures))
      }

      // 3. Standardize the feature vectors
      val scalerCats = new StandardScaler(withMean = true, withStd = true)
        .fit(dataCategories.map(lp => lp.features))
      val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))
      scaledDataCats.cache

      // num iterations
      // At this point check the output
      // the number of iterations has minor impact
      val iterResults = Seq(1, 5, 10, 50).map { param =>
        val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
        createMetrics(s"$param iterations", scaledDataCats, model)
      }
      iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
      println("\n")


      // step size
      val numIterations = 10
      val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
        val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
        createMetrics(s"$param step size", scaledDataCats, model)
      }
      stepResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
      println("\n")

      // regularization
      val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
        val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
        createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
      }
      regResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
      println("\n")

      // investigate tree depth impact for Entropy impurity
      val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
        val model = trainDTWithParams(data, param, Entropy)
        val scoreAndLabels = data.map { point =>
          val score = model.predict(point.features)
          (if (score > 0.5) 1.0 else 0.0, point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (s"$param tree depth", metrics.areaUnderROC)
      }
      dtResultsEntropy.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
      println("\n")

      // investigate tree depth impact for Gini impurity
      val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
        val model = trainDTWithParams(data, param, Gini)
        val scoreAndLabels = data.map { point =>
          val score = model.predict(point.features)
          (if (score > 0.5) 1.0 else 0.0, point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (s"$param tree depth", metrics.areaUnderROC)
      }
      dtResultsGini.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
      println("\n")

      // investigate Naive Bayes parameters
      val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
        val model = trainNBWithParams(dataNB, param)
        val scoreAndLabels = dataNB.map { point =>
          (model.predict(point.features), point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        (s"$param lambda", metrics.areaUnderROC)
      }
      nbResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
      println("\n")


    } finally {
      // Always stop Spark Context explicitly
      sc.stop
    }
  }


  // helper function to train a logistic regresson model
  def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    lr.run(input)
  }

  // helper function to create AUC metric
  def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC)
  }

  def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
  }

  def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
    val nb = new NaiveBayes
    nb.setLambda(lambda)
    nb.run(input)
  }

}
