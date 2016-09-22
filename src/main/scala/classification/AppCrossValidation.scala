package classification

import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Cross validation example
  *
  */

object AppCrossValidation {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Cross Validation example")
      .set("spark.app.id", "cross-validation-example")

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

      // 3. Standardize the feature vectors
      val scalerCats = new StandardScaler(withMean = true, withStd = true)
        .fit(dataCategories.map(lp => lp.features))
      val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))
      scaledDataCats.cache

      val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
      val train = trainTestSplit(0)
      val test = trainTestSplit(1)
      val numIterations = 10

      // now we train our model using the 'train' dataset, and compute predictions on unseen 'test' data
      // in addition, we will evaluate the differing performance of regularization on training and test datasets
      val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
        val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
        createMetrics(s"$param L2 regularization parameter", test, model)
      }
      regResultsTest.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
      println("\n")

      // training set results
      val regResultsTrain = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
        val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
        createMetrics(s"$param L2 regularization parameter", train, model)
      }
      regResultsTrain.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }


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


}
