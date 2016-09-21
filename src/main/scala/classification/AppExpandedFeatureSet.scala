package classification

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Expanded feature set
  */

object AppExpandedFeatureSet {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Expanded feature set")
      .set("spark.app.id", "expanded-feature-set")

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

      println(dataCategories.first.features)
      println(scaledDataCats.first.features)

      // 4. Train model on scaled data and evaluate metrics
      // number iterations for logistic regression and SVM
      val numIterations = 10

      val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
      // accuracy
      val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
        if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
      }.sum
      val lrAccuracyScaledCats = lrTotalCorrectScaledCats / records.count()
      // predictions
      val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
        (lrModelScaledCats.predict(point.features), point.label)
      }
      // metrics
      val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
      val lrPrCats = lrMetricsScaledCats.areaUnderPR
      val lrRocCats = lrMetricsScaledCats.areaUnderROC
      println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%")


    } finally {
      // Always stop Spark Context explicitly
      sc.stop
    }

  }
}
