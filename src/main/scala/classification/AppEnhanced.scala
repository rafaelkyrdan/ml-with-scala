package classification

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.{SparkConf, SparkContext}


/**
  *
  *
  */

object AppEnhanced {

  def main(args: Array[String]): Unit = {


    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Classification Enhanced")
      .set("spark.app.id", "classification-enhanced")

    val sc = new SparkContext(conf)

    try {

      // 1. Load the raw ratings dataset
      val rawData = sc.textFile("./data/classification/train_noheader.tsv")
      val records = rawData.map(line => line.split("\t"))

      // 2. Cleaning, trimming and fixing the missing data
      import org.apache.spark.mllib.linalg.Vectors
      import org.apache.spark.mllib.regression.LabeledPoint
      val data = records.map { r =>
        val trimmed = r.map(_.replaceAll("\"", ""))
        val label = trimmed(r.size - 1).toInt
        val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      }

      data.cache
      val numData = data.count

      // 3. Feature standardization
      import org.apache.spark.mllib.linalg.distributed.RowMatrix
      val vectors = data.map(lp => lp.features)
      val matrix = new RowMatrix(vectors)
      val matrixSummary = matrix.computeColumnSummaryStatistics()

      println("mean " + matrixSummary.mean)
      println("min " + matrixSummary.min)
      println("max " + matrixSummary.max)
      println("variance " + matrixSummary.variance)
      println("num non zeros " + matrixSummary.numNonzeros)

      // scale the input features using MLlib's StandardScaler
      import org.apache.spark.mllib.feature.StandardScaler
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
      val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
      println("\ncompare the raw features with the scaled features")
      println(data.first.features)
      println(scaledData.first.features)
      println((0.789131 - 0.41225805299526636) / math.sqrt(0.1097424416755897))

      // 4. Re-train
      // number iterations for logistic regression and SVM
      val numIterations = 10
      // the max depth tree for decision tree
      val maxTreeDepth = 5

      // 4.1 train a logistic regression model on the scaled data
      val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)

      // 4.2 train a SVM model on the scaled data
      val svmModelScaled = SVMWithSGD.train(scaledData, numIterations)

      // 5. Evaluating the performance of classification models

      // calculating accuracy for logistic regression model
      val lrTotalCorrectScaled = scaledData.map { point =>
        if (lrModelScaled.predict(point.features) == point.label) 1 else 0
      }.sum
      val lrAccuracyScaled = lrTotalCorrectScaled / numData

      // calculating accuracy for SVM model
      val svmTotalCorrectScaled = scaledData.map { point =>
        if (svmModelScaled.predict(point.features) == point.label) 1 else 0
      }.sum
      val svmAccuracyScaled = svmTotalCorrectScaled / numData

      // 6. Predictions for each model
      // Pair(prediction, true value)

      val lrPredictionsVsTrue = scaledData.map { point =>
        (lrModelScaled.predict(point.features), point.label)
      }

      val svmPredictionsVsTrue = scaledData.map { point =>
        (svmModelScaled.predict(point.features), point.label)
      }


      // 7. Metrics
      val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
      val lrPr = lrMetricsScaled.areaUnderPR
      val lrRoc = lrMetricsScaled.areaUnderROC
      println(f"\n${lrModelScaled.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%")

      val svmMetricsScaled = new BinaryClassificationMetrics(svmPredictionsVsTrue)
      val svmPr = svmMetricsScaled.areaUnderPR
      val svmRoc = svmMetricsScaled.areaUnderROC
      println(f"\n${svmModelScaled.getClass.getSimpleName}\nAccuracy: ${svmAccuracyScaled * 100}%2.4f%%\nArea under PR: ${svmPr * 100.0}%2.4f%%\nArea under ROC: ${svmRoc * 100.0}%2.4f%%")


    } finally {
      // Always stop Spark Context explicitly
      sc.stop
    }
  }
}
