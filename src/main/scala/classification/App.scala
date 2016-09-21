package classification

import org.apache.spark.{SparkConf, SparkContext}

/**
  *
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


    } finally {
      // Always stop Spark Context explicitly
      sc.stop
    }
  }
}
