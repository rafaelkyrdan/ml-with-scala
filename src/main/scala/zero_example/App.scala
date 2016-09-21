package zero_example

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._


/**
  * A zero example with Apache Spark
  *
  */


object App {

  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Zero Example")
      .set("spark.app.id", "zero-example")

    val sc = new SparkContext(conf)

    try {

      // we take the raw data and convert it into a set of records(user, product, price)
      val data = sc.textFile("data/0/PurchaseHistory.csv")
        .map(line => line.split(","))
        .map(purchaseRecord => (purchaseRecord(0), purchaseRecord(1), purchaseRecord(2)))

      // count the number of purchases
      val numPurchases = data.count()

      // count how many unique users made purchases
      val uniqueUsers = data.map { case (user, product, price) => user }.distinct().count()

      // sum up our total revenue
      val totalRevenue = data.map { case (user, product, price) => price.toDouble }.sum()

      // find our most popular product
      val productsByPopularity = data
        .map { case (user, product, price) => (product, 1) }
        .reduceByKey(_ + _)
        .collect()
        .sortBy(-_._2)
      val mostPopular = productsByPopularity(0)

      // finally, print everything out
      println("Total purchases: " + numPurchases)
      println("Unique users: " + uniqueUsers)
      println("Total revenue: " + totalRevenue)
      println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))

    } finally {
      // Always stop Spark Context explicitly
      sc.stop
    }
  }
}
