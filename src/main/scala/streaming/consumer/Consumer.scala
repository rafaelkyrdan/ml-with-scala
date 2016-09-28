package streaming.consumer

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
  * Consumer which makes some analytics
  *
  */

object Consumer {

  def main(args: Array[String]): Unit = {

    val ssc = new StreamingContext("local[2]", "Streaming App with analytics", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)

    // create stream of events from raw text elements
    val events = stream.map { record =>
      val event = record.split(",")
      (event(0), event(1), event(2))
    }

    /*
      Since each batch is an RDD, we call forEachRDD on the DStream, and apply the usual RDD functions
     */

    events.foreachRDD { (rdd, time) =>
      val numPurchases = rdd.count()
      val uniqueUsers = rdd.map { case (user, _, _) => user }.distinct().count()
      val totalRevenue = rdd.map { case (_, _, price) => price.toDouble }.sum()
      val productsByPopularity = rdd
        .map { case (user, product, price) => (product, 1) }
        .reduceByKey(_ + _)
        .collect()
        .sortBy(-_._2)
      val mostPopular = productsByPopularity(0)

      val formatter = new SimpleDateFormat
      val dateStr = formatter.format(new Date(time.milliseconds))
      println(s"== Batch start time: $dateStr ==")
      println("Total purchases: " + numPurchases)
      println("Unique users: " + uniqueUsers)
      println("Total revenue: " + totalRevenue)
      println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))
    }

    // start the context
    ssc.start()
    ssc.awaitTermination()

  }
}
