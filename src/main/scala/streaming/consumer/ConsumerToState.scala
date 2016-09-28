package streaming.consumer

import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
  * Consumer which update the state
  *
  */

object ConsumerToState {


  def updateState(prices: Seq[(String, Double)], currentTotal: Option[(Int, Double)]) = {
    val currentRevenue = prices.map(_._2).sum
    val currentNumberPurchases = prices.size
    val state = currentTotal.getOrElse((0, 0.0))
    Some((currentNumberPurchases + state._1, currentRevenue + state._2))
  }

  def main(args: Array[String]) {

    val ssc = new StreamingContext("local[2]", "Streaming App with State", Seconds(10))
    // for stateful operations, we need to set a checkpoint location
    ssc.checkpoint("./data/tmp/sparkstreaming/")
    val stream = ssc.socketTextStream("localhost", 9999)

    // create stream of events from raw text elements
    val events = stream.map { record =>
      val event = record.split(",")
      (event(0), event(1), event(2).toDouble)
    }

    val users = events.map { case (user, product, price) => (user, (product, price)) }
    val revenuePerUser = users.updateStateByKey(updateState)
    revenuePerUser.print()

    // start the context
    ssc.start()
    ssc.awaitTermination()

  }

}
