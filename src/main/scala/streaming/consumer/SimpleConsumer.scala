package streaming.consumer

import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
  * Consumer
  *
  */

object SimpleConsumer {

  def main(args: Array[String]): Unit = {

    val ssc = new StreamingContext("local[2]", "Streaming app", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)

    // here we simply print out the first few elements of each batch
    stream.print()
    ssc.start()
    ssc.awaitTermination()

  }
}
