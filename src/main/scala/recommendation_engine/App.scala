package recommendation_engine

import org.apache.spark.SparkContext

/**
  *
  *
  */

object App {

  def main(args: Array[String]): Unit = {


    val sc = new SparkContext("local[2]", "Zero Example")

    try {

      // Load the raw ratings data from a file
      val rawData = sc.textFile("./data/ml-100k/u.data")
      println("raw data ", rawData.first()) // 196	242	3	881250949

      // Extract the user id, movie id and rating only from the data set
      val rawRatings = rawData.map(_.split("\t").take(3))
      println("extracted data")
      rawRatings.first().foreach(println) // 196 242 3

      import org.apache.spark.mllib.recommendation.ALS
      import org.apache.spark.mllib.recommendation.Rating

      // Construct the RDD of Rating objects
      val ratings = rawRatings.map {
        case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble)
      }
      println("rating ", ratings.first())

      // Train the ALS model with rank=50, iterations=10, lambda=0.01
      val model = ALS.train(ratings, 50, 10, 0.01)
      // Inspect the user factors
      println("inspect user features ", model.userFeatures)
      // Count user factors and force computation
      println("count user features ", model.userFeatures.count)
      println("count product features ", model.productFeatures.count)

      // Make a prediction for a single user and movie pair
      val userId = 789
      val predictedRating = model.predict(789, 123)

      // Make predictions for a single user across all movies
      val K = 10
      val topKRecs = model.recommendProducts(userId, K)
      println(topKRecs.mkString("\n"))

      // Load movie titles to inspect the recommendations
      val movies = sc.textFile("./data/ml-100k/u.item")
      val titles = movies.map(line => line.split("\\|").take(2))
        .map(array => (array(0).toInt, array(1))).collectAsMap()
      println("\n")
      println("titles ", titles(123) )

      val moviesForUser = ratings.keyBy(_.user).lookup(userId)
      println(moviesForUser.size)
      println("\n")
      println("movies rated by given user \n")
      moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating)).foreach(println)
      println("\n")
      println("top k movies recommended for user \n")
      topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)

      // Compute item-to-item similarities between an item and the other items
      import org.jblas.DoubleMatrix
      val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))

      // Compute the cosine similarity between two vectors
      def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
        vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
      }

      val itemId = 567
      val itemFactor = model.productFeatures.lookup(itemId).head
      val itemVector = new DoubleMatrix(itemFactor)
      cosineSimilarity(itemVector, itemVector)

      val sims = model.productFeatures.map{ case (id, factor) =>
        val factorVector = new DoubleMatrix(factor)
        val sim = cosineSimilarity(factorVector, itemVector)
        (id, sim)
      }
      val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
      println("\n similarities")
      println(sortedSims.mkString("\n"))


      // We can check the movie title of our chosen movie and the most similar movies to it
      val sortedSims2 = sims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
      println("\n sims with titles")
      println(sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim) }.mkString("\n"))


      // Compute squared error between a predicted and actual rating
      // We'll take the first rating for our example user 789
      val actualRating = moviesForUser.take(1)(0)
      val predictedRatingForGivenUser = model.predict(userId, actualRating.product)
      val squaredError = math.pow(predictedRatingForGivenUser - actualRating.rating, 2.0)
      println(predictedRatingForGivenUser, " predictedRatingForGivenUser")
      println("\n")
      println(squaredError, " squaredError")
      println("\n")

      // Compute Mean Squared Error across the dataset
      // Below code is taken from the Apache Spark MLlib guide
      // at: http://spark.apache.org/docs/latest/mllib-guide.html#collaborative-filtering-1

      val usersProducts = ratings.map{ case Rating(user, product, rating)  => (user, product)}
      val predictions = model.predict(usersProducts).map{
        case Rating(user, product, rating) => ((user, product), rating)
      }

      val ratingsAndPredictions = ratings.map{
        case Rating(user, product, rating) => ((user, product), rating)
      }.join(predictions)

      val MSE = ratingsAndPredictions.map{
        case ((user, product), (actual, predicted)) =>  math.pow((actual - predicted), 2)
      }.reduce(_ + _) / ratingsAndPredictions.count
      println("Mean Squared Error = " + MSE)

      val RMSE = math.sqrt(MSE)
      println("Root Mean Squared Error = " + RMSE)

      // Compute Mean Average Precision at K
      // Function to compute average precision given a set of actual and predicted ratings
      // Code for this function is based on: https://github.com/benhamner/Metrics
      def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
        val predK = predicted.take(k)
        var score = 0.0
        var numHits = 0.0
        for ((p, i) <- predK.zipWithIndex) {
          if (actual.contains(p)) {
            numHits += 1.0
            score += numHits / (i.toDouble + 1.0)
          }
        }
        if (actual.isEmpty) {
          1.0
        } else {
          score / scala.math.min(actual.size, k).toDouble
        }
      }

      val actualMovies = moviesForUser.map(_.product)
      val predictedMovies = topKRecs.map(_.product)
      val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
      println("Average Precision at K = " +  apk10)

      // Compute recommendations for all users
      val itemFactors = model.productFeatures.map { case (id, factor) => factor }.collect()
      val itemMatrix = new DoubleMatrix(itemFactors)
      println("size of matrix")
      println(itemMatrix.rows, itemMatrix.columns)
      val imBroadcast = sc.broadcast(itemMatrix)

      // compute recommendations for each user, and sort them in order of score so that the actual input
      // for the APK computation will be correct

      val allRecs = model.userFeatures.map{ case (userId, array) =>
        val userVector = new DoubleMatrix(array)
        val scores = imBroadcast.value.mmul(userVector)
        val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
        val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
        (userId, recommendedIds)
      }

      // next get all the movie ids per user, grouped by user id
      val userMovies = ratings.map{ case Rating(user, product, rating) => (user, product) }.groupBy(_._1)

      // finally, compute the APK for each user, and average them to find MAPK
      val MAPK = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecisionK(actual, predicted, K)
      }.reduce(_ + _) / allRecs.count
      println("Mean Average Precision at K = " + MAPK)


      // Using MLlib built-in metrics
      import org.apache.spark.mllib.evaluation.RegressionMetrics
      val predictedAndTrue = ratingsAndPredictions.map { case ((user, product), (actual, predicted)) => (actual, predicted) }
      val regressionMetrics = new RegressionMetrics(predictedAndTrue)
      println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
      println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)


      import org.apache.spark.mllib.evaluation.RankingMetrics
      val predictedAndTrueForRanking = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2)
        (predicted.toArray, actual.toArray)
      }
      val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
      println("Mean Average Precision = " + rankingMetrics.meanAveragePrecision)


      // Compare to our implementation, using K = 2000 to approximate the overall MAP
      val MAPK2000 = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecisionK(actual, predicted, 2000)
      }.reduce(_ + _) / allRecs.count
      println("Mean Average Precision, MAPK2000 = " + MAPK2000)








    } finally {
      // Always stop Spark Context explicitly
      sc.stop
    }
  }
}
