package dimensionality_reduction

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, csvwrite}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Dimensionality reduction
  *
  */

object App {

  def main(args: Array[String]): Unit = {

    var conf = None: Option[SparkConf]
    var sc = None: Option[SparkContext]

    try {

      conf = Some(new SparkConf()
        .setMaster("local[2]")
        .setAppName("Dimensionality reduction")
        .set("spark.app.id", "dimensionality_reduction"))
      sc = Some(new SparkContext(conf.get))

      // load dir with files
      val path = "./data/lfw/*"
      val rdd = sc.get.wholeTextFiles(path)
      //      val first = rdd.first
      //      println(first)

      // extract just the file names
      val files = rdd.map { case (fileName, content) => fileName.replace("file:", "") }
      println(files.first)
      println(files.count)

      // load an image from a file
      def loadImageFromFile(path: String): BufferedImage = {
        ImageIO.read(new File(path))
      }

      val aePath = "./data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
      val aeImage = loadImageFromFile(aePath)

      println(aeImage)

      // convert an image to grayscale, and scale it to new width and height
      def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
        val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
        val g = bwImage.getGraphics()
        g.drawImage(image, 0, 0, width, height, null)
        g.dispose()
        bwImage
      }

      val grayImage = processImage(aeImage, 100, 100)

      println(grayImage)
      ImageIO.write(grayImage, "jpg", new File("./data/tmp/aeGray.jpg"))

      // extract the raw pixels from the image as a Double array
      def getPixelsFromImage(image: BufferedImage): Array[Double] = {
        val width = image.getWidth
        val height = image.getHeight
        val pixels = Array.ofDim[Double](width * height)
        image.getData.getPixels(0, 0, width, height, pixels)
        // pixels.map(p => p / 255.0) 		// optionally scale to [0, 1] domain
      }

      def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
        val raw = loadImageFromFile(path)
        val processed = processImage(raw, width, height)
        getPixelsFromImage(processed)
      }

      val pixels = files.map(f => extractPixels(f, 50, 50))
      println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))

      // create vectors
      val vectors = pixels.map(p => Vectors.dense(p))
      // the setName method createa a human-readable name that is displayed in the Spark Web UI
      vectors.setName("image-vectors")
      vectors.cache

      // normalize the vectors by subtracting the column means
      val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
      val scaledVectors = vectors.map(v => scaler.transform(v))
      // create distributed RowMatrix from vectors, and train PCA on it
      val matrix = new RowMatrix(scaledVectors)
      val K = 10
      val pc = matrix.computePrincipalComponents(K)

      // use Breeze to save the principal components as a CSV file
      val rows = pc.numRows
      val cols = pc.numCols
      println(rows, cols)
      val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
      csvwrite(new File("./data/tmp/pc.csv"), pcBreeze)

      // project the raw images to the K-dimensional space of the principla components
      val projected = matrix.multiply(pc)
      println(projected.numRows, projected.numCols)
      println(projected.rows.take(5).mkString("\n"))

      // relationship to SVD
      val svd = matrix.computeSVD(10, computeU = true)
      println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
      println(s"S dimension: (${svd.s.size}, )")
      println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")

      def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
        // note we ignore sign of the principal component / singular vector elements
        val bools = array1.zip(array2).map { case (v1, v2) => if (math.abs(math.abs(v1) - math.abs(v2)) > 1e-6) false else true }
        bools.fold(true)(_ & _)
      }
      // test the function
      println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0)))
      println(approxEqual(Array(1.0, 2.0, 3.0), Array(3.0, 2.0, 1.0)))
      println(approxEqual(svd.V.toArray, pc.toArray))

      // compare projections
      val breezeS = breeze.linalg.DenseVector(svd.s.toArray)
      val projectedSVD = svd.U.rows.map { v =>
        val breezeV = breeze.linalg.DenseVector(v.toArray)
        val multV = breezeV :* breezeS
        Vectors.dense(multV.data)
      }
      projected.rows.zip(projectedSVD).map { case (v1, v2) => approxEqual(v1.toArray, v2.toArray) }.filter(b => true).count

      // inspect singular values
      val sValues = (1 to 5).map { i => matrix.computeSVD(i, computeU = false).s }
      sValues.foreach(println)




    } finally {
      // Always stop Spark Context explicitly
      if (sc.isDefined) sc.get.stop
    }

  }

}
