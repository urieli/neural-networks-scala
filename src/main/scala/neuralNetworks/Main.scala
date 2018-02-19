package neuralNetworks

import org.rogach.scallop.ScallopConf

object Main {
  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val imagePath = opt[String](default = Some("data/mnist-train-images-50k.gz"))
    val labelPath = opt[String](default = Some("data/mnist-train-labels-50k.gz"))
    val epochs = opt[Int](default = Some(30))
    val miniBatchSize = opt[Int](default = Some(10))
    val learningRate = opt[Double](default = Some(3.0d), short = 'r')
    val testImagePath = opt[String](default = Some("data/mnist-test-images-10k.gz"), short = 'I')
    val testLabelPath = opt[String](default = Some("data/mnist-test-labels-10k.gz"), short = 'L')
    val trainSlice = opt[Int](short = 's')
    val testSlice = opt[Int](short = 'S')
    verify()
  }

  def main(args: Array[String]): Unit = {
    val conf = new Conf(args)

    val trainingData = MNISTLoader.load(conf.imagePath(), conf.labelPath())
    val trainingSlice = conf.trainSlice.toOption match {
      case None        => trainingData
      case Some(slice) => trainingData.slice(0, slice)
    }

    val testData = conf.testImagePath.toOption.map { case testImagePath => MNISTLoader.load(testImagePath, conf.testLabelPath()) }
    val testSlice = testData.map(testData => conf.testSlice.toOption match {
      case None        => testData
      case Some(slice) => testData.slice(0, slice)
    })

    val net = new NeuralNetwork(List(784, 30, 10))
    net.SGD(trainingSlice, conf.epochs(), conf.miniBatchSize(), conf.learningRate(), testSlice)
  }
}
