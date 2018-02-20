package neuralNetworks

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand
import org.rogach.scallop.ScallopConf
import org.slf4j.LoggerFactory

class NeuralNetwork(sizes: Seq[Int]) {
  private val log = LoggerFactory.getLogger(getClass)

  val numLayers = sizes.size

  /** Biases for layers 1 to n-1.
   */
  var biases: Seq[DenseVector[Double]] = sizes.tail.map(size => DenseVector.rand(size, Rand.gaussian))

  /** Weights from layer i-1 to i, for layers 1 to n-1.
   *  Each m X n matrix has m rows for the current layer, and n columns for the weights input from the previous layer.
   */
  var weights: Seq[DenseMatrix[Double]] =
    sizes.init.zip(sizes.tail).map { case (prev, current) => DenseMatrix.rand(current, prev, Rand.gaussian) }

  /** Network output if `a` is the input.
   *
   *  @param a a single input vector
   *  @return the output vector
   */
  def feedForward(a: DenseVector[Double]): DenseVector[Double] = {
    assert(a.size == sizes.head, "`a` must have same dimension as input layer")
    biases.zip(weights).foldLeft(a) { case (a, (b, w)) => sigmoid((w * a) + b) }
  }

  /** Train the neural network using mini-batch stochastic
   *  gradient descent.
   *
   *  @param trainingData  a list of tuples `(x, y)` representing the training inputs and the desired outputs.
   *  @param epochs        number of training epochs
   *  @param miniBatchSize size of each mini-batch
   *  @param η             the learning rate
   *  @param testData      If provided, then the network will be evaluated against the test data after each
   *                      epoch, and partial progress printed out.  This is useful for
   *                      tracking progress, but slows things down substantially.
   */
  def SGD(
    trainingData: Seq[(DenseVector[Double], DenseVector[Double])],
    epochs: Int,
    miniBatchSize: Int,
    η: Double,
    testData: Option[Seq[(DenseVector[Double], DenseVector[Double])]] = None): Unit = {
    for (i <- 0 to epochs) {
      val miniBatches = Rand.permutation(trainingData.size).draw().grouped(miniBatchSize)
        .map(indexes => indexes.map(i => trainingData(i)))
      miniBatches.foreach(miniBatch => this.updateMiniBatch(miniBatch, η))
      testData match {
        case Some(testData) =>
          log.info(f"Epoch $i: ${this.evaluate(testData)} / ${testData.size}")
        case None =>
          log.info(f"Epoch $i complete")
      }
    }
  }

  /** Update the network's weights and biases by applying
   *  gradient descent using backpropagation to a single mini batch.
   *
   *  @param miniBatch a list of input/output tuples `(x, y)`
   *  @param η         the learning rate
   */
  def updateMiniBatch(miniBatch: Seq[(DenseVector[Double], DenseVector[Double])], η: Double): Unit = {
    val δb0 = biases.map(b => DenseVector.zeros[Double](b.size))
    val δw0 = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))
    val (δb, δw) = miniBatch.foldLeft((δb0, δw0)) {
      case ((δbPrev, δwPrev), (x, y)) =>
        val (δb, δw) = this.backPropagate(x, y)
        val δbNext = δbPrev.zip(δb).map { case (a, b) => a + b }
        val δwNext = δwPrev.zip(δw).map { case (a, b) => a + b }
        (δbNext, δwNext)
    }

    weights = weights.zip(δw).map { case (w, δw) => w - ((η / miniBatch.size) * δw) }
    biases = biases.zip(δb).map { case (b, δb) => b - ((η / miniBatch.size) * δb) }
  }

  /** Back propagate for input activation x and expected output y.
   *
   *  @param x the input vector
   *  @param y the expected output vector
   *  @return the gradient to the cost function, as layer-by-layer deltas to apply to the biases and weights respectively.
   */
  def backPropagate(x: DenseVector[Double], y: DenseVector[Double]): (Array[DenseVector[Double]], Array[DenseMatrix[Double]]) = {
    assert(x.size == sizes.head, "`x` must have same dimension as input layer")
    assert(y.size == sizes.last, "`y` must have same dimension as output layer")

    val δbs = new Array[DenseVector[Double]](biases.size)
    val δws = new Array[DenseMatrix[Double]](weights.size)

    // feed forward
    val (_, as, zs) = biases.zip(weights).foldLeft((x, Seq(x), Seq.empty[DenseVector[Double]])) {
      case ((x, as, zs), (b, w)) =>
        val z = (w * x) + b
        val a = sigmoid(z)
        (a, as :+ a, zs :+ z)
    }

    // backwards pass
    var δ = costDerivative(as.last, y) *:* sigmoidPrime(zs.last)
    δbs(δbs.size - 1) = δ
    δws(δws.size - 1) = δ.toDenseMatrix.t * as(as.size - 2).toDenseMatrix

    for (l <- 2 until numLayers) {
      val z = zs(zs.size - l)
      val sp = sigmoidPrime(z)
      δ = (weights(weights.size - l + 1).t * δ) *:* sp
      val δw = δ.toDenseMatrix.t * as(as.size - l - 1).toDenseMatrix
      δbs(δbs.size - l) = δ
      δws(δws.size - l) = δw
    }

    (δbs, δws)
  }

  /** Return the number of test inputs for which the neural
   *  network outputs the correct result. Note that the neural
   *  network's output is assumed to be the index of whichever
   *  neuron in the final layer has the highest activation.
   *
   *  @param testData a list of tuples `(x, y)` representing the test inputs and the expected outputs.
   *  @return an int representing the number of correctly guessed outputs.
   */
  def evaluate(testData: Seq[(DenseVector[Double], DenseVector[Double])]): Int = {
    testData.count {
      case (x, y) =>
        argmax(this.feedForward(x)) == argmax(y)
    }
  }

  /** Return the vector of partial cost derivatives for the output activations.
   */
  def costDerivative(outputActivations: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = outputActivations - y

}

object NeuralNetwork {

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
