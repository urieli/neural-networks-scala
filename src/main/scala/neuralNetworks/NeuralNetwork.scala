package neuralNetworks

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand
import org.slf4j.LoggerFactory

class NeuralNetwork(sizes: Seq[Int]) {
  private val log = LoggerFactory.getLogger(getClass)

  /** Biases for layers 1 to n-1.
   */
  var biases: Seq[DenseVector[Double]] = sizes.tail.map(size => DenseVector.rand(size, Rand.gaussian))

  /** Weights from layer i-1 to i, for layers 1 to n-1.
   *  Each m X n matrix has m rows for the current layer, and n columns for the weights input from the previous layer.
   */
  var weights: Seq[DenseMatrix[Double]] = sizes.tail.foldLeft((sizes.head, Seq.empty[DenseMatrix[Double]])) { case ((prev, weights), current) => (current, weights :+ DenseMatrix.rand(current, prev, Rand.gaussian)) }._2

  /** Network output if `a` is the input.
   *
   *  @param a a single input vector
   *  @return the output vector
   */
  def feedForward(a: DenseVector[Double]): DenseVector[Double] = {
    assert(a.length == sizes.head, "`a` must have same dimension as input layer")
    biases.zip(weights).foldLeft(a) { case (a, (b, w)) => sigmoid((w * a) + b) }
  }

  /** Train the neural network using mini-batch stochastic
   *  gradient descent.
   *
   *  @param trainingData  a list of tuples `(x, y)` representing the training inputs and the desired outputs.
   *  @param epochs        number of training epochs
   *  @param miniBatchSize size of each mini-batch
   *  @param ϵ             the learning rate
   *  @param testData      If provided, then the network will be evaluated against the test data after each
   *                      epoch, and partial progress printed out.  This is useful for
   *                      tracking progress, but slows things down substantially.
   */
  def SGD(
    trainingData: Seq[(DenseVector[Double], DenseVector[Double])],
    epochs: Int,
    miniBatchSize: Int,
    ϵ: Double,
    testData: Option[Seq[(DenseVector[Double], DenseVector[Double])]] = None): Unit = {
    for (i <- 0 to epochs) {
      val miniBatches = Rand.permutation(trainingData.length).draw().grouped(miniBatchSize)
        .map(indexes => indexes.map(i => trainingData(i)))
      miniBatches.foreach(miniBatch => this.updateMiniBatch(miniBatch, ϵ))
      testData match {
        case Some(testData) =>
          log.info(f"Epoch $i: ${this.evaluate(testData)} / ${testData.length}")
        case None =>
          log.info(f"Epoch $i complete")
      }
    }
  }

  /** Update the network's weights and biases by applying
   *  gradient descent using backpropagation to a single mini batch.
   *
   *  @param miniBatch a list of input/output tuples `(x, y)`
   *  @param ϵ         the learning rate
   */
  def updateMiniBatch(miniBatch: Seq[(DenseVector[Double], DenseVector[Double])], ϵ: Double): Unit = {
    val δb0 = biases.map(b => DenseVector.zeros[Double](b.length))
    val δw0 = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))
    val (δb, δw) = miniBatch.foldLeft((δb0, δw0)) {
      case ((δbPrev, δwPrev), (x, y)) =>
        val (δb, δw) = this.backPropagate(x, y)
        val δbNext = δbPrev.zip(δb).map { case (a, b) => a + b }
        val δwNext = δwPrev.zip(δw).map { case (a, b) => a + b }
        (δbNext, δwNext)
    }

    weights = weights.zip(δw).map { case (w, δw) => w - ((ϵ / miniBatch.length) * δw) }
    biases = biases.zip(δb).map { case (b, δb) => b - ((ϵ / miniBatch.length) * δb) }
  }

  /** Back propagate for input activation x and expected output y.
   *
   *  @param x the input vector
   *  @param y the expected output vector
   *  @return the gradient to the cost function, as layer-by-layer deltas to apply to the biases and weights respectively.
   */
  def backPropagate(x: DenseVector[Double], y: DenseVector[Double]): (Seq[DenseVector[Double]], Seq[DenseMatrix[Double]]) = {
    assert(x.length == sizes.head, "`x` must have same dimension as input layer")
    assert(y.length == sizes.last, "`y` must have same dimension as output layer")

    // feed forward
    val (_, as, zs) = biases.zip(weights).foldLeft((x, Seq(x), Seq.empty[DenseVector[Double]])) {
      case ((x, as, zs), (b, w)) =>
        val z = (w * x) + b
        val a = sigmoid(z)
        (a, as :+ a, zs :+ z)
    }

    if (log.isTraceEnabled) {
      log.trace(f"as (${as.length}): $as")
      log.trace(f"zs (${zs.length}): $zs")
    }

    val aRev = as.reverse
    val zRev = zs.reverse
    val wRev = weights.reverse

    // backwards pass
    val δ = costDerivative(aRev.head, y) *:* sigmoidPrime(zRev.head)
    val δw = δ.toDenseMatrix.t * aRev(1).toDenseMatrix

    if (log.isTraceEnabled) {
      log.trace(f"###output layer")
      log.trace(f"δ (${δ.length}): $δ")
      log.trace(f"aRev(1) (${aRev(1).length}): ${aRev(1)}")
      log.trace(f"δw (${δw.rows}X${δw.cols}): $δw")
    }

    val (_, δbs, δws, _, _, _) = (wRev.size - 1 to 1 by -1).foldLeft((δ, Seq(δ), Seq(δw), wRev, aRev.tail.tail, zRev.tail)) {
      case ((δ, δbs, δws, w :: wRev, a :: aRev, z :: zRev), l) =>
        val sp = sigmoidPrime(z)
        val δ_next = (w.t * δ) *:* sp
        val δw = δ_next.toDenseMatrix.t * a.toDenseMatrix
        if (log.isTraceEnabled) {
          log.trace(f"###from layer $l")
          log.trace(f"δ (${δ.length}): $δ")
          log.trace(f"w (${w.rows}X${w.cols}): $w")
          log.trace(f"z (${z.length}): $z")
          log.trace(f"sp (${sp.length}): $sp")
          log.trace(f"δ_next (${δ_next.length}): ${δ_next}")
          log.trace(f"a (${a.length}): ${a}")
          log.trace(f"δw (${δw.rows}X${δw.cols}): $δw")
        }

        (δ_next, δbs :+ δ_next, δws :+ δw, wRev, aRev, zRev)
    }

    (δbs.reverse, δws.reverse)
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

