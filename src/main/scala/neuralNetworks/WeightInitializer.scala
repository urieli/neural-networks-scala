package neuralNetworks

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Rand
import enumeratum._

import scala.math.sqrt

sealed trait WeightInitializer extends EnumEntry {
  def initMatrix(m: Int, n: Int): DenseMatrix[Double]
}

object WeightInitializer extends Enum[WeightInitializer] {
  val values = findValues

  case object DefaultWeightInitializer extends WeightInitializer {
    override def initMatrix(m: Int, n: Int): DenseMatrix[Double] = DenseMatrix.rand(m, n, Rand.gaussian(0, 1 / sqrt(n)))
  }

  case object LargeWeightInitializer extends WeightInitializer {
    override def initMatrix(m: Int, n: Int): DenseMatrix[Double] = DenseMatrix.rand(m, n, Rand.gaussian)
  }

}
