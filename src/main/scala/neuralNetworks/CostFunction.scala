package neuralNetworks

import breeze.generic.{ MappingUFunc, UFunc }
import breeze.linalg.{ DenseVector, norm, sum }
import enumeratum._

import scala.math.{ log, pow }

sealed trait CostFunction extends EnumEntry {
  /** Return the cost associated with an output `a` and desired output `y`.
   *
   *  @param a
   *  @param y
   */
  def apply(a: DenseVector[Double], y: DenseVector[Double]): Double

  /** Return the error delta from the output layer.
   */
  def delta(z: DenseVector[Double], a: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double]
}

object CostFunction extends Enum[CostFunction] {
  val values = findValues

  case object CrossEntropyCost extends CostFunction {

    /** Logarithm, but returning 0 for an input of 0.
     */
    object logWith0 extends UFunc with MappingUFunc {

      implicit object implDouble extends Impl[Double, Double] {
        def apply(a: Double) = if (a == 0.0d) 0.0 else log(a)
      }

    }

    object oneMinus extends UFunc with MappingUFunc {

      implicit object implDouble extends Impl[Double, Double] {
        def apply(a: Double) = 1d - a
      }

    }

    /** Return the cost associated with an output `a` and desired output `y`.
     *
     *  @param a
     *  @param y
     */
    override def apply(a: DenseVector[Double], y: DenseVector[Double]): Double = sum(-y * logWith0(a) - oneMinus(y) * logWith0(oneMinus(a)))

    /** Return the error delta from the output layer.
     */
    override def delta(z: DenseVector[Double], a: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = a - y
  }

  case object QuadraticCost extends CostFunction {
    /** Return the cost associated with an output `a` and desired output `y`.
     *
     *  @param a
     *  @param y
     */
    override def apply(a: DenseVector[Double], y: DenseVector[Double]): Double = 0.5 * pow(norm(y - a), 2)

    /** Return the error delta from the output layer.
     */
    override def delta(z: DenseVector[Double], a: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = (a - y) * sigmoidPrime(z)
  }

}
