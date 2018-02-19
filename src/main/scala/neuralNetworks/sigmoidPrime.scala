package neuralNetworks

import breeze.generic.{ MappingUFunc, UFunc }
import breeze.numerics.sigmoid

/** Derivative of sigmoid function.
 */
object sigmoidPrime extends UFunc with MappingUFunc {
  implicit object implDouble extends Impl[Double, Double] {
    def apply(a: Double) = sigmoid(a) * (1 - sigmoid(a))
  }
}