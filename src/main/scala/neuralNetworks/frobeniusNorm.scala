package neuralNetworks

import breeze.generic.UFunc
import breeze.linalg.support.CanTraverseValues
import breeze.linalg.support.CanTraverseValues.ValuesVisitor

import scala.math.sqrt

/** Frobenius norm for a scalar, vector or matrix.
 */
object frobeniusNorm extends UFunc {
  implicit def sumFromTraverseDoubles[T](implicit traverse: CanTraverseValues[T, Double]): Impl[T, Double] = {
    new Impl[T, Double] {
      def apply(t: T): Double = {
        var sum = 0.0
        traverse.traverse(t, new ValuesVisitor[Double] {
          def visit(a: Double) = { sum += a * a }
          def zeros(count: Int, zeroValue: Double) {}
        })

        sqrt(sum)
      }
    }
  }
}