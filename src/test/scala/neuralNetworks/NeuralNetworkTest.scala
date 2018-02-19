package neuralNetworks

import breeze.linalg.DenseVector
import org.scalatest.{ FlatSpec, Matchers }

class NeuralNetworkTest extends FlatSpec with Matchers {
  "The NeuralNetwork" should "feed forward" in {
    val network = new NeuralNetwork(Seq(2, 3, 4))
    println(f"biases: ${network.biases}")
    println(f"weights: ${network.weights}")
    val output = network.feedForward(DenseVector(1.0, 2.0))
    println(f"output: $output")
  }

  it should "back propagate" in {
    val network = new NeuralNetwork(Seq(2, 3, 4, 5, 6))
    println(f"biases: ${network.biases}")
    println(f"weights: ${network.weights}")
    val errors = network.backPropagate(DenseVector(1.0, 2.0), DenseVector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    println(f"errors: $errors")
  }
}
