package neuralNetworks

import org.scalatest.{ FlatSpec, Matchers }

class NeuralNetwork2Test extends FlatSpec with Matchers {
  "The NeuralNetwork2" should "serialize/deserialize" in {
    val network = new NeuralNetwork2(Vector(2, 3, 4))
    val json = network.save
    val network2 = NeuralNetwork2.load(json)
    network should be(network2)
  }
}
