package neuralNetworks

import breeze.linalg.{ DenseMatrix, DenseVector }
import spray.json.DefaultJsonProtocol

object NeuralNetworkProtocol extends DefaultJsonProtocol {
  import spray.json._

  implicit object DenseMatrixJsonFormat extends RootJsonFormat[DenseMatrix[Double]] {
    override def write(matrix: DenseMatrix[Double]): JsValue = JsObject(
      "m" -> JsNumber(matrix.rows),
      "n" -> JsNumber(matrix.cols),
      "data" -> JsArray(matrix.data.map(v => JsNumber(v)).toVector))

    override def read(json: JsValue): DenseMatrix[Double] = json match {
      case JsObject(fields) if fields.contains("data") =>
        new DenseMatrix[Double](
          fields.get("m").map(_.convertTo[Int]).getOrElse(0),
          fields.get("n").map(_.convertTo[Int]).getOrElse(0),
          fields.get("data").map(_.convertTo[Array[Double]]).getOrElse(Array.emptyDoubleArray))
      case _ => throw new DeserializationException("DenseMatrix expected")
    }
  }

  implicit object DenseVectorJsonFormat extends RootJsonFormat[DenseVector[Double]] {
    override def write(vector: DenseVector[Double]): JsValue = JsArray(vector.data.map(v => JsNumber(v)).toVector)

    override def read(json: JsValue): DenseVector[Double] = json match {
      case vs: JsArray => new DenseVector[Double](vs.convertTo[Array[Double]])
      case _           => throw new DeserializationException("DenseVector expected")
    }
  }

  implicit object NeuralNetwork2JsonFormat extends RootJsonFormat[NeuralNetwork2] {
    def write(net: NeuralNetwork2) = JsObject(
      "sizes" -> JsArray(net.sizes.map(s => JsNumber(s))),
      "weights" -> JsArray(net.weights.map(_.toJson)),
      "biases" -> JsArray(net.biases.map(_.toJson)),
      "cost" -> JsString(net.cost.entryName),
      "weightInitializer" -> JsString(net.weightInitializer.entryName))
    def read(json: JsValue): NeuralNetwork2 = json match {
      case JsObject(fields) if fields.contains("sizes") =>
        val sizes = fields.get("sizes").map(_.convertTo[Vector[Int]]).getOrElse(Vector.empty[Int])
        val cost = fields.get("cost").map(cost => CostFunction.withName(cost.convertTo[String])).getOrElse(CostFunction.CrossEntropyCost)
        val weightInitializer = fields.get("weightInitializer").map(w => WeightInitializer.withName(w.convertTo[String])).getOrElse(WeightInitializer.DefaultWeightInitializer)
        val net = new NeuralNetwork2(sizes, cost, weightInitializer)
        val weights = fields.get("weights").map {
          case JsArray(elements) => elements.map(w => w.convertTo[DenseMatrix[Double]])
          case _                 => throw new DeserializationException("DenseMatrix array expected")
        }.getOrElse(Vector.empty[DenseMatrix[Double]])
        val biases = fields.get("biases").map {
          case JsArray(elements) => elements.map(b => b.convertTo[DenseVector[Double]])
          case _                 => throw new DeserializationException("DenseVector array expected")
        }.getOrElse(Vector.empty[DenseVector[Double]])

        net.weights = weights
        net.biases = biases
        net
      case _ => throw new DeserializationException("NeuralNetwork2 expected")
    }
  }
}