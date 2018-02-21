package neuralNetworks

import breeze.linalg.{ DenseMatrix, DenseVector }
import spray.json.DefaultJsonProtocol

object NeuralNetworkProtocol extends DefaultJsonProtocol {
  import spray.json._

  implicit object DenseMatrixJsonFormat extends RootJsonFormat[DenseMatrix[Double]] {
    override def write(matrix: DenseMatrix[Double]): JsValue = JsObject(
      "rows" -> JsNumber(matrix.rows),
      "cols" -> JsNumber(matrix.cols),
      "data" -> JsArray(matrix.data.map(v => JsNumber(v)).toVector))

    override def read(json: JsValue): DenseMatrix[Double] = json match {
      case JsObject(fields) if fields.contains("data") =>
        new DenseMatrix[Double](
          fields.get("rows").map(_.convertTo[Int]).getOrElse(0),
          fields.get("cols").map(_.convertTo[Int]).getOrElse(0),
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
      "weights" -> JsArray(net.weights.map(_.toJson)),
      "biases" -> JsArray(net.biases.map(_.toJson)),
      "cost" -> JsString(net.cost.entryName))
    def read(json: JsValue): NeuralNetwork2 = json match {
      case JsObject(fields) if fields.contains("weights") =>
        val weights = fields.get("weights").map {
          case JsArray(elements) => elements.map(w => w.convertTo[DenseMatrix[Double]])
          case _                 => throw new DeserializationException("DenseMatrix array expected")
        }.getOrElse(Vector.empty[DenseMatrix[Double]])
        val biases = fields.get("biases").map {
          case JsArray(elements) => elements.map(b => b.convertTo[DenseVector[Double]])
          case _                 => throw new DeserializationException("DenseVector array expected")
        }.getOrElse(Vector.empty[DenseVector[Double]])
        val cost = fields.get("cost").map(cost => CostFunction.withName(cost.convertTo[String])).getOrElse(CostFunction.CrossEntropyCost)

        new NeuralNetwork2(weights, biases, cost)
      case _ => throw new DeserializationException("NeuralNetwork2 expected")
    }
  }
}