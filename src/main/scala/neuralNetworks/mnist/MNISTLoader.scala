package neuralNetworks.mnist

import java.io.{ DataInputStream, FileInputStream }

import breeze.linalg.DenseVector
import org.apache.commons.compress.compressors.CompressorStreamFactory
import org.rogach.scallop.ScallopConf

/** MNIST Image file format:
 *  <pre>
 *  [offset] [type]          [value]          [description]
 *  0000     32 bit integer  0x00000803(2051) magic number
 *  0004     32 bit integer  60000            number of images
 *  0008     32 bit integer  28               number of rows
 *  0012     32 bit integer  28               number of columns
 *  0016     unsigned byte   ??               pixel
 *  0017     unsigned byte   ??               pixel
 *  ........
 *  xxxx     unsigned byte   ??               pixel
 *  </pre>
 *  Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
 *
 *  MNIST Label file format:
 *  <pre>
 *  0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 *  0004     32 bit integer  60000            number of items
 *  0008     unsigned byte   ??               label
 *  0009     unsigned byte   ??               label
 *  ........
 *  xxxx     unsigned byte   ??               label
 *  </pre>
 *  The labels values are 0 to 9.
 */
object MNISTLoader {
  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val imagePath = opt[String](default = Some("data/mnist-train-images-50k.gz"))
    val labelPath = opt[String](default = Some("data/mnist-train-labels-50k.gz"))
    val length = opt[Int](required = true, short = 'n')
    verify()
  }

  def main(args: Array[String]): Unit = {
    val conf = new Conf(args)

    val data = MNISTLoader.load(conf.imagePath(), conf.labelPath())
    data.slice(0, conf.length()).zipWithIndex.foreach { case ((image, label), i) => println(f"image $i: $image, $label") }
  }

  /** Loads corresponding MNIST image and label files
   *
   *  @param imagePath the path of the image file to read
   *  @param labelPath the path of the label file to read
   *  @return a tuple of dense vectors `(x,y)`, where `x` is the image pixels converted to values from 0.0 to 1.0, and
   *         `y` is a vector with a 1.0 at the position of the correct digit, and 0.0 elsewhere.
   */
  def load(imagePath: String, labelPath: String): Seq[(DenseVector[Double], DenseVector[Double])] = {
    val imageInputStream =
      new DataInputStream(
        if (imagePath.endsWith(".gz")) {
          new CompressorStreamFactory().createCompressorInputStream(
            CompressorStreamFactory.GZIP,
            new FileInputStream(imagePath))
        } else {
          new FileInputStream(imagePath)
        })

    try {
      imageInputStream.skip(4)
      val imageCount = imageInputStream.readInt()
      val rowCount = imageInputStream.readInt()
      val columnCount = imageInputStream.readInt()
      assert(rowCount == 28, f"unexpected rowCount: $rowCount")
      assert(columnCount == 28, f"unexpected columnCount: $columnCount")

      val imageBytes = new Array[Byte](rowCount * columnCount)

      val images = (1 to imageCount).map { i =>
        val r = imageInputStream.read(imageBytes)
        assert(r == rowCount * columnCount, f"could not read image bytes for image $i, read $r")
        val ints = imageBytes.map(b => b.toInt).map(i => if (i < 0) 256 + i else i)
        new DenseVector(ints.map(i => i.toDouble / 256d))
      }

      imageInputStream.close()

      val labelInputStream =
        new DataInputStream(
          if (labelPath.endsWith(".gz")) {
            new CompressorStreamFactory().createCompressorInputStream(
              CompressorStreamFactory.GZIP,
              new FileInputStream(labelPath))
          } else {
            new FileInputStream(labelPath)
          })

      try {
        labelInputStream.skip(4)
        val labelCount = labelInputStream.readInt()
        assert(imageCount == labelCount, f"imageCount $imageCount != labelCount $labelCount")

        val labels = (1 to imageCount).map { _ =>
          val digit = labelInputStream.readByte().toInt
          assert(digit >= 0 && digit <= 9)
          new DenseVector((0 to 9).map(i => if (i == digit) 1d else 0d).toArray)
        }

        images.zip(labels)
      } finally {
        labelInputStream.close()
      }

    } finally {
      imageInputStream.close()
    }
  }
}
