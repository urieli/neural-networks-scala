package neuralNetworks

import java.io.{ DataInputStream, DataOutputStream, FileInputStream, FileOutputStream }

import org.apache.commons.compress.compressors.CompressorStreamFactory
import org.rogach.scallop.ScallopConf

/** Slices MNIST input files, see [[MNISTLoader]] for file format details.
 */
object MNISTDataSlicer {
  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val imagePath = opt[String](default = Some("data/mnist-train-images-50k.gz"))
    val labelPath = opt[String](default = Some("data/mnist-train-labels-50k.gz"))
    val imagePathOut = opt[String](required = true, short = 'I')
    val labelPathOut = opt[String](required = true, short = 'L')
    val from = opt[Int](required = true)
    val length = opt[Int](required = true, short = 'n')
    verify()
  }

  def main(args: Array[String]): Unit = {
    val conf = new Conf(args)

    this.slice(
      conf.imagePath(),
      conf.labelPath(),
      conf.imagePathOut(),
      conf.labelPathOut(),
      conf.from(),
      conf.length())
  }

  /** Given a pair of MNIST input files, creates a pair of MNIST output files starting `from` and having length `length`.
   */
  def slice(imagePath: String, labelPath: String, imagePathOut: String, labelPathOut: String, from: Int, length: Int) = {
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
        imageBytes.clone()
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
          labelInputStream.readByte()
        }

        val imageSlice = images.slice(from, from + length)

        val imageOutputStream =
          new DataOutputStream(
            if (imagePathOut.endsWith(".gz")) {
              new CompressorStreamFactory().createCompressorOutputStream(
                CompressorStreamFactory.GZIP,
                new FileOutputStream(imagePathOut))
            } else {
              new FileOutputStream(imagePathOut)
            })

        try {
          imageOutputStream.writeInt(2051)
          imageOutputStream.writeInt(length)
          imageOutputStream.writeInt(rowCount)
          imageOutputStream.writeInt(columnCount)
          imageSlice.foreach(image => imageOutputStream.write(image))
        } finally {
          imageOutputStream.close()
        }

        val labelSlice = labels.slice(from, from + length)

        val labelOutputStream =
          new DataOutputStream(
            if (labelPathOut.endsWith(".gz")) {
              new CompressorStreamFactory().createCompressorOutputStream(
                CompressorStreamFactory.GZIP,
                new FileOutputStream(labelPathOut))
            } else {
              new FileOutputStream(labelPathOut)
            })

        try {
          labelOutputStream.writeInt(2049)
          labelOutputStream.writeInt(length)
          labelSlice.foreach(label => labelOutputStream.writeByte(label))
        } finally {
          labelOutputStream.close()
        }

      } finally {
        labelInputStream.close()
      }

    } finally {
      imageInputStream.close()
    }

  }
}
