import java.io.{BufferedOutputStream, BufferedReader, FileInputStream, FileOutputStream, InputStreamReader}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.Channels
import java.util.zip.GZIPInputStream

object Main extends App {
  require(args.length == 2)
  val inputFile = args(0)
  val outputFile = args(1)
  val psc = new PscFile(inputFile)
  psc.process(outputFile)
}

object FilterSpec {
  val jBand = FilterSpec(1.235e-6, 0.006e-6, 0.162e-6, 0.001e-6,
      3.129e-13, 5.464e-15)
  val hBand = FilterSpec(1.662e-6, 0.009e-6, 0.251e-6, 0.002e-6,
      1.133e-13, 2.212e-15)
  val kBand = FilterSpec(2.159e-6, 0.011e-6, 0.262e-6, 0.002e-6,
      4.283e-14, 8.053e-16)
}

case class FilterSpec(lambdaIso: Double, sigmaLambdaIso: Double,
    bandwidth: Double, sigmaBandwidth: Double,
    zeroMagF: Double, sigmaZeroMagF: Double) {
  def integratedFlux(mag: Double, sigmaMag: Double): (Double, Double) = {
    val flux = zeroMagF * math.pow(10.0, -0.4*mag) * bandwidth
    val sigmaFlux = {
      val tmp1 = sigmaZeroMagF / zeroMagF
      val tmp2 = 0.4 * math.log(10.0) * sigmaMag
      val tmp3 = sigmaBandwidth / bandwidth
      flux * math.sqrt(tmp1*tmp1 + tmp2*tmp2 * tmp3*tmp3)
    }
    (flux, sigmaFlux)
  }
}

class PscFile(filename: String) {
  private def fieldsDefined(fields: Array[String]): Boolean = {
    (fields(6) != "\\N") && (fields(8) != "\\N") &&
      (fields(10) != "\\N") && (fields(12) != "\\N") &&
      (fields(14) != "\\N") && (fields(16) != "\\N")
  }

  def process(outputFilename: String): Unit = {
    val buf = ByteBuffer.allocate(6*4).order(ByteOrder.LITTLE_ENDIAN)
    val outChan = Channels.newChannel(new BufferedOutputStream(
        new FileOutputStream(outputFilename)))

    val br = new BufferedReader(new InputStreamReader(new GZIPInputStream(
      new FileInputStream(filename))))
    var line = br.readLine()
    var lineNo = 0

    while (line != null) {
      val fields = line.split("\\|")
      if (fieldsDefined(fields)) {
        buf.clear()

        val jMag = fields(6).toDouble
        val sigmaJMag = fields(8).toDouble
        val hMag = fields(10).toDouble
        val sigmaHMag = fields(12).toDouble
        val kMag = fields(14).toDouble
        val sigmaKMag = fields(16).toDouble
  
        val (jFlux, sigmaJFlux) =
            FilterSpec.jBand.integratedFlux(jMag, sigmaJMag)
        val (hFlux, sigmaHFlux) =
            FilterSpec.hBand.integratedFlux(hMag, sigmaHMag)
        val (kFlux, sigmaKFlux) =
            FilterSpec.kBand.integratedFlux(kMag, sigmaKMag)

        buf.putFloat(jFlux.toFloat)
        buf.putFloat(sigmaJFlux.toFloat)
        buf.putFloat(hFlux.toFloat)
        buf.putFloat(sigmaHFlux.toFloat)
        buf.putFloat(kFlux.toFloat)
        buf.putFloat(sigmaKFlux.toFloat)
        buf.flip()
        outChan.write(buf)
      }
      line = br.readLine()
      lineNo += 1
    }

    br.close()
    outChan.close()
  }
}
