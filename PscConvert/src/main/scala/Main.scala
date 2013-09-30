import java.io.{BufferedReader, BufferedWriter, File, FileInputStream, FileOutputStream, InputStream, InputStreamReader, OutputStream, OutputStreamWriter, PrintWriter}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.Channels
import java.util.zip.{GZIPInputStream, GZIPOutputStream}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.sys.process.{BasicIO, Process, ProcessIO}

object Main extends App {
  if (args.length != 2) {
    Console.err.println("Usage: PscConvert <input_path> <output_path>")
    System.exit(1)
  }
  val inputPath = new File(args(0))
  val outputPath = new File(args(1))
  val psds = new PointSourceDataset(inputPath)

  def patchName(index: Int): String = {
    if (index < 57) {
      val c3 = index % 26
      val c2 = index / 26
      "a" + ('a' + c2).toChar + ('a' + c3).toChar
    } else {
      val c3 = (index - 57) % 26
      val c2 = (index - 57) / 26
      "b" + ('a' + c2).toChar + ('a' + c3).toChar
    }
  }
  val nPatches = 92
  val patches = ((0 until nPatches) map patchName).par
  //val nTasks = (Runtime.getRuntime.availableProcessors - 1) / 3 + 1
  val nTasks = 6
  patches.tasksupport = new ForkJoinTaskSupport(
      new scala.concurrent.forkjoin.ForkJoinPool(nTasks))
  //val patches = List("acc")
  //for (patch ← patches) psds.convertPatch(patch, outputPath)
  for (patch ← patches) {
    val patchOutDir = new File(outputPath, patch)
    patchOutDir.mkdirs()
    psds.convertPatch(patch, patchOutDir)
  }
}

class PointSourceDataset(pscPath: File) {
  def convertPatch(patch: String, outputPath: File) {
    def processOutput(in: InputStream) {
      // Assumes 'in' is already buffered
      val inChan = Channels.newChannel(in)

      val nFields = 4
      val fieldSize = 4
      val buf = ByteBuffer.allocate(nFields*fieldSize).order(
          ByteOrder.nativeOrder)

      val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
          new GZIPOutputStream(new FileOutputStream(new File(
          outputPath, s"cat_$patch.gz"))))))

      var nRead = inChan.read(buf)
      while (nRead >= 0) {
        buf.flip()
        val lon = buf.getFloat()
        val lat = buf.getFloat()
        val temp = buf.getFloat()
        val mag = buf.getFloat()
        out.println(f"$lon%g $lat%g $mag%g $temp%g")

        buf.clear()
        nRead = inChan.read(buf)
      }

      out.close()
      inChan.close()
    }

    val psc = new PscFile(new File(pscPath, s"psc_$patch.gz"))
    val workingDir = new File("../mags2temp")
    //val pb = Process.apply("bin/mags2temp", workingDir)
    val pb = Process.apply(Seq("bin/mags2temp", outputPath.getCanonicalPath),
        workingDir)
    //val pio = new ProcessIO(psc.process, processOutput, BasicIO.toStdErr)
    val pio = new ProcessIO(psc.process, BasicIO.toStdOut, BasicIO.toStdErr)
    // Block until completion
    pb.run(pio).exitValue
  }
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

class PscFile(file: File) {
  private def fieldsDefined(fields: Array[String]): Boolean = {
    (fields(6) != "\\N") && (fields(8) != "\\N") &&
      (fields(10) != "\\N") && (fields(12) != "\\N") &&
      (fields(14) != "\\N") && (fields(16) != "\\N")
  }

  def process(out: OutputStream): Unit = {
    val nFields = 8
    val fieldSize = 4
    val buf = ByteBuffer.allocate(nFields*fieldSize).order(
        ByteOrder.nativeOrder)
    // Assumes 'out' is already buffered
    val outChan = Channels.newChannel(out)

    val br = new BufferedReader(new InputStreamReader(new GZIPInputStream(
      new FileInputStream(file))))
    var line = br.readLine()
    var lineNo = 0

    while (line != null) {
      val fields = line.split("\\|")
      if (fieldsDefined(fields)) {
        buf.clear()

        val lon = fields(0).toDouble
        val lat = fields(1).toDouble
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

        buf.putFloat(lon.toFloat)
        buf.putFloat(lat.toFloat)
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
