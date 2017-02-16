package experiments

import breeze.linalg.DenseMatrix
import computation.base_nodes.Tensor
import nn.FeedForwardLayerSigmoid

import scala.io.Source
import scala.util.Random

/**
  * 2 layer 3 spirals experiment
  *
  * by Daniel Kohlsdorf
  */
object SpiralMultiLayer {

  /**
    * Create the three spiral data
    */
  final val spiral = Random.shuffle(Source.fromFile("spiral.csv").getLines().map (line => {
    val cmp = line.trim().split(",")
    (DenseMatrix(cmp.slice(0, 2).map(_.toDouble)), cmp(2).toDouble.toInt)
  }).toList)

  def main(args: Array[String]): Unit = {
    // define tensors
    val input = Tensor()
    val W1 = Tensor()
    val B1 = Tensor()
    val W2 = Tensor()
    val B2 = Tensor()

    // split data into 80 train and 20 test
    val train = spiral.slice(0, (spiral.size * 0.8).toInt)
    val test  = spiral.slice((spiral.size * 0.2).toInt, spiral.size)

    // initialize the weights and shapes
    input :=  DenseMatrix.rand[Double](1,   2)
    W1    := (DenseMatrix.rand[Double](2, 100) - 0.5) * 0.01
    B1    :=         DenseMatrix.zeros(1, 100)
    W2    := (DenseMatrix.rand[Double](100, 3) - 0.5) * 0.01
    B2    :=         DenseMatrix.zeros(1,   3)

    // 2 layer network
    val nn = FeedForwardLayerSigmoid(FeedForwardLayerSigmoid(input, W1, B1), W2, B2)

    for {
      i <- 0 until 2500
    } {
      var total = 0.0
      for ((x, y) <- train) {
        val correct    = DenseMatrix.zeros[Double](1, 3)
        correct(0, y)  = 1.0

        // forward pass through complete network
        input := x
        val prediction = nn.fwd

        // compute error and compute all gradients
        // also update weights then reset all values
        val error      = -(correct - prediction)
        nn.bwd(error)
        nn.update(1.0)
        nn.reset

        total += 0.5 * (correct - prediction).map(x => math.pow(x, 2)).sum
      }
      if(i % 1000 == 0) println(total / 1000)
    }

    // compute accuracy
    var c = 0.0
    for ((x, y) <- test) {
      input := x
      nn.reset
      c = c + (if ( (nn.fwd.t.argmax._1) == y ) 1.0 else 0.0)
    }
    println("Acc: " + c / test.size.toDouble)
  }

}
