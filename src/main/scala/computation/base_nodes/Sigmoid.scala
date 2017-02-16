package computation.base_nodes

import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid
import computation.{ComputationNode, DMat}

/**
  * A sigmoid unit
  *
  * fwd(x) = 1.0 / (1.0 + math.exp(-x))
  * bwd    = (fwd(x) :* (1 - fwd(x)))
  *
  * [1] http://ufldl.stanford.edu/wiki/index.php/Neural_Networks
  *
  * by Daniel Kohlsdorf
  */
case class Sigmoid(x: ComputationNode) extends ComputationNode {

  def fwd = {
    assignVal(sigmoid(x.fwd))
  }

  def bwd(error: DMat) = {
    collectGradient(error)
    val ones = DenseMatrix.ones[Double](value.rows, value.cols)
    val derivative = (ones - value) :* value
    x.bwd(error :* derivative)
  }

  def reset = {
    gradientOpt = None
    valueOpt    = None
    x.reset
  }

}