package computation.base_nodes

import computation._

/**
  * Add two matrices
  *
  * error d is propagated unchanged to x and y
  *
  * fwd(x, y) = x + y
  * bwd(d, x) = d
  * bwd(d, y) = d
  *
  * by Daniel Kohlsdorf
  */
case class Addition(x: ComputationNode, y: ComputationNode) extends ComputationNode {

  def fwd = assignVal(x.fwd + y.fwd)

  def bwd(error: DMat) = {
    collectGradient(error)
    x.bwd(error)
    y.bwd(error)
  }

  def reset = {
    gradientOpt = None
    valueOpt    = None
    x.reset
    y.reset
  }

}

