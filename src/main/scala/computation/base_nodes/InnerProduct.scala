package computation.base_nodes

import computation.{ComputationNode, DMat}

/**
  * Inner product between two matrices
  *
  * fwd = x dot y
  * bwd(d, x) = d   * y.t
  * bwd(d, y) = x.t * d
  *
  * by Daniel Kohlsdorf
  */
case class InnerProduct(x: ComputationNode, y: ComputationNode) extends ComputationNode {

  def fwd = {
    assignVal(x.fwd * y.fwd)
  }

  def bwd(error: DMat) = {
    collectGradient(error)
    x.bwd(error * y.fwd.t)
    y.bwd(x.fwd.t * error)
  }

  def reset = {
    gradientOpt = None
    valueOpt    = None
    x.reset
    y.reset
  }

}
