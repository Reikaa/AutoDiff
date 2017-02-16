package computation

import computation.base_nodes._


/**
  * A computation is basically a nice way to build the computation graph.
  *
  * by Daniel Kohlsdorf
  */
class Computation(val x: ComputationNode) {

  def +(y: ComputationNode): Computation = new Computation(Addition(x, y))

  def *(y: ComputationNode): Computation = new Computation(InnerProduct(x, y))

}

object Computation {

  def apply(x: ComputationNode): Computation = new Computation(x)

  def sigmoid(x: Computation): Computation = new Computation(Sigmoid(x.x))

}