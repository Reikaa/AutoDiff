package computation.base_nodes

import computation._

/**
  * Base variable to initialize a computation
  *
  * examples for Tensors include, weights, inputs and biases
  *
  * fwd: simply return value
  * bwd: just save error
  *
  * by Daniel Kohlsdorf
  */
class Tensor extends ComputationNode {

  def fwd = value

  def bwd(error: DMat) = collectGradient(error)

  def :=(x: DMat) = valueOpt = Some(x)

  def reset = {
    gradientOpt = None
  }

}

object Tensor {

  def apply(): Tensor = new Tensor()

}

