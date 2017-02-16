package nn

import computation.{Computation, ComputationNode, DMat}
import computation.base_nodes.Tensor

case class FeedForwardLayerSigmoid(x: ComputationNode, w: Tensor, b: Tensor) extends ComputationNode {

  final val Logit = Computation(x) * w + b

  final val Layer = Computation.sigmoid(Logit).x

  def fwd = Layer.fwd

  def bwd(error: DMat) = Layer.bwd(error)

  def update(rate: Double): Unit = {
    w := w.value - (rate * w.gradient)
    b := b.value - (rate * b.gradient)
    if(x.isInstanceOf[FeedForwardLayerSigmoid]) x.asInstanceOf[FeedForwardLayerSigmoid].update(rate)
  }

  def reset = Layer.reset


}
