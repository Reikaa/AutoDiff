package computation

/**
  * Base class for all computation
  *
  * values and gradients act as a singleton
  * so results do not have to be recomputed.
  *
  * incoming errors are summed up [2] happens during weight sharing
  * but not in our case.
  *
  * however if the computation is executed multiple times the value and gradient
  * have to be reset recursively
  *
  * [1] http://cs231n.github.io/optimization-2/
  * [2] http://pages.cs.wisc.edu/~cs701-1/LectureNotes/trunk/cs701-lec-12-1-2015/cs701-lec-12-01-2015.pdf
  *
  * by Daniel Kohlsdorf
  */
abstract class ComputationNode {

  var valueOpt: Option[DMat]    = None
  var gradientOpt: Option[DMat] = None

  def gradient = gradientOpt.get

  def value = valueOpt.get

  def assignVal(x: DMat): DMat = {
    if(valueOpt == None) {
      valueOpt = Some(x)
    }
    value
  }

  def collectGradient(x: DMat): Unit = {
    if(gradientOpt == None) gradientOpt = Some(x)
    else gradientOpt = Some(x + gradient)
  }

  def reset: Unit

  def fwd: DMat

  def bwd(error: DMat): Unit

}

