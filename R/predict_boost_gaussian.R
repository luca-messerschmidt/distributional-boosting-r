#' Predict Method for Gaussian Gradient Boosting
#'
#' Predicts values from a fitted Gaussian boosting model.
#'
#' @param object A fitted boost_gaussian object.
#' @param newdata A data frame or matrix of new predictor values.
#' @param ... Further arguments, currently ignored.
#'
#' @returns A numeric vector of predictions.
#' @export
#' @method predict boost_gaussian
#'
#' @examples
#' set.seed(123)
#' x <- data.frame(
#'   x1 = rnorm(100),
#'   x2 = rnorm(100)
#' )
#' y <- 1 + 2 * x$x1 + rnorm(100)
#'
#' fit <- boost_gaussian(x, y)
#' predict(fit, x)

predict.boost_gaussian <- function(object, newdata, ...) {
  
  newdata <- as.data.frame(newdata)
  
  prediction <- rep(object$initial_value, nrow(newdata))
  
  for (m in seq_len(object$mstop)) {
    
    variable <- object$selected_variables[m]
    coefficient <- object$coefficients[m]
    
    prediction <- prediction + coefficient * newdata[[variable]]
  }
  
  return(prediction)
}