#' Generate CV Folds
#'
#' @param n number of observations
#' @param k number of folds
#' @param seed optional seed
#'
#' @returns integer vector of fold assignments
generate_folds <- function(n, k, seed = NULL){
  if(!is.null(seed)){
    set.seed(seed)
  }

  sample(rep(1:k, length.out = n))
}
