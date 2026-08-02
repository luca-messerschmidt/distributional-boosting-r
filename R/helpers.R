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

# Capitalizes the first letter of a string (e.g. "gamma" -> "Gamma"), used
# for family-name labels in print()/summary() output.
.capitalize <- function(s) {
  paste0(toupper(substring(s, 1, 1)), substring(s, 2))
}
