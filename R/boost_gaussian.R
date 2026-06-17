#' Distributional Gaussian Gradient Boosting
#'
#' Fits a component-wise gradient boosting model for Gaussian responses,
#' jointly estimating the location (mean) and scale (standard deviation) via
#' cyclic updates on two separate design matrices.
#'
#' @param X A data frame or matrix of predictors for the location (mean) model.
#' @param Z A data frame or matrix of predictors for the scale (log-sd) model.
#' @param y A numeric response vector.
#' @param mstop Number of boosting iterations (each iteration updates both
#'   the location and the scale submodel once).
#' @param nu_mu  Learning rate for the location submodel.
#' @param nu_sigma Learning rate for the scale submodel.
#'
#' @details
#' This implements component-wise gradient boosting for a Gaussian
#' location-scale model (in the spirit of \code{gamboostLSS}).  The two
#' distribution parameters are updated cyclically, one base learner at a time:
#' \enumerate{
#'   \item \strong{Location step} – the negative gradient of the Gaussian
#'     log-likelihood w.r.t. \eqn{\mu} is
#'     \eqn{u_\mu = (y - \mu) / \sigma^2}.
#'     Each column of \code{X} is fit by a simple linear base learner
#'     (\code{lm(u_mu ~ x_j)}, with intercept); the column with the lowest
#'     residual sum of squares is selected and \eqn{\hat\mu} is updated by a
#'     small step \code{nu_mu}.
#'   \item \strong{Scale step} – the negative gradient w.r.t. \eqn{\log\sigma}
#'     is \eqn{u_\sigma = (y - \mu)^2 / \sigma^2 - 1}.  A simple linear base
#'     learner is fit to each column of \code{Z} and the best is selected;
#'     \eqn{\log\hat\sigma} is updated by a small step \code{nu_sigma}.
#'     Working on the log scale keeps \eqn{\hat\sigma > 0} and naturally bounds
#'     the step size, so no additional gradient rescaling is needed.
#' }
#'
#' \strong{Predictor standardisation}: both \code{X} and \code{Z} are
#' standardised internally (zero mean, unit variance) before fitting so that
#' all base learners compete on the same scale.  Stored coefficients are
#' converted back to the original predictor scale (see below).
#'
#' \strong{Base learners include an intercept}: each base learner is a full
#' simple linear regression \code{lm(u ~ x_j)}.  The intercept term allows the
#' overall level of the linear predictor to be corrected at every step.  This
#' matters most for the scale submodel, whose level is initialised at
#' \eqn{\log(\mathrm{sd}(y))} and must be free to move toward the correct
#' residual scale.
#'
#' Coefficients for both submodels are stored and reported on the
#' \emph{original predictor scale} (not the standardised scale).  For the
#' scale submodel the linear predictor is \eqn{\log\sigma}, so a coefficient
#' expresses the additive change in \eqn{\log\sigma} per unit change in the
#' original predictor (equivalently, a multiplicative factor
#' \eqn{\exp(\cdot)} on \eqn{\sigma}).
#'
#' @returns A list of class \code{"boost_gaussian"} containing fitted values,
#'   coefficient paths, centering/scaling constants, and diagnostics for both
#'   the location and scale submodels.
#'
#' @importFrom stats sd
#' @export
#'
#' @examples
#' set.seed(123)
#' X <- data.frame(x1 = rnorm(100), x2 = rnorm(100))
#' Z <- data.frame(z1 = rnorm(100), z2 = rnorm(100))
#' y <- 2 + 3 * X$x1 + exp(0.5 * Z$z1) * rnorm(100)
#'
#' fit <- boost_gaussian(X, Z, y, mstop = 100, nu_mu = 0.1, nu_sigma = 0.1)
#' predict(fit, newdata_X = X[1:5, ], newdata_Z = Z[1:5, ])

boost_gaussian <- function(X, Z, y, mstop = 100, nu_mu = 0.1, nu_sigma = 0.1) {
  
  # ── Type checks BEFORE coercion ──────────────────────────────────────────────
  if (!is.matrix(X) && !is.data.frame(X)) {
    stop("X must be a matrix or data.frame.", call. = FALSE)
  }
  if (!is.matrix(Z) && !is.data.frame(Z)) {
    stop("Z must be a matrix or data.frame.", call. = FALSE)
  }
  
  X <- as.data.frame(X)
  Z <- as.data.frame(Z)
  y <- as.numeric(y)
  
  # ── Scalar parameter validation ──────────────────────────────────────────────
  if (!is.numeric(mstop) || length(mstop) != 1L || is.na(mstop) ||
      !is.finite(mstop) || mstop < 1L || mstop != as.integer(mstop)) {
    stop("mstop must be a single positive integer.", call. = FALSE)
  }
  mstop <- as.integer(mstop)
  
  if (!is.numeric(nu_mu) || length(nu_mu) != 1L || is.na(nu_mu) ||
      !is.finite(nu_mu) || nu_mu <= 0 || nu_mu > 1) {
    stop("nu_mu must be a single finite value in (0, 1].", call. = FALSE)
  }
  if (!is.numeric(nu_sigma) || length(nu_sigma) != 1L || is.na(nu_sigma) ||
      !is.finite(nu_sigma) || nu_sigma <= 0 || nu_sigma > 1) {
    stop("nu_sigma must be a single finite value in (0, 1].", call. = FALSE)
  }
  
  # ── Dimension checks ─────────────────────────────────────────────────────────
  if (nrow(X) != length(y)) {
    stop("Number of rows in X must match length of y.", call. = FALSE)
  }
  if (nrow(Z) != length(y)) {
    stop("Number of rows in Z must match length of y.", call. = FALSE)
  }
  
  # ── Numeric-type checks for predictor columns ────────────────────────────────
  if (!all(vapply(X, is.numeric, logical(1)))) {
    stop("All columns of X must be numeric.", call. = FALSE)
  }
  if (!all(vapply(Z, is.numeric, logical(1)))) {
    stop("All columns of Z must be numeric.", call. = FALSE)
  }
  
  # ── Finite-value checks ──────────────────────────────────────────────────────
  if (anyNA(X)) stop("X must not contain missing (NA/NaN) values.", call. = FALSE)
  if (anyNA(Z)) stop("Z must not contain missing (NA/NaN) values.", call. = FALSE)
  if (anyNA(y)) stop("y must not contain missing (NA/NaN) values.", call. = FALSE)
  if (!all(is.finite(as.matrix(X)))) stop("X must not contain infinite values.", call. = FALSE)
  if (!all(is.finite(as.matrix(Z)))) stop("Z must not contain infinite values.", call. = FALSE)
  if (!all(is.finite(y)))            stop("y must not contain infinite values.", call. = FALSE)
  
  n  <- length(y)
  pX <- ncol(X)
  pZ <- ncol(Z)
  
  # ── Standardise X and Z (zero mean, unit variance) ──────────────────────────
  # Predictor standardisation ensures all base learners compete on the same
  # scale regardless of the original units of X and Z.
  X_center <- vapply(X, mean, numeric(1))
  X_scale  <- vapply(X, sd,   numeric(1))
  X_scale[X_scale == 0] <- 1   # guard against constant columns
  
  Z_center <- vapply(Z, mean, numeric(1))
  Z_scale  <- vapply(Z, sd,   numeric(1))
  Z_scale[Z_scale == 0] <- 1
  
  X_std <- as.data.frame(sweep(sweep(as.matrix(X), 2L, X_center, "-"),
                               2L, X_scale, "/"))
  Z_std <- as.data.frame(sweep(sweep(as.matrix(Z), 2L, Z_center, "-"),
                               2L, Z_scale, "/"))
  
  # ── Initialisation ────────────────────────────────────────────────────────────
  y_sd <- sd(y)
  if (!is.finite(y_sd) || y_sd == 0) y_sd <- 1
  
  mu_hat        <- rep(mean(y), n)
  log_sigma_hat <- rep(log(y_sd), n)
  sigma_hat     <- exp(log_sigma_hat)
  
  # Per-iteration tracking - coefficients stored on the STANDARDISED scale
  # internally; conversion to original scale happens in the post-fit block.
  # Each base learner contributes a SLOPE (coef_*) and an intercept
  # (intercept_step_*); both are accumulated.
  selected_mu          <- character(mstop)
  coef_mu              <- numeric(mstop)   # slope, standardised scale
  intercept_step_mu    <- numeric(mstop)   # intercept contribution per step
  selected_sigma       <- character(mstop)
  coef_sigma           <- numeric(mstop)   # slope, standardised scale (log-sigma)
  intercept_step_sigma <- numeric(mstop)   # intercept contribution per step
  
  # ── Cyclic boosting loop ──────────────────────────────────────────────────────
  for (m in seq_len(mstop)) {
    
    # ── Location step ──────────────────────────────────────────────────────────
    # Negative gradient of the Gaussian log-likelihood w.r.t. mu
    u_mu <- (y - mu_hat) / sigma_hat^2
    
    best_var_mu       <- NULL
    best_slope_mu     <- NULL
    best_intercept_mu <- NULL
    best_rss_mu       <- Inf
    
    # Each base learner is a simple linear regression with intercept:
    # Because x_j is mean-centred (standardised), the OLS intercept is exactly
    # a = mean(u_mu) and the slope is b = <x_j, u_mu> / <x_j, x_j>.
    # The intercept term lets the model correct the overall LEVEL of the linear
    # predictor, not just its dependence on x_j.
    a_mu <- mean(u_mu)
    for (j in seq_len(pX)) {
      xj   <- X_std[[j]]
      ss   <- sum(xj^2)
      b    <- if (ss > 0) sum(xj * u_mu) / ss else 0
      pred <- a_mu + b * xj
      rss  <- sum((u_mu - pred)^2)
      if (rss < best_rss_mu) {
        best_rss_mu       <- rss
        best_var_mu       <- names(X_std)[j]
        best_slope_mu     <- b
        best_intercept_mu <- a_mu
      }
    }
    
    # Update mu by a small step nu_mu along the selected base learner.
    mu_hat <- mu_hat +
      nu_mu * (best_intercept_mu + best_slope_mu * X_std[[best_var_mu]])
    
    selected_mu[m]       <- best_var_mu
    coef_mu[m]           <- nu_mu * best_slope_mu       # slope contribution
    intercept_step_mu[m] <- nu_mu * best_intercept_mu   # intercept contribution
    
    # ── Scale step ──────────────────────────────────────────────────────────────
    # Negative gradient of the Gaussian log-likelihood w.r.t. log(sigma)
    # The log link keeps sigma > 0. Additional gradient stabilisation is not
    # implemented here; stability is controlled through small learning rates.
    u_sigma <- (y - mu_hat)^2 / sigma_hat^2 - 1
    
    best_var_sigma       <- NULL
    best_slope_sigma     <- NULL
    best_intercept_sigma <- NULL
    best_rss_sigma       <- Inf
    
    # Intercept-corrected base learner (see location step).  For the scale
    # model the intercept is what lets the overall log-sigma level move from
    # its initialisation toward the correct residual scale.
    a_sigma <- mean(u_sigma)
    for (j in seq_len(pZ)) {
      zj   <- Z_std[[j]]
      ss   <- sum(zj^2)
      b    <- if (ss > 0) sum(zj * u_sigma) / ss else 0
      pred <- a_sigma + b * zj
      rss  <- sum((u_sigma - pred)^2)
      if (rss < best_rss_sigma) {
        best_rss_sigma       <- rss
        best_var_sigma       <- names(Z_std)[j]
        best_slope_sigma     <- b
        best_intercept_sigma <- a_sigma
      }
    }
    
    # Update log-sigma by a small step nu_sigma along the selected base learner.
    log_sigma_hat <- log_sigma_hat +
      nu_sigma * (best_intercept_sigma + best_slope_sigma * Z_std[[best_var_sigma]])
    sigma_hat <- exp(log_sigma_hat)
    
    selected_sigma[m]       <- best_var_sigma
    coef_sigma[m]           <- nu_sigma * best_slope_sigma
    intercept_step_sigma[m] <- nu_sigma * best_intercept_sigma
  }
  
  # ── Post-fit diagnostics ──────────────────────────────────────────────────────
  residuals <- y - mu_hat
  tss       <- sum((y - mean(y))^2)
  r_squared <- if (tss == 0) NA_real_ else 1 - sum(residuals^2) / tss
  
  # ── Net coefficients: standardised scale ────────────────────────────────────
  net_coef_mu_std <- tapply(coef_mu, selected_mu, sum)
  net_coef_mu_std <- net_coef_mu_std[names(X_std)]
  names(net_coef_mu_std) <- names(X_std)
  net_coef_mu_std[is.na(net_coef_mu_std)] <- 0
  
  net_coef_sigma_std <- tapply(coef_sigma, selected_sigma, sum)
  net_coef_sigma_std <- net_coef_sigma_std[names(Z_std)]
  names(net_coef_sigma_std) <- names(Z_std)
  net_coef_sigma_std[is.na(net_coef_sigma_std)] <- 0
  
  # ── Net coefficients: original predictor scale ───────────────────────────────
  # The fitted linear predictor on the standardised scale is
  #   init + sum(intercept_steps) + sum_j net_slope_std_j * x_std_j.
  # Converting x_std_j = (x_j - center_j)/scale_j to the original scale gives
  # slope_orig_j = slope_std_j / scale_j and folds the -center_j/scale_j terms
  # into the intercept.
  total_intercept_step_mu <- sum(intercept_step_mu)
  net_coef_mu_orig <- net_coef_mu_std / X_scale
  intercept_mu     <- mean(y) + total_intercept_step_mu -
    sum(net_coef_mu_orig * X_center)
  
  # Scale: same logic on the log-sigma linear predictor.
  #   One unit change in original z_j changes log(sigma) by beta_sigma_orig_j,
  #   i.e. multiplies sigma by exp(beta_sigma_orig_j).
  total_intercept_step_sigma <- sum(intercept_step_sigma)
  net_coef_sigma_orig <- net_coef_sigma_std / Z_scale
  intercept_sigma     <- log(y_sd) + total_intercept_step_sigma -
    sum(net_coef_sigma_orig * Z_center)
  
  result <- list(
    # ── Location submodel ──
    initial_mu             = mean(y),
    selected_mu            = selected_mu,
    coef_mu                = coef_mu,
    intercept_step_mu      = intercept_step_mu,
    net_coef_mu_std        = net_coef_mu_std,
    net_coef_mu_orig       = net_coef_mu_orig,
    intercept_mu           = intercept_mu,
    fitted_mu              = as.numeric(mu_hat),
    # ── Scale submodel ──
    initial_log_sigma      = log(y_sd),
    selected_sigma         = selected_sigma,
    coef_sigma             = coef_sigma,
    intercept_step_sigma   = intercept_step_sigma,
    net_coef_sigma_std     = net_coef_sigma_std,
    net_coef_sigma_orig    = net_coef_sigma_orig,
    intercept_sigma        = intercept_sigma,
    fitted_log_sigma       = as.numeric(log_sigma_hat),
    fitted_sigma           = as.numeric(sigma_hat),
    # ── Shared ──
    residuals              = as.numeric(residuals),
    r_squared              = r_squared,
    mstop                  = mstop,
    nu_mu                  = nu_mu,
    nu_sigma               = nu_sigma,
    n                      = n,
    pX                     = pX,
    pZ                     = pZ,
    X_names                = names(X),
    Z_names                = names(Z),
    X_center               = X_center,
    X_scale                = X_scale,
    Z_center               = Z_center,
    Z_scale                = Z_scale,
    # Standardised training design (named matrices). Stored so that predict()
    # can reconstruct fitted values at any earlier mstop on the training data
    # (in-sample early stopping) using the same accumulation as out-of-sample.
    X_std                  = as.matrix(X_std),
    Z_std                  = as.matrix(Z_std),
    call                   = match.call()
  )
  
  class(result) <- "boost_gaussian"
  return(result)
}