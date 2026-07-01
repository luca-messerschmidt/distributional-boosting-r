#' Distributional Gaussian Gradient Boosting
#'
#' Fits a component-wise gradient boosting model for Gaussian responses,
#' jointly estimating the location (mean) and scale (standard deviation).
#'
#' @param X A data frame or matrix of predictors for the location (mean) model.
#'   Ignored if \code{formula} is supplied.
#' @param Z A data frame or matrix of predictors for the scale (log-sd) model.
#'   Ignored if \code{formula} is supplied.
#' @param y A numeric response vector. Ignored if \code{formula} is supplied.
#' @param formula A list with components \code{mu} (a two-sided formula,
#'   e.g. \code{y ~ x1 + x2}) and \code{sigma} (a formula for the scale
#'   submodel, e.g. \code{~ z1 + z2}). When supplied, \code{X}, \code{Z}, and
#'   \code{y} are constructed from \code{data} instead.
#' @param data A data frame containing the variables referenced in
#'   \code{formula}. Required (and only used) when \code{formula} is supplied.
#' @param mstop Number of boosting iterations. Either a single positive
#'   integer (applied to both submodels) or, for \code{method = "cyclic"}
#'   only, a named vector/list \code{c(mu = ..., sigma = ...)} giving separate
#'   iteration budgets for the two submodels. When \code{patience} is
#'   supplied, \code{mstop} is instead the maximum number of rounds to search
#'   (must be a single integer in that case).
#' @param nu_mu  Learning rate for the location submodel.
#' @param nu_sigma Learning rate for the scale submodel.
#' @param method Boosting update schedule. \code{"cyclic"} (default)
#'   alternates location/scale updates each iteration (until each submodel's
#'   own \code{mstop} budget, if separate, is exhausted). \code{"noncyclic"}
#'   picks, at each iteration, whichever submodel update yields the larger
#'   Gaussian log-likelihood improvement, and updates only that one.
#' @param patience If supplied, enables internal early stopping: a single
#'   positive integer giving the number of consecutive rounds without
#'   validation-risk improvement allowed before stopping the search early.
#'   Requires a single scalar \code{mstop} (the maximum rounds to search).
#'   The optimal number of rounds found this way (\code{best_round}) is then
#'   used to refit the model on the \emph{full} \code{X}/\code{Z}/\code{y}, so
#'   the returned model itself is always trained on all the supplied data.
#' @param validation_split Fraction of rows (in \code{(0, 1)}) held out to
#'   evaluate validation risk during the \code{patience} search. Ignored if
#'   \code{patience} is \code{NULL}. Default \code{0.2}.
#' @param seed Optional integer seed for the random train/validation split
#'   used when \code{patience} is supplied (for reproducibility). Ignored if
#'   \code{patience} is \code{NULL}.
#'
#' @details
#' This implements component-wise gradient boosting for a Gaussian
#' location-scale model (in the spirit of \code{gamboostLSS}). The two
#' distribution parameters are updated one base learner at a time:
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
#'     the step size.
#' }
#' Under \code{method = "cyclic"} both steps are taken every iteration (each
#' submodel stops once it reaches its own \code{mstop} budget, while the other
#' keeps going). Under \code{method = "noncyclic"} only the step with the
#' larger likelihood improvement is taken each iteration.
#'
#' \strong{Risk trace}: the in-sample Gaussian negative log-likelihood is
#' recorded after every round (\code{risk}, plus the pre-boosting baseline
#' \code{risk0}), so convergence can be inspected directly or via
#' \code{plot(fit, type = "risk")}.
#'
#' \strong{Internal early stopping}: when \code{patience} is supplied, a
#' single random train/validation split (\code{validation_split}) is used to
#' search for the number of rounds (up to \code{mstop}) that minimizes
#' validation risk, stopping the search once \code{patience} rounds pass
#' without improvement. The model that is actually returned is then refit on
#' the complete data with \code{mstop} set to that optimal round count; the
#' search details (including the validation risk trace) are stored in
#' \code{fit$early_stopping}.
#'
#' \strong{Predictor standardisation}: both \code{X} and \code{Z} are
#' standardised internally (zero mean, unit variance) before fitting so that
#' all base learners compete on the same scale. Stored coefficients are
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
#' @importFrom stats sd dnorm model.frame model.matrix terms delete.response
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
#'
#' # Non-cyclic updates: only the more useful submodel update is taken
#' fit_nc <- boost_gaussian(X, Z, y, mstop = 100, method = "noncyclic")
#'
#' # Separate iteration budgets for mu and sigma (cyclic only)
#' fit_2m <- boost_gaussian(X, Z, y, mstop = c(mu = 100, sigma = 30))
#'
#' # Formula interface
#' df <- cbind(X, Z, y = y)
#' fit_f <- boost_gaussian(formula = list(mu = y ~ x1 + x2, sigma = ~ z1 + z2),
#'                         data = df, mstop = 100)
#'
#' # Internal early stopping (searches up to mstop rounds on a 80/20 split,
#' # then refits the chosen number of rounds on all the data)
#' fit_es <- boost_gaussian(X, Z, y, mstop = 200, patience = 10, seed = 1)
#' fit_es$early_stopping$best_round

boost_gaussian <- function(X = NULL, Z = NULL, y = NULL, formula = NULL,
                           data = NULL, mstop = 100, nu_mu = 0.1,
                           nu_sigma = 0.1, method = c("cyclic", "noncyclic"),
                           patience = NULL, validation_split = 0.2,
                           seed = NULL) {

  method <- match.arg(method)

  # ── Formula interface: build X, Z, y from data ──────────────────────────────
  terms_mu <- NULL
  terms_sigma <- NULL
  if (!is.null(formula)) {
    built       <- .build_design_from_formula(formula, data)
    X           <- built$X
    Z           <- built$Z
    y           <- built$y
    terms_mu    <- built$terms_mu
    terms_sigma <- built$terms_sigma
  }

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

  # ── mstop / method validation ────────────────────────────────────────────────
  mstop_parsed <- .parse_mstop(mstop, method)
  mstop_mu     <- mstop_parsed$mstop_mu
  mstop_sigma  <- mstop_parsed$mstop_sigma

  if (!is.numeric(nu_mu) || length(nu_mu) != 1L || is.na(nu_mu) ||
      !is.finite(nu_mu) || nu_mu <= 0 || nu_mu > 1) {
    stop("nu_mu must be a single finite value in (0, 1].", call. = FALSE)
  }
  if (!is.numeric(nu_sigma) || length(nu_sigma) != 1L || is.na(nu_sigma) ||
      !is.finite(nu_sigma) || nu_sigma <= 0 || nu_sigma > 1) {
    stop("nu_sigma must be a single finite value in (0, 1].", call. = FALSE)
  }

  if (!is.null(patience) && mstop_mu != mstop_sigma) {
    stop("patience (internal early stopping) requires a single scalar ",
         "mstop (the maximum search budget); separate mu/sigma mstop ",
         "values are not supported together with early stopping.",
         call. = FALSE)
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

  # ── Internal early stopping: search for the best round count on a holdout
  # split, then fall through to the normal full-data fit below using that
  # round count as mstop_mu/mstop_sigma. ──────────────────────────────────────
  early_stopping_info <- NULL
  if (!is.null(patience)) {
    search <- .early_stop_search(
      X = X, Z = Z, y = y, mstop_max = mstop_mu, nu_mu = nu_mu,
      nu_sigma = nu_sigma, method = method,
      validation_split = validation_split, patience = patience, seed = seed
    )
    mstop_mu    <- search$best_round
    mstop_sigma <- search$best_round
    early_stopping_info <- list(
      used             = TRUE,
      patience         = as.integer(patience),
      validation_split = validation_split,
      seed             = seed,
      mstop_max        = mstop_parsed$mstop_mu,
      best_round       = search$best_round,
      n_train          = search$n_train,
      n_val            = search$n_val,
      val_risk0        = search$val_risk0,
      val_risk         = search$val_risk
    )
  }

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

  # ── Boosting loop ────────────────────────────────────────────────────────────
  # Per-step history is recorded per submodel (selected_*, coef_*,
  # intercept_step_*), in the order each submodel's steps were actually
  # applied, together with the GLOBAL round at which each step happened
  # (round_*). For the classic single-mstop cyclic case round_mu == round_sigma
  # == seq_len(mstop), reproducing the original algorithm exactly. For
  # noncyclic / two-mstop-cyclic, round_mu and round_sigma can differ in
  # length and have gaps, which is how predict()/plot() reconstruct partial
  # fits correctly regardless of update schedule.
  fit_state <- .run_boosting_loop(
    X_std = X_std, Z_std = Z_std, y = y,
    mu_hat = mu_hat, sigma_hat = sigma_hat, log_sigma_hat = log_sigma_hat,
    nu_mu = nu_mu, nu_sigma = nu_sigma,
    method = method, mstop_mu = mstop_mu, mstop_sigma = mstop_sigma
  )

  mu_hat        <- fit_state$mu_hat
  sigma_hat     <- fit_state$sigma_hat
  log_sigma_hat <- fit_state$log_sigma_hat

  selected_mu          <- fit_state$selected_mu
  coef_mu              <- fit_state$coef_mu
  intercept_step_mu    <- fit_state$intercept_step_mu
  round_mu             <- fit_state$round_mu
  selected_sigma       <- fit_state$selected_sigma
  coef_sigma           <- fit_state$coef_sigma
  intercept_step_sigma <- fit_state$intercept_step_sigma
  round_sigma          <- fit_state$round_sigma
  n_rounds             <- fit_state$n_rounds
  step_log             <- fit_state$step_log

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
    round_mu               = round_mu,
    net_coef_mu_std        = net_coef_mu_std,
    net_coef_mu_orig       = net_coef_mu_orig,
    intercept_mu           = intercept_mu,
    fitted_mu              = as.numeric(mu_hat),
    # ── Scale submodel ──
    initial_log_sigma      = log(y_sd),
    selected_sigma         = selected_sigma,
    coef_sigma             = coef_sigma,
    intercept_step_sigma   = intercept_step_sigma,
    round_sigma            = round_sigma,
    net_coef_sigma_std     = net_coef_sigma_std,
    net_coef_sigma_orig    = net_coef_sigma_orig,
    intercept_sigma        = intercept_sigma,
    fitted_log_sigma       = as.numeric(log_sigma_hat),
    fitted_sigma           = as.numeric(sigma_hat),
    # ── Shared ──
    residuals              = as.numeric(residuals),
    r_squared              = r_squared,
    method                 = method,
    mstop                  = n_rounds,
    mstop_mu               = length(selected_mu),
    mstop_sigma            = length(selected_sigma),
    step_log               = step_log,
    risk                   = fit_state$risk,
    risk0                  = fit_state$risk0,
    early_stopping         = early_stopping_info,
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
    terms_mu               = terms_mu,
    terms_sigma            = terms_sigma,
    call                   = match.call()
  )

  class(result) <- "boost_gaussian"
  return(result)
}

# ── Internal: mstop parsing ──────────────────────────────────────────────────

# Accepts a single positive integer (applied to both submodels) or, for
# method == "cyclic", a named vector/list c(mu = ..., sigma = ...) giving
# separate iteration budgets.
.parse_mstop <- function(mstop, method) {
  err <- function() {
    stop("mstop must be a single positive integer, or (for method = ",
         "'cyclic') a named vector/list with elements 'mu' and 'sigma'.",
         call. = FALSE)
  }

  if (is.list(mstop)) mstop <- unlist(mstop)
  if (!is.numeric(mstop) || length(mstop) == 0L) err()

  if (length(mstop) == 1L) {
    if (is.na(mstop) || !is.finite(mstop) || mstop < 1L ||
        mstop != as.integer(mstop)) {
      err()
    }
    m <- as.integer(mstop)
    return(list(mstop_mu = m, mstop_sigma = m))
  }

  if (length(mstop) == 2L && !is.null(names(mstop)) &&
      setequal(names(mstop), c("mu", "sigma"))) {
    if (method != "cyclic") {
      stop("Separate mu/sigma mstop values are only supported for ",
           "method = 'cyclic'.", call. = FALSE)
    }
    if (anyNA(mstop) || any(!is.finite(mstop)) || any(mstop < 1L) ||
        any(mstop != as.integer(mstop))) {
      err()
    }
    return(list(mstop_mu    = as.integer(mstop[["mu"]]),
                mstop_sigma = as.integer(mstop[["sigma"]])))
  }

  err()
}

# ── Internal: formula interface ──────────────────────────────────────────────

.build_design_from_formula <- function(formula, data) {
  if (!is.list(formula) || is.null(formula$mu) || is.null(formula$sigma)) {
    stop("formula must be a list with components 'mu' and 'sigma'.",
         call. = FALSE)
  }
  f_mu    <- formula$mu
  f_sigma <- formula$sigma
  if (!inherits(f_mu, "formula") || length(f_mu) != 3L) {
    stop("formula$mu must be a two-sided formula, e.g. y ~ x1 + x2.",
         call. = FALSE)
  }
  if (!inherits(f_sigma, "formula")) {
    stop("formula$sigma must be a formula, e.g. ~ z1 + z2.", call. = FALSE)
  }
  if (is.null(data) || !is.data.frame(data)) {
    stop("data must be a data.frame when formula is supplied.", call. = FALSE)
  }

  terms_mu <- stats::terms(f_mu)
  mf_mu    <- stats::model.frame(terms_mu, data = data)
  y        <- mf_mu[[1L]]
  X        <- stats::model.matrix(terms_mu, data = mf_mu)
  X        <- X[, colnames(X) != "(Intercept)", drop = FALSE]

  terms_sigma <- stats::terms(f_sigma)
  mf_sigma    <- stats::model.frame(terms_sigma, data = data)
  Z           <- stats::model.matrix(terms_sigma, data = mf_sigma)
  Z           <- Z[, colnames(Z) != "(Intercept)", drop = FALSE]

  list(X = as.data.frame(X), Z = as.data.frame(Z), y = as.numeric(y),
       terms_mu = terms_mu, terms_sigma = terms_sigma)
}

# ── Internal: early stopping search ─────────────────────────────────────────

# Splits (X, Z, y) into a train/validation partition, runs the boosting loop
# on the train partition up to mstop_max rounds while tracking validation
# risk after every round (using the SAME per-round base-learner updates,
# applied to the standardised validation design), and stops early once
# `patience` rounds pass without validation-risk improvement. Returns only
# the round count that achieved the best validation risk (plus the trace) —
# the throwaway train-only fit itself is discarded; the calling boost_gaussian()
# refits on the full data at that round count.
.early_stop_search <- function(X, Z, y, mstop_max, nu_mu, nu_sigma, method,
                               validation_split, patience, seed) {
  if (!is.numeric(validation_split) || length(validation_split) != 1L ||
      is.na(validation_split) || validation_split <= 0 ||
      validation_split >= 1) {
    stop("validation_split must be a single value in (0, 1).", call. = FALSE)
  }
  if (!is.numeric(patience) || length(patience) != 1L || is.na(patience) ||
      !is.finite(patience) || patience < 1L ||
      patience != as.integer(patience)) {
    stop("patience must be a single positive integer.", call. = FALSE)
  }
  patience <- as.integer(patience)

  n <- nrow(X)
  if (n < 3L) {
    stop("Need at least 3 observations to use internal early stopping.",
         call. = FALSE)
  }
  if (!is.null(seed)) set.seed(seed)

  n_val <- max(1L, round(validation_split * n))
  if (n_val >= n) n_val <- n - 1L
  val_idx   <- sample.int(n, n_val)
  train_idx <- setdiff(seq_len(n), val_idx)

  train_X <- X[train_idx, , drop = FALSE]
  train_Z <- Z[train_idx, , drop = FALSE]
  train_y <- y[train_idx]
  val_X   <- X[val_idx, , drop = FALSE]
  val_Z   <- Z[val_idx, , drop = FALSE]
  val_y   <- y[val_idx]

  # Standardise using TRAIN statistics only, mirroring predict()'s
  # out-of-sample standardisation path.
  X_center <- vapply(train_X, mean, numeric(1))
  X_scale  <- vapply(train_X, sd,   numeric(1))
  X_scale[X_scale == 0] <- 1
  Z_center <- vapply(train_Z, mean, numeric(1))
  Z_scale  <- vapply(train_Z, sd,   numeric(1))
  Z_scale[Z_scale == 0] <- 1

  train_X_std <- as.data.frame(sweep(sweep(as.matrix(train_X), 2L, X_center, "-"),
                                     2L, X_scale, "/"))
  train_Z_std <- as.data.frame(sweep(sweep(as.matrix(train_Z), 2L, Z_center, "-"),
                                     2L, Z_scale, "/"))
  val_X_std   <- sweep(sweep(as.matrix(val_X), 2L, X_center, "-"), 2L, X_scale, "/")
  val_Z_std   <- sweep(sweep(as.matrix(val_Z), 2L, Z_center, "-"), 2L, Z_scale, "/")

  y_sd <- sd(train_y)
  if (!is.finite(y_sd) || y_sd == 0) y_sd <- 1

  mu_hat        <- rep(mean(train_y), length(train_idx))
  log_sigma_hat <- rep(log(y_sd), length(train_idx))
  sigma_hat     <- exp(log_sigma_hat)

  val_mu_hat        <- rep(mean(train_y), n_val)
  val_log_sigma_hat <- rep(log(y_sd), n_val)

  fit_state <- .run_boosting_loop(
    X_std = train_X_std, Z_std = train_Z_std, y = train_y,
    mu_hat = mu_hat, sigma_hat = sigma_hat, log_sigma_hat = log_sigma_hat,
    nu_mu = nu_mu, nu_sigma = nu_sigma, method = method,
    mstop_mu = mstop_max, mstop_sigma = mstop_max,
    val_X_std = val_X_std, val_Z_std = val_Z_std, val_y = val_y,
    val_mu_hat = val_mu_hat, val_log_sigma_hat = val_log_sigma_hat,
    patience = patience
  )

  list(
    best_round = fit_state$best_round,
    val_risk0  = fit_state$val_risk0,
    val_risk   = fit_state$val_risk,
    n_train    = length(train_idx),
    n_val      = n_val
  )
}

# ── Internal: boosting loop (cyclic and noncyclic) ──────────────────────────

# Validation tracking (val_*, patience) is only used by .early_stop_search();
# when val_y is NULL the loop behaves exactly as before (no early break, no
# validation risk computed).
.run_boosting_loop <- function(X_std, Z_std, y, mu_hat, sigma_hat,
                                log_sigma_hat, nu_mu, nu_sigma, method,
                                mstop_mu, mstop_sigma,
                                val_X_std = NULL, val_Z_std = NULL,
                                val_y = NULL, val_mu_hat = NULL,
                                val_log_sigma_hat = NULL, patience = NULL) {
  pX <- ncol(X_std)
  pZ <- ncol(Z_std)
  track_val <- !is.null(val_y)

  # Fits the best base learner of u against every column of `design`.
  .best_base_learner <- function(design, u) {
    a   <- mean(u)
    best_rss <- Inf
    best_var <- NULL
    best_slope <- NULL
    best_intercept <- NULL
    for (j in seq_len(ncol(design))) {
      xj   <- design[[j]]
      ss   <- sum(xj^2)
      b    <- if (ss > 0) sum(xj * u) / ss else 0
      pred <- a + b * xj
      rss  <- sum((u - pred)^2)
      if (rss < best_rss) {
        best_rss       <- rss
        best_var       <- names(design)[j]
        best_slope     <- b
        best_intercept <- a
      }
    }
    list(var = best_var, slope = best_slope, intercept = best_intercept)
  }

  risk0 <- -sum(dnorm(y, mu_hat, sigma_hat, log = TRUE))
  if (track_val) {
    val_sigma_hat <- exp(val_log_sigma_hat)
    val_risk0     <- -sum(dnorm(val_y, val_mu_hat, val_sigma_hat, log = TRUE))
    best_val_risk <- Inf
    best_round    <- 0L
    no_improve    <- 0L
  }
  executed_rounds <- 0L

  if (method == "cyclic") {
    n_rounds <- max(mstop_mu, mstop_sigma)

    selected_mu          <- character(mstop_mu)
    coef_mu              <- numeric(mstop_mu)
    intercept_step_mu    <- numeric(mstop_mu)
    round_mu             <- integer(mstop_mu)
    selected_sigma       <- character(mstop_sigma)
    coef_sigma           <- numeric(mstop_sigma)
    intercept_step_sigma <- numeric(mstop_sigma)
    round_sigma          <- integer(mstop_sigma)

    log_submodel <- character(mstop_mu + mstop_sigma)
    log_variable <- character(mstop_mu + mstop_sigma)
    log_coef     <- numeric(mstop_mu + mstop_sigma)
    log_intercept<- numeric(mstop_mu + mstop_sigma)
    log_round    <- integer(mstop_mu + mstop_sigma)
    log_idx      <- 0L

    risk    <- numeric(n_rounds)
    val_risk <- if (track_val) numeric(n_rounds) else NULL

    for (round in seq_len(n_rounds)) {

      if (round <= mstop_mu) {
        # ── Location step ──
        u_mu     <- (y - mu_hat) / sigma_hat^2
        best     <- .best_base_learner(X_std, u_mu)
        mu_hat <- mu_hat +
          nu_mu * (best$intercept + best$slope * X_std[[best$var]])
        if (track_val) {
          val_mu_hat <- val_mu_hat +
            nu_mu * (best$intercept + best$slope * val_X_std[, best$var])
        }

        selected_mu[round]       <- best$var
        coef_mu[round]           <- nu_mu * best$slope
        intercept_step_mu[round] <- nu_mu * best$intercept
        round_mu[round]          <- round

        log_idx <- log_idx + 1L
        log_submodel[log_idx]  <- "mu"
        log_variable[log_idx]  <- best$var
        log_coef[log_idx]      <- coef_mu[round]
        log_intercept[log_idx] <- intercept_step_mu[round]
        log_round[log_idx]     <- round
      }

      if (round <= mstop_sigma) {
        # ── Scale step ──
        u_sigma     <- (y - mu_hat)^2 / sigma_hat^2 - 1
        best        <- .best_base_learner(Z_std, u_sigma)
        log_sigma_hat <- log_sigma_hat +
          nu_sigma * (best$intercept + best$slope * Z_std[[best$var]])
        sigma_hat <- exp(log_sigma_hat)
        if (track_val) {
          val_log_sigma_hat <- val_log_sigma_hat +
            nu_sigma * (best$intercept + best$slope * val_Z_std[, best$var])
        }

        selected_sigma[round]       <- best$var
        coef_sigma[round]           <- nu_sigma * best$slope
        intercept_step_sigma[round] <- nu_sigma * best$intercept
        round_sigma[round]          <- round

        log_idx <- log_idx + 1L
        log_submodel[log_idx]  <- "sigma"
        log_variable[log_idx]  <- best$var
        log_coef[log_idx]      <- coef_sigma[round]
        log_intercept[log_idx] <- intercept_step_sigma[round]
        log_round[log_idx]     <- round
      }

      risk[round]     <- -sum(dnorm(y, mu_hat, sigma_hat, log = TRUE))
      executed_rounds <- round

      if (track_val) {
        val_sigma_hat   <- exp(val_log_sigma_hat)
        val_risk[round] <- -sum(dnorm(val_y, val_mu_hat, val_sigma_hat, log = TRUE))
        if (val_risk[round] < best_val_risk) {
          best_val_risk <- val_risk[round]
          best_round    <- round
          no_improve    <- 0L
        } else {
          no_improve <- no_improve + 1L
        }
        if (no_improve >= patience) break
      }
    }

    mu_idx_final    <- min(executed_rounds, mstop_mu)
    sigma_idx_final <- min(executed_rounds, mstop_sigma)
    selected_mu          <- selected_mu[seq_len(mu_idx_final)]
    coef_mu              <- coef_mu[seq_len(mu_idx_final)]
    intercept_step_mu    <- intercept_step_mu[seq_len(mu_idx_final)]
    round_mu             <- round_mu[seq_len(mu_idx_final)]
    selected_sigma       <- selected_sigma[seq_len(sigma_idx_final)]
    coef_sigma           <- coef_sigma[seq_len(sigma_idx_final)]
    intercept_step_sigma <- intercept_step_sigma[seq_len(sigma_idx_final)]
    round_sigma          <- round_sigma[seq_len(sigma_idx_final)]

  } else {
    # ── Non-cyclic: pick whichever submodel update gives the larger
    # Gaussian log-likelihood improvement, each round. mstop_mu and
    # mstop_sigma are equal here (single total budget) and act as the round
    # cap. ──
    n_rounds <- mstop_mu  # mstop_mu == mstop_sigma == total budget

    selected_mu          <- character(0)
    coef_mu              <- numeric(0)
    intercept_step_mu    <- numeric(0)
    round_mu             <- integer(0)
    selected_sigma       <- character(0)
    coef_sigma           <- numeric(0)
    intercept_step_sigma <- numeric(0)
    round_sigma          <- integer(0)

    log_submodel <- character(n_rounds)
    log_variable <- character(n_rounds)
    log_coef     <- numeric(n_rounds)
    log_intercept<- numeric(n_rounds)
    log_round    <- integer(n_rounds)
    log_idx      <- 0L

    risk     <- numeric(n_rounds)
    val_risk <- if (track_val) numeric(n_rounds) else NULL

    for (round in seq_len(n_rounds)) {

      u_mu       <- (y - mu_hat) / sigma_hat^2
      best_mu    <- .best_base_learner(X_std, u_mu)
      mu_cand    <- mu_hat +
        nu_mu * (best_mu$intercept + best_mu$slope * X_std[[best_mu$var]])

      u_sigma        <- (y - mu_hat)^2 / sigma_hat^2 - 1
      best_sigma     <- .best_base_learner(Z_std, u_sigma)
      log_sigma_cand <- log_sigma_hat +
        nu_sigma * (best_sigma$intercept +
                      best_sigma$slope * Z_std[[best_sigma$var]])
      sigma_cand <- exp(log_sigma_cand)

      nll_current   <- -sum(dnorm(y, mu_hat,  sigma_hat,  log = TRUE))
      nll_after_mu  <- -sum(dnorm(y, mu_cand,  sigma_hat,  log = TRUE))
      nll_after_sg  <- -sum(dnorm(y, mu_hat,   sigma_cand, log = TRUE))

      improvement_mu    <- nll_current - nll_after_mu
      improvement_sigma <- nll_current - nll_after_sg

      if (improvement_mu >= improvement_sigma) {
        mu_hat <- mu_cand
        if (track_val) {
          val_mu_hat <- val_mu_hat +
            nu_mu * (best_mu$intercept + best_mu$slope * val_X_std[, best_mu$var])
        }

        selected_mu       <- c(selected_mu, best_mu$var)
        coef_mu           <- c(coef_mu, nu_mu * best_mu$slope)
        intercept_step_mu <- c(intercept_step_mu, nu_mu * best_mu$intercept)
        round_mu          <- c(round_mu, round)

        log_idx <- log_idx + 1L
        log_submodel[log_idx]  <- "mu"
        log_variable[log_idx]  <- best_mu$var
        log_coef[log_idx]      <- nu_mu * best_mu$slope
        log_intercept[log_idx] <- nu_mu * best_mu$intercept
        log_round[log_idx]     <- round
      } else {
        log_sigma_hat <- log_sigma_cand
        sigma_hat     <- sigma_cand
        if (track_val) {
          val_log_sigma_hat <- val_log_sigma_hat +
            nu_sigma * (best_sigma$intercept +
                          best_sigma$slope * val_Z_std[, best_sigma$var])
        }

        selected_sigma       <- c(selected_sigma, best_sigma$var)
        coef_sigma           <- c(coef_sigma, nu_sigma * best_sigma$slope)
        intercept_step_sigma <- c(intercept_step_sigma,
                                  nu_sigma * best_sigma$intercept)
        round_sigma          <- c(round_sigma, round)

        log_idx <- log_idx + 1L
        log_submodel[log_idx]  <- "sigma"
        log_variable[log_idx]  <- best_sigma$var
        log_coef[log_idx]      <- nu_sigma * best_sigma$slope
        log_intercept[log_idx] <- nu_sigma * best_sigma$intercept
        log_round[log_idx]     <- round
      }

      risk[round]     <- -sum(dnorm(y, mu_hat, sigma_hat, log = TRUE))
      executed_rounds <- round

      if (track_val) {
        val_sigma_hat   <- exp(val_log_sigma_hat)
        val_risk[round] <- -sum(dnorm(val_y, val_mu_hat, val_sigma_hat, log = TRUE))
        if (val_risk[round] < best_val_risk) {
          best_val_risk <- val_risk[round]
          best_round    <- round
          no_improve    <- 0L
        } else {
          no_improve <- no_improve + 1L
        }
        if (no_improve >= patience) break
      }
    }
  }

  risk <- risk[seq_len(executed_rounds)]
  if (track_val) val_risk <- val_risk[seq_len(executed_rounds)]

  step_log <- data.frame(
    round          = log_round[seq_len(log_idx)],
    submodel       = log_submodel[seq_len(log_idx)],
    variable       = log_variable[seq_len(log_idx)],
    coef           = log_coef[seq_len(log_idx)],
    intercept_step = log_intercept[seq_len(log_idx)],
    stringsAsFactors = FALSE
  )

  list(
    mu_hat = mu_hat, sigma_hat = sigma_hat, log_sigma_hat = log_sigma_hat,
    selected_mu = selected_mu, coef_mu = coef_mu,
    intercept_step_mu = intercept_step_mu, round_mu = round_mu,
    selected_sigma = selected_sigma, coef_sigma = coef_sigma,
    intercept_step_sigma = intercept_step_sigma, round_sigma = round_sigma,
    n_rounds = n_rounds, step_log = step_log,
    risk = risk, risk0 = risk0,
    best_round = if (track_val) best_round else NULL,
    val_risk   = if (track_val) val_risk else NULL,
    val_risk0  = if (track_val) val_risk0 else NULL
  )
}
