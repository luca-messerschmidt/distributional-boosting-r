# ── Family objects for distributional gradient boosting ─────────────────────
# A family is a plain list, one parameter of the distribution boosted per
# link-scale linear predictor:
#   name, parameters (character vector, in update order)
#   link/invlink   - response scale <-> link scale, per parameter
#   ngradient      - function(y, par), d(loglik)/d(eta_k), fit each round
#   risk           - function(y, par), total negative log-likelihood
#   init           - function(y), starting values per parameter
# `par` is always a named list of response-scale values, e.g.
# list(mu = ..., sigma = ...) for a 2-parameter family.

.sd_safe <- function(y) {
  s <- stats::sd(y)
  if (!is.finite(s) || s == 0) 1 else s
}

.family_gaussian <- function() {
  list(
    name       = "gaussian",
    parameters = c("mu", "sigma"),
    link       = list(mu = identity, sigma = log),
    invlink    = list(mu = identity, sigma = exp),
    ngradient  = list(
      mu    = function(y, par) (y - par$mu) / par$sigma^2,
      sigma = function(y, par) (y - par$mu)^2 / par$sigma^2 - 1
    ),
    risk = function(y, par) {
      -sum(stats::dnorm(y, par$mu, par$sigma, log = TRUE))
    },
    init = function(y) {
      n <- length(y)
      list(mu = rep(mean(y), n), sigma = rep(.sd_safe(y), n))
    }
  )
}

# Gamma location-scale: mu = mean (log-linked), shape = shape (log-linked),
# parameterized so that Var(y) = mu^2 / shape (R's dgamma(shape, rate) has
# mean = shape/rate, so rate = shape/mu gives mean = mu).
#
# log f(y) = shape*log(shape) - shape*log(mu) - lgamma(shape)
#            + (shape - 1)*log(y) - shape*y/mu
# d(loglik)/d(mu)        = (shape/mu) * (y/mu - 1)
# d(loglik)/d(eta_mu)    = mu * d(loglik)/d(mu) = shape * (y/mu - 1)         [eta_mu = log(mu)]
# d(loglik)/d(shape)     = log(shape) + 1 - digamma(shape) - log(mu) + log(y) - y/mu
# d(loglik)/d(eta_shape) = shape * d(loglik)/d(shape)                        [eta_shape = log(shape)]
# Requires y > 0. `mu` is clipped away from 0 before risk is computed to
# avoid a non-finite log-density.
.family_gamma <- function() {
  eps <- 1e-10
  list(
    name       = "gamma",
    parameters = c("mu", "shape"),
    link       = list(mu = log, shape = log),
    invlink    = list(mu = exp, shape = exp),
    ngradient  = list(
      mu    = function(y, par) par$shape * (y / par$mu - 1),
      shape = function(y, par) {
        par$shape * (log(par$shape) + 1 - digamma(par$shape) -
                        log(par$mu) + log(y) - y / par$mu)
      }
    ),
    risk = function(y, par) {
      mu <- pmax(par$mu, eps)
      -sum(stats::dgamma(y, shape = par$shape, rate = par$shape / mu,
                          log = TRUE))
    },
    init = function(y) {
      n    <- length(y)
      m    <- mean(y)
      v    <- stats::var(y)
      shape0 <- if (is.finite(v) && v > 0) m^2 / v else 1
      if (!is.finite(shape0) || shape0 <= 0) shape0 <- 1
      list(mu = rep(m, n), shape = rep(shape0, n))
    }
  )
}

# Poisson: mu = mean (log-linked). Canonical link, so the negative gradient
# takes the classic clean form y - mu (matches mboost's Poisson() family).
# log f(y) = y*log(mu) - mu - log(y!)
# d(loglik)/d(mu) = y/mu - 1; d(mu)/d(eta) = mu (eta = log(mu))
# d(loglik)/d(eta) = mu*(y/mu - 1) = y - mu
# Requires y >= 0. `mu` is clipped away from 0 before risk is computed to
# avoid a non-finite log-density.
.family_poisson <- function() {
  eps <- 1e-10
  list(
    name       = "poisson",
    parameters = c("mu"),
    link       = list(mu = log),
    invlink    = list(mu = exp),
    ngradient  = list(mu = function(y, par) y - par$mu),
    risk = function(y, par) {
      mu <- pmax(par$mu, eps)
      -sum(stats::dpois(y, lambda = mu, log = TRUE))
    },
    init = function(y) {
      n <- length(y)
      m <- mean(y)
      if (!is.finite(m) || m <= 0) m <- 1
      list(mu = rep(m, n))
    }
  )
}

# Binomial/Bernoulli: mu = P(y = 1) (logit-linked). Canonical link, same
# clean negative-gradient form as Poisson (matches mboost's Binomial()).
# log f(y) = y*log(p) + (1-y)*log(1-p); d(p)/d(eta) = p*(1-p) (eta = logit(p))
# d(loglik)/d(eta) = p(1-p)*(y/p - (1-y)/(1-p)) = y - p
# Requires y in {0, 1}. `mu` is clipped away from the boundary before risk is
# computed to avoid -Inf in the log-density.
.family_binomial <- function() {
  eps <- 1e-10
  list(
    name       = "binomial",
    parameters = c("mu"),
    link       = list(mu = stats::qlogis),
    invlink    = list(mu = stats::plogis),
    ngradient  = list(mu = function(y, par) y - par$mu),
    risk = function(y, par) {
      p <- pmin(pmax(par$mu, eps), 1 - eps)
      -sum(stats::dbinom(y, size = 1, prob = p, log = TRUE))
    },
    init = function(y) {
      n <- length(y)
      m <- mean(y)
      m <- min(max(m, eps), 1 - eps)
      list(mu = rep(m, n))
    }
  )
}

# Internal sanity check, called defensively at the top of the generalized
# boosting loop.
.validate_family <- function(family) {
  if (!is.list(family) || is.null(family$parameters) ||
      !is.character(family$parameters) || length(family$parameters) < 1L) {
    stop("family$parameters must be a non-empty character vector.", call. = FALSE)
  }
  P <- family$parameters
  for (comp in c("link", "invlink", "ngradient")) {
    x <- family[[comp]]
    if (!is.list(x) || !setequal(names(x), P)) {
      stop(sprintf(
        "family$%s must be a named list covering exactly family$parameters.",
        comp
      ), call. = FALSE)
    }
  }
  if (!is.function(family$risk)) {
    stop("family$risk must be a function.", call. = FALSE)
  }
  if (!is.function(family$init)) {
    stop("family$init must be a function.", call. = FALSE)
  }
  invisible(TRUE)
}
