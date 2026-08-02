# tests/testthat/test-families.R
#
# Each family's ngradient[[k]] must equal d(loglik)/d(eta_k), verified via
# central-difference numerical differentiation of -risk() w.r.t. the
# link-scale linear predictor. This is the single most important
# correctness check for the multi-family feature: a sign or
# parameterization error here would silently converge to the wrong model.

.numeric_dloglik_deta <- function(family, y, par, k, h = 1e-5) {
  eta_k <- family$link[[k]](par[[k]])
  par_plus  <- par; par_plus[[k]]  <- family$invlink[[k]](eta_k + h)
  par_minus <- par; par_minus[[k]] <- family$invlink[[k]](eta_k - h)
  (-family$risk(y, par_plus) - -family$risk(y, par_minus)) / (2 * h)
}

test_that("family gradients match numerical differentiation of the log-likelihood", {
  set.seed(1)
  n <- 30
  cases <- list(
    gaussian = list(family = .family_gaussian(), y = rnorm(n, 2, 1.5),
                    par = list(mu = rep(2.3, n), sigma = rep(1.7, n))),
    gamma    = list(family = .family_gamma(), y = rgamma(n, shape = 3, rate = 3 / 2.5),
                    par = list(mu = rep(2.5, n), shape = rep(3, n))),
    poisson  = list(family = .family_poisson(), y = rpois(n, 4),
                    par = list(mu = rep(4.2, n))),
    binomial = list(family = .family_binomial(), y = rbinom(n, 1, 0.35),
                    par = list(mu = rep(0.4, n)))
  )

  for (nm in names(cases)) {
    cs <- cases[[nm]]
    for (k in cs$family$parameters) {
      analytic <- sum(cs$family$ngradient[[k]](cs$y, cs$par))
      numeric_ <- .numeric_dloglik_deta(cs$family, cs$y, cs$par, k)
      expect_equal(analytic, numeric_, tolerance = 1e-4, info = paste(nm, k))
    }
    expect_true(is.finite(cs$family$risk(cs$y, cs$par)), info = nm)
  }
})

test_that("family init() returns sensible, in-domain starting values", {
  set.seed(2)
  n <- 25

  init_g <- .family_gaussian()$init(rnorm(n, 3, 2))
  expect_true(all(init_g$sigma > 0))

  init_ga <- .family_gamma()$init(rgamma(n, shape = 4, rate = 2))
  expect_true(all(init_ga$mu > 0) && all(init_ga$shape > 0))

  init_p <- .family_poisson()$init(rpois(n, 5))
  expect_true(all(init_p$mu > 0))

  init_b <- .family_binomial()$init(rbinom(n, 1, 0.6))
  expect_true(all(init_b$mu > 0 & init_b$mu < 1))
})

test_that(".validate_family() rejects malformed family objects", {
  expect_error(.validate_family(list(parameters = character(0))))
  expect_error(.validate_family(list(
    parameters = "mu", link = list(), invlink = list(mu = exp),
    ngradient = list(mu = function(y, par) y), risk = function(y, par) 0,
    init = function(y) list(mu = 0)
  )))
  expect_true(.validate_family(.family_gaussian()))
})
