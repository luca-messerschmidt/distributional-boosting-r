# tests/testthat/helper-fixtures.R

make_boost_data <- function(n = 80, pX = 3, pZ = 3, seed = 42, noise_sd = 0.5) {
  set.seed(seed)
  
  X <- as.data.frame(matrix(rnorm(n * pX), nrow = n, ncol = pX))
  names(X) <- paste0("x", seq_len(pX))
  
  Z <- as.data.frame(matrix(rnorm(n * pZ), nrow = n, ncol = pZ))
  names(Z) <- paste0("z", seq_len(pZ))
  
  # True model: mu = 2 + 3*x1 (and -1.5*x3 if pX >= 3), sigma = exp(0.4*z1)
  mu_true <- 2 + 3 * X$x1
  if (pX >= 3) {
    mu_true <- mu_true - 1.5 * X$x3
  }
  sigma_true <- exp(0.4 * Z$z1)
  
  y <- mu_true + sigma_true * rnorm(n)
  
  list(
    X_df  = X,
    X_mat = as.matrix(X),
    Z_df  = Z,
    Z_mat = as.matrix(Z),
    y     = y,
    n     = n,
    pX    = pX,
    pZ    = pZ
  )
}

make_boost_fit <- function(n      = 80,
                           pX     = 3,
                           pZ     = 3,
                           seed   = 42,
                           mstop  = 50,
                           nu_mu  = 0.1,
                           nu_sigma = 0.1) {
  dat <- make_boost_data(n = n, pX = pX, pZ = pZ, seed = seed)
  fit <- boost_gaussian(
    X        = dat$X_df,
    Z        = dat$Z_df,
    y        = dat$y,
    mstop    = mstop,
    nu_mu    = nu_mu,
    nu_sigma = nu_sigma
  )
  
  list(
    fit   = fit,
    X_df  = dat$X_df,
    X_mat = dat$X_mat,
    Z_df  = dat$Z_df,
    Z_mat = dat$Z_mat,
    y     = dat$y,
    n     = dat$n,
    pX    = dat$pX,
    pZ    = dat$pZ
  )
}