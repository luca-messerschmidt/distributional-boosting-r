# asp26boost

Component-wise gradient boosting for distributional regression in R, in the
spirit of `mboost`/`gamboostLSS`. Every distribution parameter (e.g. mean and
standard deviation) is boosted on its own linear predictor, one base learner
at a time.

## Features

- **Gaussian location-scale boosting** (`boost_gaussian()`): jointly models
  the mean and standard deviation of a continuous response.
- **Linear or nonlinear base learners**: each predictor can be fit with a
  simple linear term (`learner = "linear"`, the default) or a df-equalized
  P-spline (`learner = "spline"`), or the better of the two can be chosen
  automatically per step (`learner = "auto"`).
- **Additional response families**: `boost_gamma()` (positive, right-skewed
  responses), `boost_poisson()` (counts), `boost_binomial()` (binary
  responses) — built on the same underlying boosting engine.
- **Cross-validation**: `cv_boost_gaussian()` and `cv_boost_grid()` select
  the number of boosting iterations (and, optionally, the base learner type)
  via k-fold cross-validation.
- `predict()`, `print()`, `summary()`, and `plot()` methods for all fitted
  models, including partial-effect plots for nonlinear terms.

## Installation

This package is not on CRAN. Install it directly from this repository:

```r
# install.packages("remotes")
remotes::install_github("luca-messerschmidt/distributional-boosting-r")
```

Or, if you already have the repository cloned locally:

```r
# install.packages("devtools")
devtools::install(".")
```

## Quick start

```r
library(asp26boost)

set.seed(42)
n <- 200
X <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
Z <- data.frame(z1 = rnorm(n))
y <- 2 + 3 * X$x1 + exp(0.5 * Z$z1) * rnorm(n)

fit <- boost_gaussian(X, Z, y, mstop = 100, nu_mu = 0.1, nu_sigma = 0.1)
print(fit)
predict(fit, newdata_X = X[1:5, ], newdata_Z = Z[1:5, ])
```

### Nonlinear (P-spline) base learners

```r
y_nonlin <- 2 + 3 * sin(2 * X$x1) + exp(0.4 * Z$z1) * rnorm(n)
fit_auto <- boost_gaussian(X, Z, y_nonlin, mstop = 100, learner = "auto")

summary(fit_auto)
plot(fit_auto, type = "partial", submodel = "mu", variable = "x1")
```

### Other distribution families

```r
fit_pois <- boost_poisson(X, y = rpois(n, lambda = exp(0.5 + 0.4 * X$x1)),
                           mstop = 100)
fit_bin  <- boost_binomial(X, y = rbinom(n, 1, 0.5), mstop = 100)
```

### Cross-validation

```r
cv_fit <- cv_boost_gaussian(X, Z, y, k = 5, mstop = 100)
print(cv_fit)
```

See `vignette("boost_gaussian", package = "asp26boost")` for a fuller
walkthrough, and `?boost_gaussian`, `?boost_gamma`, `?boost_poisson`,
`?boost_binomial`, `?cv_boost_gaussian` for detailed documentation.

## Testing

```r
devtools::test()
```

## Project report

The accompanying term paper describes the statistical background, package
architecture, and simulation study in detail:

- [Component-wise Gradient Boosting for Distributional Regression in R](docs/component-wise-gradient-boosting-report.pdf)

The public copy omits access credentials for supporting material and the
signed declaration of originality.

## Project background

This package was developed collaboratively in 2026 for the Advanced
Statistical Programming with R course at the University of Göttingen. This
GitHub repository is a portfolio mirror of the
[original GWDG GitLab project](https://gitlab.gwdg.de/bjarne.herbst01/asp26boost),
with the complete commit history retained to preserve authorship.

Luca Messerschmidt's contributions included work on the Gaussian
location-scale implementation and its model methods, the generalized boosting
engine and family abstraction, Gamma, Poisson, and Binomial response families,
P-spline base learners, cross-validation, automated tests, and package
documentation.

## License

MIT — see `LICENSE.md`.

## Authors

- Bjarne Herbst
- Benedict Schnitzler
- Luca Messerschmidt
