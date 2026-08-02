# ── Demo script for asp26boost (distributional gradient boosting) ────────────
# Run interactively; each section is self-contained.

devtools::load_all()
set.seed(42)
n <- 200

# ── 1. Gaussian location-scale, linear base learner (the original model) ─────
X <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
Z <- data.frame(z1 = rnorm(n), z2 = rnorm(n))
y_gauss <- 2 + 3 * X$x1 - 1.5 * X$x2 + exp(0.5 * Z$z1) * rnorm(n)

fit <- boost_gaussian(X, Z, y_gauss, mstop = 150)
print(fit)
print(summary(fit))
predict(fit, newdata_X = X[1:3, ], newdata_Z = Z[1:3, ], what = "both")

par(mfrow = c(2, 2))
plot(fit, type = "path", submodel = "mu")
plot(fit, type = "frequency", submodel = "mu")
plot(fit, type = "risk")
par(mfrow = c(1, 1))

# ── 2. P-spline base learner: learner = "spline"/"auto" ───────────────────────
# x1 enters nonlinearly here; "auto" tries both a linear and a spline
# candidate per column and keeps whichever fits the working response better.
y_nonlin <- 2 + 3 * sin(2 * X$x1) - X$x2 + exp(0.4 * Z$z1) * rnorm(n)

fit_linear <- boost_gaussian(X, Z, y_nonlin, mstop = 100, learner = "linear")
fit_auto   <- boost_gaussian(X, Z, y_nonlin, mstop = 100, learner = "auto")
cat("Final risk, linear-only  :", tail(fit_linear$risk, 1), "\n")
cat("Final risk, linear+spline:", tail(fit_auto$risk, 1), "\n")

# A spline-fit variable no longer has a single coefficient: summary() reports
# it as a "smooth term" instead, and its fitted (possibly nonlinear) effect
# is visualised with plot(type = "partial").
print(summary(fit_auto))
plot(fit_auto, type = "partial", submodel = "mu", variable = "x1")

# ── 3. Other distribution families (same engine, family-specific link) ───────
# Gamma location-scale: positive, right-skewed response.
mu_g <- exp(1 + 0.5 * X$x1); shape_g <- exp(0.4 * Z$z1 + 1)
y_gamma <- rgamma(n, shape = shape_g, rate = shape_g / mu_g)
fit_gamma <- boost_gamma(X, Z, y_gamma, mstop = 100)
print(fit_gamma)

# Poisson: count response (single parameter).
y_pois <- rpois(n, lambda = exp(0.5 + 0.4 * X$x1))
fit_pois <- boost_poisson(X, y_pois, mstop = 100)
print(fit_pois)
predict(fit_pois, newdata_X = X[1:3, ])

# Binomial/Bernoulli: binary response (single parameter).
y_bin <- rbinom(n, 1, plogis(0.3 + 0.8 * X$x1))
fit_bin <- boost_binomial(X, y_bin, mstop = 100)
print(fit_bin)
print(summary(fit_bin))
plot(fit_bin, type = "path")

# ── 4. Cross-validation ─────────────────────────
cv_fit <- cv_boost_gaussian(X, Z, y_gauss, k = 5, mstop = 100)
print(cv_fit)

# ── 5. Help pages ──────────────────────────────────────────────────────────────
?boost_gaussian
?boost_gamma
?boost_poisson
?boost_binomial
?predict.boost_dist
?plot.boost_gaussian
?summary.boost_gaussian

