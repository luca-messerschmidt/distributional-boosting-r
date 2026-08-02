# ── Shared internal engine for boost_gamma()/boost_poisson()/boost_binomial() ──
#
# boost_gaussian() keeps its own hand-written body (untouched beyond
# delegating to .run_boosting_loop_general(), so its output stays exactly
# what it was before). The three new distributional entry points share this
# engine instead of duplicating boost_gaussian()'s validation/standardisation/
# post-fit logic three times; each wrapper only does its own family-specific
# y-domain validation and argument assembly before calling .boost_dist_fit().

# Generalizes .parse_mstop() (boost_gaussian.R) to an arbitrary number of
# named parameters. Returns a named list of integer mstops keyed by
# `param_names`.
.parse_mstop_general <- function(mstop, method, param_names) {
  err <- function() {
    stop("mstop must be a single positive integer, or (for method = ",
         "'cyclic') a named vector/list with elements matching: ",
         paste(param_names, collapse = ", "), ".", call. = FALSE)
  }

  if (is.list(mstop)) mstop <- unlist(mstop)
  if (!is.numeric(mstop) || length(mstop) == 0L) err()

  if (length(mstop) == 1L) {
    if (is.na(mstop) || !is.finite(mstop) || mstop < 1L ||
        mstop != as.integer(mstop)) {
      err()
    }
    m <- as.integer(mstop)
    return(stats::setNames(as.list(rep(m, length(param_names))), param_names))
  }

  if (length(mstop) == length(param_names) && !is.null(names(mstop)) &&
      setequal(names(mstop), param_names)) {
    if (method != "cyclic") {
      stop("Separate per-parameter mstop values are only supported for ",
           "method = 'cyclic'.", call. = FALSE)
    }
    if (anyNA(mstop) || any(!is.finite(mstop)) || any(mstop < 1L) ||
        any(mstop != as.integer(mstop))) {
      err()
    }
    return(stats::setNames(as.list(as.integer(mstop[param_names])), param_names))
  }

  err()
}

# Generalizes .build_design_from_formula() (boost_gaussian.R) to families
# with any number of parameters. For a single-parameter family, `formula`
# must be a plain two-sided formula (e.g. y ~ x1 + x2). For a two-parameter
# family, `formula` must be a list with one component per parameter, the
# first two-sided (establishing y) and the rest one-sided (e.g.
# list(mu = y ~ x1, shape = ~ z1)).
.build_design_from_formula_general <- function(formula, data, param_names) {
  if (is.null(data) || !is.data.frame(data)) {
    stop("data must be a data.frame when formula is supplied.", call. = FALSE)
  }

  if (length(param_names) == 1L) {
    f <- formula
    if (!inherits(f, "formula") || length(f) != 3L) {
      stop("formula must be a two-sided formula, e.g. y ~ x1 + x2.", call. = FALSE)
    }
    terms_obj <- stats::terms(f)
    mf <- stats::model.frame(terms_obj, data = data)
    y  <- mf[[1L]]
    X  <- stats::model.matrix(terms_obj, data = mf)
    X  <- X[, colnames(X) != "(Intercept)", drop = FALSE]

    designs <- stats::setNames(list(as.data.frame(X)), param_names)
    terms_list <- stats::setNames(list(terms_obj), param_names)
    return(list(designs = designs, y = as.numeric(y), terms = terms_list))
  }

  if (!is.list(formula) || !setequal(names(formula), param_names)) {
    stop(sprintf("formula must be a list with components '%s'.",
                 paste(param_names, collapse = "', '")), call. = FALSE)
  }

  f1 <- formula[[param_names[1L]]]
  if (!inherits(f1, "formula") || length(f1) != 3L) {
    stop(sprintf("formula$%s must be a two-sided formula, e.g. y ~ x1 + x2.",
                 param_names[1L]), call. = FALSE)
  }
  terms1 <- stats::terms(f1)
  mf1    <- stats::model.frame(terms1, data = data)
  y      <- mf1[[1L]]
  X1     <- stats::model.matrix(terms1, data = mf1)
  X1     <- X1[, colnames(X1) != "(Intercept)", drop = FALSE]

  designs    <- stats::setNames(vector("list", length(param_names)), param_names)
  terms_list <- stats::setNames(vector("list", length(param_names)), param_names)
  designs[[param_names[1L]]]    <- as.data.frame(X1)
  terms_list[[param_names[1L]]] <- terms1

  for (k in param_names[-1L]) {
    fk <- formula[[k]]
    if (!inherits(fk, "formula")) {
      stop(sprintf("formula$%s must be a formula, e.g. ~ z1 + z2.", k), call. = FALSE)
    }
    termsk <- stats::terms(fk)
    mfk    <- stats::model.frame(termsk, data = data)
    Xk     <- stats::model.matrix(termsk, data = mfk)
    Xk     <- Xk[, colnames(Xk) != "(Intercept)", drop = FALSE]
    designs[[k]]    <- as.data.frame(Xk)
    terms_list[[k]] <- termsk
  }

  list(designs = designs, y = as.numeric(y), terms = terms_list)
}

# Generalizes .early_stop_search() (boost_gaussian.R) to any family/parameter
# count, reusing .run_boosting_loop_general()'s validation-tracking support.
.early_stop_search_general <- function(designs, y, family, nus, mstop_max,
                                        method, validation_split, patience,
                                        seed, learner) {
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

  P <- family$parameters
  n <- length(y)
  if (n < 3L) {
    stop("Need at least 3 observations to use internal early stopping.",
         call. = FALSE)
  }
  if (!is.null(seed)) set.seed(seed)

  n_val <- max(1L, round(validation_split * n))
  if (n_val >= n) n_val <- n - 1L
  val_idx   <- sample.int(n, n_val)
  train_idx <- setdiff(seq_len(n), val_idx)

  train_designs <- val_designs_std <- stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    train_df <- designs[[k]][train_idx, , drop = FALSE]
    val_df   <- designs[[k]][val_idx, , drop = FALSE]
    ctr <- vapply(train_df, mean, numeric(1))
    scl <- vapply(train_df, sd, numeric(1))
    scl[scl == 0] <- 1
    train_designs[[k]]   <- as.data.frame(sweep(sweep(as.matrix(train_df), 2L, ctr, "-"),
                                                2L, scl, "/"))
    val_designs_std[[k]] <- sweep(sweep(as.matrix(val_df), 2L, ctr, "-"), 2L, scl, "/")
  }
  train_y <- y[train_idx]
  val_y   <- y[val_idx]

  fit_state <- .run_boosting_loop_general(
    designs = train_designs, y = train_y, family = family, nus = nus,
    mstops  = stats::setNames(as.list(rep(mstop_max, length(P))), P),
    method  = method, val_designs = val_designs_std, val_y = val_y,
    patience = patience, learner = learner
  )

  list(best_round = fit_state$best_round, val_risk0 = fit_state$val_risk0,
       val_risk = fit_state$val_risk, n_train = length(train_idx),
       n_val = n_val, mstop_max = mstop_max)
}

# The shared fitting engine: standardises `designs`, optionally runs the
# early-stopping search, fits the generalized boosting loop, and aggregates
# net coefficients per parameter (via .legacy_net_coef(), reused from
# boosting_loop_general.R). `designs`/`nus` are named lists keyed by
# `family$parameters`.
.boost_dist_fit <- function(designs, y, family, nus, mstop,
                            method = c("cyclic", "noncyclic"),
                            patience = NULL, validation_split = 0.2,
                            seed = NULL, learner = c("linear", "spline", "auto")) {
  method  <- match.arg(method)
  learner <- match.arg(learner)
  P <- family$parameters
  n <- length(y)

  for (k in P) {
    if (!is.matrix(designs[[k]]) && !is.data.frame(designs[[k]])) {
      stop(sprintf("The '%s' design must be a matrix or data.frame.", k), call. = FALSE)
    }
    designs[[k]] <- as.data.frame(designs[[k]])
    if (nrow(designs[[k]]) != n) {
      stop(sprintf("Number of rows in the '%s' design must match length of y.", k),
           call. = FALSE)
    }
    if (!all(vapply(designs[[k]], is.numeric, logical(1)))) {
      stop(sprintf("All columns of the '%s' design must be numeric.", k), call. = FALSE)
    }
    if (anyNA(designs[[k]])) {
      stop(sprintf("The '%s' design must not contain missing (NA/NaN) values.", k),
           call. = FALSE)
    }
    if (!all(is.finite(as.matrix(designs[[k]])))) {
      stop(sprintf("The '%s' design must not contain infinite values.", k), call. = FALSE)
    }
  }
  if (anyNA(y)) stop("y must not contain missing (NA/NaN) values.", call. = FALSE)
  if (!all(is.finite(y))) stop("y must not contain infinite values.", call. = FALSE)

  for (k in P) {
    nuk <- nus[[k]]
    if (!is.numeric(nuk) || length(nuk) != 1L || is.na(nuk) ||
        !is.finite(nuk) || nuk <= 0 || nuk > 1) {
      stop(sprintf("The learning rate for '%s' must be a single finite value in (0, 1].", k),
           call. = FALSE)
    }
  }

  mstop_parsed <- .parse_mstop_general(mstop, method, P)

  if (!is.null(patience) && length(unique(unlist(mstop_parsed))) != 1L) {
    stop("patience (internal early stopping) requires a single scalar mstop; ",
         "separate per-parameter mstop values are not supported together ",
         "with early stopping.", call. = FALSE)
  }

  centers <- scales <- stds <- stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    ctr <- vapply(designs[[k]], mean, numeric(1))
    scl <- vapply(designs[[k]], sd, numeric(1))
    scl[scl == 0] <- 1
    centers[[k]] <- ctr
    scales[[k]]  <- scl
    stds[[k]] <- as.data.frame(sweep(sweep(as.matrix(designs[[k]]), 2L, ctr, "-"),
                                     2L, scl, "/"))
  }

  early_stopping_info <- NULL
  if (!is.null(patience)) {
    search <- .early_stop_search_general(
      designs = designs, y = y, family = family, nus = nus,
      mstop_max = mstop_parsed[[P[1L]]], method = method,
      validation_split = validation_split, patience = patience, seed = seed,
      learner = learner
    )
    for (k in P) mstop_parsed[[k]] <- search$best_round
    early_stopping_info <- list(
      used = TRUE, patience = as.integer(patience),
      validation_split = validation_split, seed = seed,
      mstop_max = search$mstop_max, best_round = search$best_round,
      n_train = search$n_train, n_val = search$n_val,
      val_risk0 = search$val_risk0, val_risk = search$val_risk
    )
  }

  fit_state <- .run_boosting_loop_general(
    designs = stds, y = y, family = family, nus = nus,
    mstops = mstop_parsed, method = method, learner = learner
  )

  init_vals <- family$init(y)
  net_std <- net_orig <- intercept <- init_eta <-
    stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    agg <- .legacy_net_coef(fit_state$coef[[k]], fit_state$intercept_step[[k]],
                            fit_state$selected[[k]], names(stds[[k]]))
    net_std[[k]]   <- agg$net_std
    net_orig[[k]]  <- agg$net_std / scales[[k]]
    init_eta[[k]]  <- family$link[[k]](init_vals[[k]][1L])
    intercept[[k]] <- init_eta[[k]] + agg$total_intercept_step -
      sum(net_orig[[k]] * centers[[k]], na.rm = TRUE)
  }

  list(
    family = family, parameters = P, n = n,
    fit_state = fit_state,
    designs_std = stds, centers = centers, scales = scales,
    design_names = stats::setNames(lapply(P, function(k) names(designs[[k]])), P),
    net_coef_std = net_std, net_coef_orig = net_orig, intercept = intercept,
    init_eta = init_eta,
    mstop_parsed = mstop_parsed, method = method, learner = learner, nus = nus,
    early_stopping = early_stopping_info
  )
}

# Assembles the public result list returned by boost_gamma()/boost_poisson()/
# boost_binomial() from a `.boost_dist_fit()` core, in a family-agnostic
# shape (named lists keyed by family$parameters throughout) that
# predict.boost_dist()/print.boost_dist()/summary.boost_dist()/
# plot.boost_dist() (added in a later phase) can consume without knowing how
# many parameters the family has.
.assemble_boost_dist_result <- function(core, terms_list, call) {
  family <- core$family
  P      <- core$parameters
  fs     <- core$fit_state

  fitted <- stats::setNames(lapply(P, function(k) as.numeric(fs$par[[k]])), P)
  eta    <- stats::setNames(lapply(P, function(k) as.numeric(fs$eta[[k]])), P)

  list(
    family_name    = family$name,
    parameters     = P,
    fitted         = fitted,
    eta            = eta,
    selected       = fs$selected,
    coef           = fs$coef,
    intercept_step = fs$intercept_step,
    round          = fs$round,
    net_coef_std   = core$net_coef_std,
    net_coef_orig  = core$net_coef_orig,
    intercept      = core$intercept,
    init_eta       = core$init_eta,
    residuals      = NULL,
    method         = core$method,
    learner        = core$learner,
    mstop          = fs$n_rounds,
    mstop_per_param = core$mstop_parsed,
    nus            = core$nus,
    step_log       = fs$step_log,
    risk           = fs$risk,
    risk0          = fs$risk0,
    early_stopping = core$early_stopping,
    n              = core$n,
    p              = stats::setNames(vapply(P, function(k) ncol(core$designs_std[[k]]), integer(1)), P),
    design_names   = core$design_names,
    centers        = core$centers,
    scales         = core$scales,
    designs_std    = stats::setNames(lapply(P, function(k) as.matrix(core$designs_std[[k]])), P),
    terms          = terms_list,
    family         = family,
    call           = call
  )
}
