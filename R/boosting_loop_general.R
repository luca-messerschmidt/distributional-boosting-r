# ── Generalized component-wise base learner selection ────────────────────────
# Fits candidate base learners to every column of `design` against target
# `u` and returns the (column, learner-type) with the lowest RSS. `learner`
# picks which candidates are tried per column:
#   "linear" - one linear (intercept + slope) candidate
#   "spline" - one df-equalized P-spline candidate (needs spline_cache_k,
#              built by .build_spline_cache())
#   "auto"   - both, true RSS-best-of-both
# `fitted` in the returned list is the candidate's raw prediction (before
# scaling by nu), used to update the linear predictor.
.best_base_learner_general <- function(design, u, learner = "linear",
                                        spline_cache_k = NULL) {
  best_rss <- Inf
  best     <- NULL

  if (learner %in% c("linear", "auto")) {
    a <- mean(u)
    for (j in seq_len(ncol(design))) {
      xj   <- design[[j]]
      ss   <- sum(xj^2)
      b    <- if (ss > 0) sum(xj * u) / ss else 0
      pred <- a + b * xj
      rss  <- sum((u - pred)^2)
      if (rss < best_rss) {
        best_rss <- rss
        best <- list(var = names(design)[j], type = "linear",
                     slope = b, intercept = a, fitted = pred)
      }
    }
  }

  if (learner %in% c("spline", "auto")) {
    for (j in seq_len(ncol(design))) {
      var_name <- names(design)[j]
      sc       <- spline_cache_k[[var_name]]
      coefs    <- as.numeric(solve(sc$BtB + sc$lambda * sc$K,
                                   crossprod(sc$basis$B, u)))
      fitted_j <- as.numeric(sc$basis$B %*% coefs)
      rss      <- sum((u - fitted_j)^2)
      if (rss < best_rss) {
        best_rss <- rss
        best <- list(var = var_name, type = "spline", coefs = coefs,
                     knots = sc$basis$knots, boundary = sc$basis$boundary,
                     degree = sc$basis$degree, edf = sc$edf, lambda = sc$lambda,
                     fitted = fitted_j)
      }
    }
  }

  best
}

# Bakes `nu` into a candidate's stored slope/intercept (linear) or basis
# coefficients (spline), so predict() can just apply the stored numbers
# without separate nu bookkeeping.
.scale_step <- function(best, nu) {
  if (best$type == "linear") {
    list(type = "linear", var = best$var,
         slope = nu * best$slope, intercept = nu * best$intercept)
  } else {
    list(type = "spline", var = best$var, coefs = nu * best$coefs,
         knots = best$knots, boundary = best$boundary, degree = best$degree,
         edf = best$edf, lambda = best$lambda)
  }
}

# Aggregates one parameter's per-step history into a named net-coefficient
# vector (one entry per variable) plus the total intercept drift. Variables
# that ever received a spline step get NA instead of a slope sum, since a
# single scalar can't describe a smooth effect (see .smooth_terms_table()).
.legacy_net_coef <- function(coef_k, intercept_step_k, selected_k, var_names) {
  if (!is.list(coef_k)) {
    net_std <- tapply(coef_k, selected_k, sum)
    net_std <- net_std[var_names]
    names(net_std) <- var_names
    net_std[is.na(net_std)] <- 0
    return(list(net_std = net_std,
                total_intercept_step = sum(intercept_step_k),
                smooth_vars = character(0)))
  }

  slope_sum   <- stats::setNames(rep(0, length(var_names)), var_names)
  smooth_vars <- character(0)
  total_intercept <- 0
  for (s in coef_k) {
    if (s$type == "linear") {
      slope_sum[s$var] <- slope_sum[s$var] + s$slope
      total_intercept  <- total_intercept + s$intercept
    } else {
      smooth_vars <- union(smooth_vars, s$var)
    }
  }
  if (length(smooth_vars) > 0L) slope_sum[smooth_vars] <- NA_real_

  list(net_std = slope_sum, total_intercept_step = total_intercept,
       smooth_vars = smooth_vars)
}

# ── Generalized boosting loop, driven by a family object ────────────────────
# designs/nus/mstops are named lists keyed by family$parameters (e.g.
# list(mu = X_std, sigma = Z_std) for 2 parameters, list(mu = X_std) for 1).
# method = "cyclic" updates every parameter each round until its own
# mstops[[k]] budget is used up; "noncyclic" updates only whichever
# parameter's candidate step gives the largest risk decrease. Validation
# tracking (val_designs + val_y) enables early stopping via `patience`.
.run_boosting_loop_general <- function(designs, y, family, nus, mstops, method,
                                        val_designs = NULL, val_y = NULL,
                                        patience = NULL, learner = "linear") {
  .validate_family(family)
  P <- family$parameters
  track_val <- !is.null(val_y)

  init <- family$init(y)
  eta  <- stats::setNames(lapply(P, function(k) family$link[[k]](init[[k]])), P)
  par  <- stats::setNames(lapply(P, function(k) family$invlink[[k]](eta[[k]])), P)

  if (track_val) {
    n_val   <- length(val_y)
    val_eta <- stats::setNames(lapply(P, function(k) rep(eta[[k]][1], n_val)), P)
    val_par <- stats::setNames(lapply(P, function(k) family$invlink[[k]](val_eta[[k]])), P)
  }

  # only built when a spline candidate is actually usable, so learner =
  # "linear" pays nothing extra
  spline_cache <- stats::setNames(vector("list", length(P)), P)
  if (learner %in% c("spline", "auto")) {
    for (k in P) spline_cache[[k]] <- .build_spline_cache(designs[[k]])
  }

  # steps[[k]] holds each step as a step object (.scale_step()); collapsed
  # back into flat coef/intercept_step vectors at the end when every step
  # for a parameter is linear.
  selected <- roundv <- steps <-
    stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    selected[[k]] <- character(0)
    roundv[[k]]   <- integer(0)
    steps[[k]]    <- list()
  }

  risk0 <- family$risk(y, par)
  if (track_val) {
    val_risk0     <- family$risk(val_y, val_par)
    best_val_risk <- Inf
    best_round    <- 0L
    no_improve    <- 0L
  }
  executed_rounds <- 0L

  log_submodel  <- character(0)
  log_variable  <- character(0)
  log_coef      <- numeric(0)
  log_intercept <- numeric(0)
  log_round     <- integer(0)

  # Fits and applies one base-learner step for parameter `k`, updating
  # eta/par/selected/steps/roundv (and validation accumulators, if tracked)
  # via <<- into the enclosing environment.
  .apply_step <- function(k, round) {
    u    <- family$ngradient[[k]](y, par)
    best <- .best_base_learner_general(designs[[k]], u, learner, spline_cache[[k]])
    step <- nus[[k]] * best$fitted
    eta[[k]] <<- eta[[k]] + step
    par[[k]] <<- family$invlink[[k]](eta[[k]])

    stored_step <- .scale_step(best, nus[[k]])
    selected[[k]] <<- c(selected[[k]], best$var)
    steps[[k]]    <<- c(steps[[k]], list(stored_step))
    roundv[[k]]   <<- c(roundv[[k]], round)

    if (track_val) {
      x_val    <- val_designs[[k]][, best$var]
      val_step <- if (best$type == "linear") {
        nus[[k]] * (best$intercept + best$slope * x_val)
      } else {
        B_val <- splines::bs(x_val, knots = best$knots, degree = best$degree,
                              Boundary.knots = best$boundary, intercept = TRUE)
        nus[[k]] * as.numeric(B_val %*% best$coefs)
      }
      val_eta[[k]] <<- val_eta[[k]] + val_step
      val_par[[k]] <<- family$invlink[[k]](val_eta[[k]])
    }

    log_submodel <<- c(log_submodel, k)
    log_variable <<- c(log_variable, best$var)
    log_round    <<- c(log_round, round)
    if (best$type == "linear") {
      log_coef      <<- c(log_coef, nus[[k]] * best$slope)
      log_intercept <<- c(log_intercept, nus[[k]] * best$intercept)
    } else {
      log_coef      <<- c(log_coef, NA_real_)
      log_intercept <<- c(log_intercept, NA_real_)
    }

    invisible(NULL)
  }

  if (method == "cyclic") {
    n_rounds <- max(unlist(mstops))
    risk     <- numeric(n_rounds)
    val_risk <- if (track_val) numeric(n_rounds) else NULL

    for (round in seq_len(n_rounds)) {
      for (k in P) {
        if (round <= mstops[[k]]) .apply_step(k, round)
      }

      risk[round]     <- family$risk(y, par)
      executed_rounds <- round

      if (track_val) {
        val_risk[round] <- family$risk(val_y, val_par)
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
  } else {
    n_rounds <- mstops[[P[1L]]]  # single shared budget across all parameters
    risk     <- numeric(n_rounds)
    val_risk <- if (track_val) numeric(n_rounds) else NULL

    for (round in seq_len(n_rounds)) {
      risk_current <- family$risk(y, par)

      candidates <- lapply(P, function(k) {
        u    <- family$ngradient[[k]](y, par)
        best <- .best_base_learner_general(designs[[k]], u, learner, spline_cache[[k]])
        eta_k_cand <- eta[[k]] + nus[[k]] * best$fitted
        par_cand <- par
        par_cand[[k]] <- family$invlink[[k]](eta_k_cand)
        list(best = best, eta_k_cand = eta_k_cand,
             risk_cand = family$risk(y, par_cand))
      })

      improvements <- risk_current - vapply(candidates, `[[`, numeric(1), "risk_cand")
      win_idx <- which.max(improvements)   # ties favor the earlier parameter, matching >=
      k_win   <- P[win_idx]
      chosen  <- candidates[[win_idx]]

      eta[[k_win]] <- chosen$eta_k_cand
      par[[k_win]] <- family$invlink[[k_win]](eta[[k_win]])

      stored_step <- .scale_step(chosen$best, nus[[k_win]])
      selected[[k_win]] <- c(selected[[k_win]], chosen$best$var)
      steps[[k_win]]    <- c(steps[[k_win]], list(stored_step))
      roundv[[k_win]]   <- c(roundv[[k_win]], round)

      if (track_val) {
        x_val <- val_designs[[k_win]][, chosen$best$var]
        val_step <- if (chosen$best$type == "linear") {
          nus[[k_win]] * (chosen$best$intercept + chosen$best$slope * x_val)
        } else {
          B_val <- splines::bs(x_val, knots = chosen$best$knots,
                                degree = chosen$best$degree,
                                Boundary.knots = chosen$best$boundary,
                                intercept = TRUE)
          nus[[k_win]] * as.numeric(B_val %*% chosen$best$coefs)
        }
        val_eta[[k_win]] <- val_eta[[k_win]] + val_step
        val_par[[k_win]] <- family$invlink[[k_win]](val_eta[[k_win]])
      }

      log_submodel <- c(log_submodel, k_win)
      log_variable <- c(log_variable, chosen$best$var)
      log_round    <- c(log_round, round)
      if (chosen$best$type == "linear") {
        log_coef      <- c(log_coef, nus[[k_win]] * chosen$best$slope)
        log_intercept <- c(log_intercept, nus[[k_win]] * chosen$best$intercept)
      } else {
        log_coef      <- c(log_coef, NA_real_)
        log_intercept <- c(log_intercept, NA_real_)
      }

      risk[round]     <- family$risk(y, par)
      executed_rounds <- round

      if (track_val) {
        val_risk[round] <- family$risk(val_y, val_par)
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
    round          = log_round,
    submodel       = log_submodel,
    variable       = log_variable,
    coef           = log_coef,
    intercept_step = log_intercept,
    stringsAsFactors = FALSE
  )

  # collapse steps[[k]] into flat coef/intercept_step vectors when every
  # step is linear; otherwise coef[[k]] keeps the full step-object list
  coef <- intercept_step <- stats::setNames(vector("list", length(P)), P)
  for (k in P) {
    steps_k <- steps[[k]]
    all_linear <- length(steps_k) == 0L ||
      all(vapply(steps_k, function(s) s$type == "linear", logical(1)))
    if (all_linear) {
      coef[[k]]           <- vapply(steps_k, function(s) s$slope, numeric(1))
      intercept_step[[k]] <- vapply(steps_k, function(s) s$intercept, numeric(1))
    } else {
      coef[[k]]           <- steps_k
      intercept_step[[k]] <- NA_real_
    }
  }

  list(
    eta = eta, par = par,
    selected = selected, coef = coef,
    intercept_step = intercept_step, round = roundv,
    n_rounds = n_rounds, step_log = step_log,
    risk = risk, risk0 = risk0,
    best_round = if (track_val) best_round else NULL,
    val_risk   = if (track_val) val_risk else NULL,
    val_risk0  = if (track_val) val_risk0 else NULL
  )
}
