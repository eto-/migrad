closure <- function(f, n, s, t, ...) { 
  i <- 0; 
  e <- environment()
  function(p) { 
    e$i <- e$i + 1
    names(p) <- n
    v <- s * f(p, ...)
    if (t > 0 && (e$i %% t) == 0) cat("migrad step", e$i, " f(", paste(p, collapse=","), ") =", v, "\n")
    v
  }
}

migrad <- function (par, fn, gr = NULL, ..., method = NULL, lower = -Inf, upper = Inf, control = list(), hessian = FALSE) {
  if (!length(par)) return(fn(par, ...))

  control.default <- list(trace=0, fnscale = 1, maxit = 0, check.gradient = F, strategy = 1)
  if (length(un <- setdiff(names(control), names(control.default)))) warning("unknown names in control: ", paste(un, collapse = ", "))
  control.default[names(control)] <- control
  control <- control.default

  f <- closure(fn, names (par), control$fnscale, control$trace, ...)
  g <- closure(gr, names (par), control$fnscale, 0, ...)

  lower <- as.numeric(rep_len(lower, length(par)))
  upper <- as.numeric(rep_len(upper, length(par)))

  r <- list()
  if (is.null(gr)) {
    r <- migrad_cpp(par, f, lower, upper, control)
    r$counts <- c("function"=r$counts, gradient=NA)
  } else {
    r <- migrad_grad_cpp(par, f, g, lower, upper, control)
    r$counts <- c("function"=r$counts, gradient=get("i", environment(g)))
  }

  r <- c(r, list(message=NULL))
  r$value <- r$value / control$fnscale
  if (hessian) r$hessian <- solve(r$covariace) / control$fnscale
  r$covariace <- NULL

  r
}
