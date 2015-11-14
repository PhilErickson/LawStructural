## Utility functions for R (extraction methods for texreg) ##

extract.nls <- function(model, include.rsquared = FALSE,
                          include.adjrs = FALSE, include.nobs = FALSE,
                          include.iter = TRUE, ...) {
    s <- summary(model, ...)
    names <- rownames(s$coef)
    co <- s$coef[, 1]
    se <- s$coef[, 2]
    pval <- s$coef[, 4]

    gof <- numeric()
    gof.names <- character()
    gof.decimal <- logical()
    if (include.rsquared == TRUE) {
        rs <- s$r.squared
        gof <- c(gof, rs)
        gof.names <- c(gof.names, "R$^2$")
        gof.decimal <- c(gof.decimal, TRUE)
    }
    if (include.adjrs == TRUE) {
        adj <- s$adj.r.squared
        gof <- c(gof, adj)
        gof.names <- c(gof.names, "Adj.\\ R$^2$")
        gof.decimal <- c(gof.decimal, TRUE)
    }
    if (include.nobs == TRUE) {
        n <- nobs(model)
        gof <- c(gof, n)
        gof.names <- c(gof.names, "Num.\\ obs.")
        gof.decimal <- c(gof.decimal, FALSE)
    }
    if (include.iter == TRUE) {
        iter <- s$convInfo$finIter
        gof <- c(gof, iter)
        gof.names <- c(gof.names, "Iterations")
        gof.decimal <- c(gof.decimal, FALSE)
    }
    tr <- createTexreg(
        coef.names = names,
        coef = co,
        se = se,
        pvalues = pval,
        gof.names = gof.names,
        gof = gof,
        gof.decimal = gof.decimal
    )
    return(tr)
}


extract.arima <- function(model, include.nobs = TRUE,
                          include.aic = TRUE, include.ll = TRUE, ...) {
    names <- names(coef(model))
    co <- coef(model)
    se <- sqrt(diag(vcov(model)))
    n <- as.numeric(summary(model)['residuals', 'Length'])
    df <- n - as.numeric(summary(model)['coef', 'Length'])
    pval <- 2 * (1 - pt(co / se, df))

    gof <- numeric()
    gof.names <- character()
    gof.decimal <- logical()
    if (include.nobs == TRUE) {
        gof <- c(gof, n)
        gof.names <- c(gof.names, "Num.\\ obs.")
        gof.decimal <- c(gof.decimal, FALSE)
    }
    if (include.aic == TRUE) {
        aic <- model$aic
        gof <- c(gof, aic)
        gof.names <- c(gof.names, "AIC")
        gof.decimal <- c(gof.decimal, TRUE)
    }
    if (include.ll == TRUE) {
        loglik <- model$loglik
        gof <- c(gof, loglik)
        gof.names <- c(gof.names, "Log Likelihood")
        gof.decimal <- c(gof.decimal, TRUE)
    }
    tr <- createTexreg(
        coef.names = names,
        coef = co,
        se = se,
        pvalues = pval,
        gof.names = gof.names,
        gof = gof,
        gof.decimal = gof.decimal
    )
    return(tr)
}

