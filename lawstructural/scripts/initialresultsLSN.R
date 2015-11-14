library("texreg")
library("xtable")
library("reshape2")
library("ggplot2")
library("grid")

PATH <- file.path(dirname(dirname(getwd())), 'Results', 'Initial')


AppDistGrapher <- function(data) {
    # Generate distributions of applicant credentials by year
    #

    pdf(file.path(PATH, 'Figures', 'app_dist.pdf'))
    par(mfrow=c(4, 3))
    for (year in unique(data$year)) {
        lsat_mean <- mean(data[data$year==year,]$LSAT, na.rm=TRUE)
        lsat_std <- sd(data[data$year==year,]$LSAT, na.rm=TRUE)
        xlab <- bquote(paste('LSAT: ', mu == .(lsat_mean), ', ',
                           sigma == .(lsat_std)))
        plot(density(data[data$year==year,]$LSAT), main=year, xlab=xlab,
             ylab='Density')
        abline(v=lsat_mean)
    }
    dev.off()
}


SummaryStats <- function(data, sum_var) {
    # Summary statistics for variable sum_var
    sum_var_out <- c('Applications', 'Admissions', 'Matriculations')
    names(sum_var_out) <- c('app', 'admit', 'matric')
    vars <- c(sum_var, 'LSAT', 'LSDAS_GPA', 'OverallRank', 'Tuition')
    varsOut <- c(sum_var_out[sum_var], 'LSAT', 'GPA', 'Rank', 'Tuition')
    operations <- c(mean, sd, min, median, max)
    opOut <- c("Mean", "SD", "Min", "Median", "Max", "n")
    align <- c("l", rep("r", length(opOut)))
    display <- c("s", rep("f", length(opOut) - 1), "d")
    dataVars <- data[,vars]
    out <- c()
    for (op in operations) {
        out <- cbind(out, apply(dataVars, 2, op, na.rm=TRUE))
    }
    nObs <- c()
    for (v in vars) {
        dataV <- dataVars[,v]
        nObs <- c(nObs, length(dataV[!is.na(dataV)]))
    }
    out <- cbind(out, nObs)

    colnames(out) <- opOut
    rownames(out) <- varsOut
    sumTable <- xtable(out,
        caption=paste("Summary Statistics:", sum_var_out[sum_var]),
        label=paste("tab:summary", sum_var, sep=""),
        align=align, display=display
    )
    path <- file.path(PATH, "Tables",
                      paste("summary", sum_var, ".tex", sep=""))
    print(sumTable, type="latex", file=path)

}


SummarySampleSubsets <- function(data_in) {
    # Summary statistics per sample subset determined by positive event.
    # For example, summary stats on all the students who have been admitted
    # to a school
    out <- c()
    op <- c(mean, sd)
    for (treat in c(0, 1)) {
        data <- data_in[data_in$treat == treat,]
        for (i in c(1, 2)) {
            # Apply mean/sd to each of the of the variables
            out_row <- apply(
                data[,c('LSAT', 'LSDAS_GPA', 'app_n', 'admit_binary')],
                2, op[[i]], na.rm=TRUE
            )
            out_row <- round(out_row, digits=2)
            # Probability of matriculation doesn't apply yet
            out_row <- c(out_row, NA)
            if (i == 1) {
                # Not exactly what I'm trying to get, I think
                out_row <- c(out_row, dim(data)[1])
                out_row = c("Applicants", out_row)
            } else {
                out_row <- c(out_row, NA)
                for (i in seq(length(out_row))) {
                    if (!is.na(out_row[i])) {
                        out_row[i] <- paste('(', toString(out_row[i]), ')',
                                            sep="")
                    }
                }
                out_row = c(NA, out_row)
            }
            out <- rbind(out, out_row)
        }
        for (i in c(1, 2)) {
            data_sub <- data[data$admit_binary == 1,]
            out_row <- apply(data_sub[c('LSAT', 'LSDAS_GPA', 'app_n')],
                             2, op[[i]], na.rm=TRUE)
            out_row <- round(out_row, digits=2)
            out_row <- c(out_row, NA)
            out_row <- c(out_row,
                         round(op[[i]](data_sub$matric_binary), digits=2))
            if (i == 1) {
                out_row <- c(out_row, dim(data_sub)[1])
                out_row = c("Admitted", out_row)
            } else {
                out_row <- c(out_row, NA)
                for (i in seq(length(out_row))) {
                    if (!is.na(out_row[i])) {
                        out_row[i] <- paste('(', toString(out_row[i]), ')',
                                            sep="")
                    }
                }
                out_row = c(NA, out_row)
            }
            out <- rbind(out, out_row)
        }
        for (i in c(1, 2)) {
            data_sub <- data[data$matric_binary == 1,]
            out_row <- apply(data_sub[c('LSAT', 'LSDAS_GPA', 'app_n')],
                             2, op[[i]], na.rm=TRUE)
            out_row <- round(out_row, digits=2)
            out_row <- c(out_row, c(NA, NA))
            if (i == 1) {
                out_row <- c(out_row, dim(data_sub)[1])
                out_row = c("Matriculants", out_row)
            } else {
                out_row <- c(out_row, NA)
                for (i in seq(length(out_row))) {
                    if (!is.na(out_row[i])) {
                        out_row[i] <- paste('(', toString(out_row[i]), ')',
                                            sep="")
                    }
                }
                out_row = c(NA, out_row)
            }
            out <- rbind(out, out_row)
        }
    }
    out <- cbind(c('Treat=0', rep(NA, 5), 'Treat=1', rep(NA, 5)), out)
    colnames(out) <- c('Info', 'Subset', 'LSAT', 'GPA', '$n_{Apps}$',
                       'Pr(Admit)', 'Pr(Matric)', '$n$')
    rownames(out) <- seq(dim(out)[1])
    out_table <- xtable(
        out, label='tab:summary_appadmit',
        caption="Stage Game Student Profile Summaries"
    )
    align(out_table) = "rcr|cccccc"
    print(out_table, include.rownames=FALSE, hline.after=c(-1, 0, 6),
          sanitize.text.function=function(x) {x},
          file=file.path(PATH, 'Tables', 'summary_appadmit.tex'))
}


QuantileSummary <- function(data_in) {
    # Summary statistics of applicant profile and application/admission
    # behavior based on profile quantile and treatment periods
    op <- c(mean, sd)
    out_vars <- c('LSAT', 'LSDAS_GPA', 'app_n', 'admit_binary',
                  'matric_binary')
    for (qvar in c('LSAT', 'LSDAS_GPA')) {
        if (qvar == 'LSDAS_GPA') {
            qvar_out <- 'GPA'
        } else {
            qvar_out <- 'LSAT'
        }
        out <- c()
        for (treat in c(0, 1)) {
            data <- data_in[data_in$treat == treat,]
            quantiles <- quantile(data[qvar], na.rm=TRUE)
            for (quant in seq(length(quantiles) - 1)) {
                data_sub <- data[(data[qvar] >= quantiles[quant]) &
                                 (data[qvar] <= quantiles[quant + 1]),]
                for (i in c(1, 2)) {
                    out_row <- apply(data_sub[out_vars],
                                     2, op[[i]], na.rm=TRUE)
                    out_row <- round(out_row, digits=2)
                    if (i == 1) {
                        pctl <- quant * 25
                        out_row <- c(paste(pctl - 25, "-", pctl, "th", sep=""),
                                     out_row)
                    } else {
                        for (i in seq(length(out_row))) {
                            out_row[i] <- paste('(', toString(out_row[i]),
                                                ')', sep="")
                        }
                        out_row <- c(NA, out_row)
                    }
                    if ((quant == 1) & (i == 1)) {
                        out_row <- c(paste("Treat=", treat, sep=""), out_row)
                    } else {
                        out_row <- c(NA, out_row)
                    }
                    out <- rbind(out, out_row)
                }
            }
        }
        colnames(out) <- c('Info', 'Pctl.', 'LSAT', 'GPA', '$n_{Apps}$',
                           'pr(Admit)', 'pr(Matric)')
        rownames(out) <- seq(dim(out)[1])
        out_table <- xtable(
            out, label=paste('tab:summary_quant_', qvar_out, sep=""),
            caption=paste("Summary by", qvar_out, "Quantile")
        )
        align(out_table) = "rcr|ccccc"
        print(
            out_table, include.rownames=FALSE, hline.after=c(-1, 0, 8),
            sanitize.text.function=function(x) {x},
            file=file.path(
                PATH, 'Tables',
                paste('summary_quant_', qvar_out, '.tex', sep="")
            )
        )
    }

}


Script <- function() {
    data <- read.csv(file.path(dirname(getwd()), "data",
                               "lawschoolnumbers.csv"))
    #AppDistGrapher(data)
    #SummarySampleSubsets(data)
    QuantileSummary(data)
    #data <- read.csv(file.path(dirname(getwd()), "data",
    #                           "lsnLong.csv"))
    #SummaryStats(data, 'app')
    #SummaryStats(data[data$app == 1,], 'admit')
    #SummaryStats(data[data$admit == 1,], 'matric')
}

Script()
