library("texreg")
library("xtable")
library("reshape2")
library("ggplot2")
library("grid")

PATH <- file.path(dirname(dirname(getwd())), 'Results', 'FirstStage')


ModelList <- function(lhs) {
    # Model Specification
    list(
        paste(
            lhs,
            "~ OverallRank + Tuition + LSAT + LSDAS_GPA"
        ),
        paste(
            lhs,
            "~ OverallRank + Tuition + LSAT + LSDAS_GPA + treat"
        ),
        paste(
            lhs,
            "~ OverallRank + Tuition + LSAT + LSDAS_GPA + treat + year"
        ),
        paste(
            lhs,
            "~ OverallRank*Tuition*treat + LSAT + LSDAS_GPA + year"
        ),
        paste(
            lhs,
            "~ OverallRank + Tuition + LSAT*LSDAS_GPA*treat + year"
        )#,
        #paste(
        #    lhs,
        #    "~ OverallRank*Tuition*LSAT*LSDAS_GPA*treat + year"
        #)
    )
}


LogitEst <- function(data, lhs) {
    # Estimates logit model for given lhs
    models <- ModelList(lhs)
    nModels <- length(models)
    estimates <- list()
    for (i in 1:nModels) {
        estimates[[i]] <- lm(formula(models[[i]]), data)
    }

    caption <- paste("Logit:", lhs)
    label <- paste("tab:", lhs, sep="")
    path <- file.path(PATH, "Tables",
                      paste("logit", lhs, ".tex", sep=""))
    texreg(estimates,
           caption=caption, label=label, dcolumn=FALSE, booktabs=FALSE,
           use.packages=FALSE, scriptsize=TRUE, file=path)
}


Script <- function() {
    # Driver
    data <- read.csv(file.path(dirname(getwd()), "data",
                               "lsnLong.csv"))
    LogitEst(data, 'app')
    LogitEst(data[data$app == 1,], 'admit')
    LogitEst(data[data$admit == 1,], 'matric')

}

Script()
