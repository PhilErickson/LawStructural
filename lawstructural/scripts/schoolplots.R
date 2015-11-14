
PATH <- file.path(dirname(dirname(getwd())), 'Results')
SCHOOL <- "University of Iowa"

PlotVars <- function(lhsIn) {
    if (lhsIn == "tuition") {
        lhs <- "Tuition"
        lhsOut <- "Tuition"
    } else if (lhsIn == "class") {
        lhs <- "FreshmanEnrolled"
        lhsOut <- "Freshmen"
    } else if (lhsIn == "gpa") {
        lhs <- "UndergraduatemedianGPA"
        lhsOut <- "UndergradGPA"
    } else if (lhsIn == "lsat") {
        lhs <- "MedianLSAT"
        lhsOut <- "LSAT"
    } else if (lhsIn == "sal25") {
        lhs <- "p25privatesectorstartingsalary"
        lhsOut <- "Salary25"
    } else if (lhsIn == "sal75") {
        lhs <- "p75privatesectorstartingsalary"
        lhsOut <- "Salary75"
    } else if (lhsIn == "report") {
        lhs <- "Percentofgraduatesinprivatesectorreportingsalaryinformation"
        lhsOut <- "PercentReportingSalary"
    } else if (lhsIn == "apps") {
        lhs <- "Numberofapplicantsfulltime"
        lhsOut <- "Applicants"
    } else if (lhsIn == "ratio") {
        lhs <- "ratio"
        lhsOut <- "Ratio"
    } else if (lhsIn == "grants") {
        lhs <- "Percentofstudentsreceivinggrantsfulltime"
        lhsOut <- "Grants"
    } else if (lhsIn == "medgrants") {
        lhs <- "Mediangrantfulltime"
        lhsOut <- "Grants50"
    } else {
        stop("Invalid lhs variable.")
    }
    c(lhs, lhsOut)
}

Plotter <- function(data, plotvar, school) {
    pvar <- PlotVars(plotvar)
    path <- file.path(PATH, "Initial", "Figures", "School",
                      paste(pvar[2], ".pdf", sep=""))
    pdf(path)
    scatter.smooth(data$OverallRank, data[[pvar[1]]],
                   lpars=list(col="red", lwd=3, lty=3),
                   xlab="Rank", ylab=pvar[2])
    points(data[school,]$OverallRank, data[school, pvar[1]],
           pch="+", col="blue")
    dev.off()
}

SchoolPlotsScript <- function() {
    plotVars <- c("tuition", "class", "gpa", "lsat", "sal25", "sal75",
                  "report", "apps", "ratio", "grants", "medgrants")
    data <- read.csv(file.path(dirname(getwd()), "data", "lawData.csv"))
    data <- data[data$OverallRank < 195,]
    school <- data$school == SCHOOL
    tryCatch(dir.create(file.path(PATH, "Initial", "Figures", "School")),
                        warning=function(w){""})
    for (pvar in plotVars) {
        Plotter(data, pvar, school)
    }
}

SchoolPlotsScript()
