library("texreg")
library("xtable")
#library("pastecs")
#library("DataCombine")
library("reshape2")
library("ggplot2")
library("grid")

PATH <- file.path(dirname(dirname(getwd())), 'Results')

# --------------- #
#    UTILITIES    #
# --------------- #

## FROM http://stackoverflow.com/questions/7519790/assign-multiple-new-variables-in-a-single-line-in-r

# Generic form
'%=%' = function(l, r, ...) UseMethod('%=%')

# Binary Operator
'%=%.lbunch' = function(l, r, ...) {
  Envir = as.environment(-1)

  if (length(r) > length(l))
    warning("RHS has more args than LHS. Only first", length(l), "used.")

  if (length(l) > length(r))  {
    warning("LHS has more args than RHS. RHS will be repeated.")
    r <- extendToMatch(r, l)
  }

  for (II in 1:length(l)) {
    do.call('<-', list(l[[II]], r[[II]]), envir=Envir)
  }
}

# Used if LHS is larger than RHS
extendToMatch <- function(source, destin) {
  s <- length(source)
  d <- length(destin)

  # Assume that destin is a length when it is a single number and source is not
  if(d==1 && s>1 && !is.null(as.numeric(destin)))
    d <- destin

  dif <- d - s
  if (dif > 0) {
    source <- rep(source, ceiling(d/s))[1:d]
  }
  return (source)
}

# Grouping the left hand side
g = function(...) {
  List = as.list(substitute(list(...)))[-1L]
  class(List) = 'lbunch'
  return(List)
}

# Little utility to see if number is single digit or not
digit_check <- function(vec) all(vec >= 0 & vec <= 9 & vec%%1==0)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
    # Multiple plot function
    #
    # ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
    # - cols:   Number of columns in layout
    # - layout: A matrix specifying the layout. If present, 'cols' is ignored.
    #
    # If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
    # then plot 1 will go in the upper left, 2 will go in the upper right, and
    # 3 will go all the way across the bottom.
    #
    # http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

g_legend <- function(a.gplot){
    # Extract legend from a ggplot object
    # Source:
    #   http://stackoverflow.com/questions/11883844/
    #   inserting-a-table-under-the-legend-in-a-ggplot2-histogram
    tmp <- ggplot_gtable(ggplot_build(a.gplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)}

DataClean <- function(dataIn, varIn) {
    # Remove obs with NA and Inf in important variable from dataframe
    data <- dataIn[!is.na(dataIn[varIn]),]
    data <- data[data[varIn] != Inf,]
    data
}

# ----------------------------------------------- #
#    CHECK PERSISTENCE OF FIRM PLACEMENT TYPES    #
# ----------------------------------------------- #
IntersectSeveral <- function(...) {
    # Get common elements of list of vectors
    Reduce(intersect, list(...))
}


PersistenceCheck <- function(){
    # Makes tables for AR(1) models for placement and various sizes of firms
    #
    # Args:
    #
    # Returns:
    #   null; generates new table
    data <- PersistenceDataGen()
    testVars <- c("Solo", "X2.10", "X11.25", "X26.50", "X51.100", "X101.250",
                  "X251.500", "X501.PLUS", "Unknown", "ratio")
    testOut <- c("Solo", "2-10", "11-25", "26-50", "51-100", "101-250",
                 "251-500", "500+", "Unknown", "Ratio")
    testCoefs <- c()
    testSE <- c()
    for (tvar in testVars) {
        data <- slide(data, Var=tvar, GroupVar="SchoolName",
                      NewVar=paste(tvar, ".L1", sep=""), slideBy=-1,
                      reminder=FALSE)
        form <- formula(paste(tvar, "~", tvar, ".L1", sep=""))
        testModel <- summary(lm(form, data))
        testCoefs <- c(testCoefs, coef(testModel)[2, 1])
        testSE <- c(testSE, coef(testModel)[2, 2])
    }

    out <- rbind(testCoefs, testSE)
    colnames(out) <- testOut
    rownames(out) <- c("Coefficient", "SE")
    out <- xtable(out,
                  caption="Students placed in different firm sizes: AR(1)",
                  label="tab:placement")
    print(out, type="latex",
          file=file.path(PATH, "FirstStage", "Tables", "placement.tex"))
}



# ---------------------------------------- #
#    DIFFERENCE-IN-DIFFERENCE FUNCTIONS    #
# ---------------------------------------- #
LhsNames <- function(lhsIn) {
    # Returns variable names, labels, and corresponding control variables
    # for input specification
    #
    # Args:
    #   lhsIn: left-hand-side variable for DiffInDiff()
    #
    # Returns:
    #   list(variable name, label, all other possible lhs)
    if (lhsIn == "tuition") {
        lhs <- "Tuition"
        lhsOut <- "Tuition"
    } else if (lhsIn == "class") {
        lhs <- "FreshmanEnrolled"
        lhsOut <- "Students"
    } else if (lhsIn == "gpa") {
        lhs <- "UndergraduatemedianGPA"
        lhsOut <- "Undergrad GPA"
    } else if (lhsIn == "lsat") {
        lhs <- "MedianLSAT"
        lhsOut <- "LSAT"
    } else if (lhsIn == "sal25") {
        lhs <- "p25privatesectorstartingsalary"
        lhsOut <- "Salary (25th pct.)"
    } else if (lhsIn == "sal75") {
        lhs <- "p75privatesectorstartingsalary"
        lhsOut <- "Salary (75th pct.)"
    } else if (lhsIn == "report") {
        lhs <- "Percentofgraduatesinprivatesectorreportingsalaryinformation"
        lhsOut <- "PercentReportingSalary"
    } else if (lhsIn == "apps") {
        lhs <- "Numberofapplicantsfulltime"
        lhsOut <- "Applicants"
    } else if (lhsIn == "rank") {
        lhs <- "OverallRank"
        lhsOut <- "Rank"
    } else if (lhsIn == "ratio") {
        lhs <- "ratio"
        lhsOut <- "Ratio"
    } else if (lhsIn == "grants") {
        lhs <- "Percentofstudentsreceivinggrantsfulltime"
        lhsOut <- "Number of Grants"
    } else if (lhsIn == "medgrants") {
        lhs <- "Mediangrantfulltime"
        lhsOut <- "Grants (50th pct.)"
    } else {
        stop("Invalid lhs variable.")
    }

    reacVars <- c("Tuition", "FreshmanEnrolled",
                   "UndergraduatemedianGPA", "MedianLSAT")
    reacControls <- reacVars[!reacVars == lhs]
    reacOut <- c("Tuition", "Students", "Undergrad GPA", "LSAT")
    reacOut <- reacOut[!reacOut == lhsOut]
    list(lhs, lhsOut, reacControls, reacOut)
}


RhsNames <- function(rhsIn) {
    # Returns variable names and labels for rhs specification
    #
    # Args:
    #   rhsIn: right-hand-side variable for DiffInDiff()
    #
    # Returns:
    #   list(variable name, label)
    if (rhsIn == "rank") {
        rhs <- "OverallRank"
        rhsOut <- "Rank"
    } else if (rhsIn == "ratio") {
        rhs <- "ratio"
        rhsOut <- "Ratio"
    } else {
        stop("Invalid rhs variable.")
    }
    list(rhs, rhsOut)
}


NonReacControls <- function() {
    # Return control variables and labels outside of reaction variables
    nonReacControls <- c("Mediangrantfulltime", "Percentofstudentsreceivinggrantsfulltime",
                         "Roomboardbooksmiscellaneousexpenses", "Estimatedcostofbooks",
                         "Studentfacultyratio", "Numberofapplicantsacceptedfulltime",
                         "Acceptanceratefulltime",
                         "Barpassagerateinjurisdictionfirsttimetesttakers")
    nonReacOut <- c("Median Grant", "Percent Grants", "Room/Board Expenses",
                    "Cost of Books", "Student/Faculty Ratio", "Accepted",
                    "Acceptance Rate", "Bar Passage Rate")
    list(nonReacControls, nonReacOut)
}


ModelList <- function(lhs, rhs, reacControls, nonReacControls) {
    # Returns list of model speficiations, determines TableNames()
    #
    # Args:
    #   lhs: Left-hand-side variable
    #   reacControls: reaction control variables (other possible lhs)
    #
    # Returns:
    #   List of models specifications for DiffInDiff()
    if (rhs == "ratio") {
        f0 <- paste(lhs, "~ ratio * treat")
        f1 <- paste(f0, "+", "OverallRank")
    } else {
        f1 <- paste(lhs, "~ OverallRank * treat")
    }
    f2 <- paste(f1, "+", paste(reacControls, collapse=" + "))
    f3 <- paste(f1, "+", paste(nonReacControls[1:4], collapse=" + "))
    f4 <- paste(f1, "+", paste(nonReacControls[5:length(nonReacControls)],
                               collapse=" + "))
    f5 <- paste(f1, "+", paste(reacControls, collapse=" + "),
                "+", paste(nonReacControls, collapse=" + "))

    if (rhs == "ratio") {
        list(f0, f1, f2, f3, f4, f5)
    } else {
        list(f1, f2, f3, f4, f5)
    }
}


TableNames <- function(rhsOut, reacOut, nonReacOut) {
    # Aggregates variable labels for use in texreg for DiffInDiff(), order
    # determined by ModelList()
    #
    # Args:
    #   rhsOut: rhs label, returned from RhsNames()
    #   reacOut: reaction control variable labels, returned from LhsNames()
    #   nonReacOut: non-reaction control variable labels, returned from
    #               RhsNames()
    #
    # Returns:
    #   Vector of custom names
    names <- c("(Intercept)", rhsOut, "Post-2010",
               paste("Post-2010 *", rhsOut))
    if (rhsOut == "Ratio") {
        names <- c(names, "Rank")
    }
    c(names, reacOut, nonReacOut)
}


DiffInDiffDriver <- function(data, compact=FALSE) {
    # Driver for running DiffInDiff specifications and compiling graphs of
    # results
    #
    # Args:
    #   data: Output from function DataFormat()
    #
    ## Use this spec for only two reaction variables
    #lhsVector <- c("tuition", "class")
    ## Uncomment for all reaction variables
    lhsVector <- c("tuition", "class", "gpa", "lsat", "sal25", "sal75",
                   "report", "apps")
    compactPath <- ""
    if (compact) {
        lhsVector <- c("tuition", "class", "gpa", "lsat")
        compactPath <- "compact"
    }
    rhsVector <- c("ratio", "rank")
    for (rhs in rhsVector) {
        path <- file.path(PATH, "FirstStage", "Figures",
                          paste(rhs, "DinD", compactPath, ".pdf", sep=""))
        plots <- list()
        for (lhs in lhsVector) {
            g(plots[[lhs]]) %=% DiffInDiff(data, lhs, rhs)
        }
        pdf(path)
        multiplot(plotlist=plots, cols=2)
        dev.off()
    }
}


DiffInDiff <- function(dataIn, lhsIn, rhsIn) {
    # Estimates basic diff-in-diff models for variable "lhs" with treatment
    # at year 2010. Generates tables accordingly.
    #
    # Args:
    #   data: Output from function DataFormat()
    #   lhsIn: variable to estimate model for. Has options:
    #       - "tuition" (default): Tuition
    #           Note: This is combined out-of-state tuition and regular
    #                 tuition for schools who do not offer
    #                 an out-of-state/in-state menu
    #       - "class": FreshmanEnrolled
    #       - "gpa": UndergraduatemedianGPA
    #       - "lsat": MedianLSAT
    #   rhsIn: Diff-in-diff variable to be interacted with treatment. Options:
    #       - "rank": OverallRank
    #       - "ratio": variable generated by RatioGen()
    #
    # Returns:
    #   null; generates tables with various control specs
    g(lhs, lhsOut, reacControls, reacOut) %=% LhsNames(lhsIn)
    g(rhs, rhsOut) %=% RhsNames(rhsIn)
    g(nonReacControls, nonReacOut) %=% NonReacControls()

    data <- DataClean(dataIn, rhs)
    models <- ModelList(lhs, rhs, reacControls, nonReacControls)
    nModels <- length(models)
    estimates <- list()
    for (i in 1:nModels) {
        estimates[[i]] <- lm(formula(models[[i]]), data)
    }

    g(p) %=% CoefGrapher(estimates=estimates, lhsIn=lhsIn, rhsIn=rhsIn)
    namesOut <- TableNames(rhsOut, reacOut, nonReacOut)
    modelsOut <- paste(lhsOut, " (", seq(1:nModels), ")", sep="")
    caption <- paste("Difference-in-differences:", lhsOut, "vs", rhsOut)
    label <- paste("tab:", lhsIn, "-", rhsIn, sep="")
    path <- file.path(PATH, "FirstStage", "Tables",
                      paste(lhsIn, "-", rhsIn, ".tex", sep=""))
    texreg(estimates, custom.coef.names=namesOut, model.names=modelsOut,
           caption=caption, label=label, dcolumn=FALSE, booktabs=FALSE,
           use.packages=FALSE, scriptsize=TRUE, file=path)
    list(p)
}


DiffInDiffInDiff <- function(data) {
    # Estimate diff-in-diff-in-diff with Rank, Ratio, and Treatment
    #
    # Args:
    #   data: Output from function DataFormat()
    #
    # Returns:
    #   null; generates table
    data <- DataClean(data, 'ratio')
    data <- DataClean(data, 'OverallRank')
    m3d <- lm(Tuition ~ ratio*OverallRank*treat, data)
    names <- c("(Intercept)", "Ratio", "Rank", "Post-2010", "Ratio * Rank",
               "Ratio * Post-2010", "Rank * Post-2010",
               "Ratio * Rank * Post-2010")
    model <- "Tuition"
    caption <- "Diff-in-Diff-in-Diff: Ratio, Rank, Treatment"
    label <- "tab:3d"
    path <- file.path(PATH, "FirstStage", "Tables", "ddd.tex")
    texreg(m3d, custom.coef.names=names, model.names=model, caption=caption,
           label=label, dcolumn=FALSE, booktabs=FALSE, use.packages=FALSE,
           file=path)
}

CoefGrapher <- function(estimates, lhsIn, rhsIn="ratio") {
    # Graph estimated diff-in-diff coefficients for list of estimates
    #
    # Args:
    #   estimates: List of estimated linear models
    #   lhsIn: left-hand-side variable
    #   rhsIn: interaction variable with treatment (ratio or rank)
    #
    # Returns:
    #   null; generates table
    g(lhs, lhsOut, reacControls, reacOut) %=% LhsNames(lhsIn)
    g(rhs, rhsOut) %=% RhsNames(rhsIn)
    diffVar <- paste(rhs, ":treat", sep="")
    diffEstimates <- c()
    diffSE <- c()
    for (est in estimates) {
        diffEstimates <- c(diffEstimates, summary(est)$coef[,1][diffVar])
        diffSE <- c(diffSE, summary(est)$coef[,2][diffVar])
    }
    ci95p <- diffEstimates + 1.96*diffSE
    ci95m <- diffEstimates - 1.96*diffSE
    path <- file.path(PATH, "FirstStage", "Figures",
                      paste(lhsIn, "-", rhsIn, ".pdf", sep=""))
    modelsOut <- paste(lhsOut, " (", seq(1:length(estimates)), ")", sep="")
    caption <- paste(lhsOut, "vs", rhsOut)
    label <- paste("fig:", lhsIn, "-", rhsIn, sep="")
    ylab <- paste(rhsOut, "* Treat")
    modelsAxis <- seq(1:length(estimates))
    names(modelsAxis) <- modelsOut

    #pdf(path)
    #plot(diffEstimates, ylim=c(min(ci95m), max(ci95p)), main=caption,
    #     xlab="Models", ylab=ylab)
    #lines(diffEstimates)
    #lines(ci95p, lty=2)
    #lines(ci95m, lty=2)
    #legend((length(estimates)-1.5), max(ci95p), c("Coef", "95% C.I."),
    #       lty=c(1, 2))
    #abline(h=0, lty=3)
    #dev.off()
    if (rhs == 'ratio') {
        modelNames <- c("1: Base", "2: Rank", "3: Choice", "4: Cost",
                        "5: Selectivity", "6: All")
        modelNames <- seq(6)
    } else {
        modelNames <- c("1: Base", "2: Choice", "3: Cost",
                        "4: Selectivity", "5: All")
        modelNames <- seq(5)
    }
    df <- data.frame(
        models=factor(modelNames),
        diffEstimates=diffEstimates,
        diffSE=diffSE
    )
    limits <- aes(ymax=diffEstimates + 1.96*diffSE,
                  ymin=diffEstimates - 1.96*diffSE)
    dodge <- position_dodge(width=0.9)
    p <- ggplot(df, aes(y=diffEstimates, x=models))
    p <- p + geom_point()
    p <- p + geom_errorbar(limits, width=0.2)
    p <- p + labs(x="Models", y=expression(beta[3]), title=caption)
    #ggsave(p, file=path)
    list(p)
}


RatioGrapher <- function(dataIn, compact=FALSE) {
    # Graph TS of averages in each Ratio/Rank quantile
    data <- dataIn[dataIn$OverallRank < 195, ]
    basevars <- c("ratio", "OverallRank", "p25privatesectorstartingsalary",
                  "p75privatesectorstartingsalary")
    compactPath <- ""
    cols <- 3
    if (compact) {
        compactPath <- "compact"
        cols <- 2
    }
    for (base in basevars) {
        data <- DataClean(data, base)
        data$quant <- cut(
            unname(unlist(data[base])),
            breaks=quantile(data[base], probs=seq(0, 1, by=0.25), na.rm=TRUE),
            include.lowest=TRUE)
        plots <- list()
        #names = c("tuition", "class", "gpa", "lsat", "sal25", "sal75",
        #          "report", "apps")
        names <- c("tuition", "class", "gpa", "lsat", "sal25", "sal75",
                  "apps", "grants", "medgrants")
        if (compact) {
            names <- c("tuition", "class", "gpa", "lsat")
        }
        #if (base == "ratio") {
        #    names = c(names, "rank")
        #} else {
        #    names = c(names, "ratio")
        #}
        #if (base == ""p25privatesectorstartingsalary"") {
        #
        #} else if (base == "p75privatesectorstartingsalary") {
        #
        #}
        for (name in names) {
            g(lhs, lhsOut, reacControls, reacOut) %=% LhsNames(name)
            dataQ <- with(data, tapply(
                get(lhs),
                list(year, quant), mean,
                na.rm=TRUE
            ))
            dataQ <- melt(dataQ)
            if (base == "ratio") {
                qname <- "RatioQuantile"
            } else {
                qname <- "RankQuantile"
            }
            colnames(dataQ) <- c("year", qname, lhs)
            dataQ <- dataQ[dataQ$year >= 2001, ]
            #dataQ$year <- as.numeric(substring(as.character(dataQ$year), 3))
            #dataQ[lhsOut] <- (dataQ[lhsOut] -
            #                  mean(unname(unlist(dataQ[lhsOut]))))
            means <- tapply(unname(unlist(dataQ[lhs])),
                            unname(unlist(dataQ[qname])),
                            mean)
            means <- rep(means, each=length(unique(dataQ$year)))
            #dataQ[lhsOut] <- dataQ[lhsOut] - means
            plots[[name]] <- ggplot(dataQ, aes_string("year", lhs,
                        colour=qname))
            plots[[name]] <- plots[[name]] + geom_line() + labs(y=lhsOut)
            plots[[name]] <- plots[[name]] + geom_vline(xintercept=2010)
            legend <- g_legend(plots[[name]])
            plots[[name]] <- plots[[name]] + theme(legend.position="none")

        }
        path <- file.path(PATH, "Initial", "Figures",
                          paste(sub('\\.', '', base),
                                "Ave", compactPath, ".pdf", sep=""))
        pathLegend <- file.path(PATH, "Initial", "Figures",
                                paste(sub('\\.', '', base),
                                      "AveLegend", compactPath, ".pdf",
                                      sep=""))
        pdf(path)
        multiplot(plotlist=plots, cols=cols)
        dev.off()
        pdf(pathLegend)
        grid.draw(legend)
        dev.off()
        #ggsave(filename=path, plot=p)
    }
}

ScatterGrapher <- function(dataIn) {
    # Draw scatter plots for rank/ratio conditional reactions grouped by
    # pre/post treatment
    dataIn$treat <- as.factor(dataIn$treat)
    for (rhsIn in c("ratio", "rank")) {
        g(rhs, rhsOut) %=% RhsNames(rhsIn)
        data <- DataClean(dataIn, rhs)
        plots = list()
        #names = c("tuition", "class", "gpa", "lsat", "sal25", "sal75",
        #          "report", "apps")
        #if (rhsIn == "ratio") {
        #    names = c(names, "rank")
        #} else {
        #    names = c(names, "ratio")
        #}
        names = c("tuition", "class", "gpa", "lsat", "sal25", "sal75",
                  "apps", "grants", "medgrants")
        for (name in names) {
            g(lhs, lhsOut, reacControls, reacOut) %=% LhsNames(name)
            plots[[name]] <- ggplot(data, aes_string(x=rhs, y=lhs,
                        colour="treat"))
            plots[[name]] <- plots[[name]] + geom_point()
            plots[[name]] <- plots[[name]] + scale_colour_discrete(
                name="Post-2010")
            plots[[name]] <- plots[[name]] + labs(x=rhsOut, y=lhsOut)
            legend <- g_legend(plots[[name]])
            plots[[name]] <- plots[[name]] + theme(legend.position="none")

        }
        path <- file.path(PATH, "Initial", "Figures",
                          paste(rhs, "Scatter.pdf", sep=""))
        pathLegend <- file.path(PATH, "Initial", "Figures",
                                paste(rhs,"ScatterLegend.pdf", sep=""))
        pdf(path)
        multiplot(plotlist=plots, cols=3)
        dev.off()
        pdf(pathLegend)
        grid.draw(legend)
        dev.off()
    }
}

# ------------------------ #
#    SUMMARY STATISTICS    #
# ------------------------ #
SumStats <- function(data) {
    # Generate tables of summary statistics for relevant variables
    #
    # Args:
    #   data: Output from function DataFormat()
    #
    # Returns:
    #   null; generates table
    g(lhs, lhsOut, reacControls, reacOut) %=% LhsNames("tuition")
    g(rankVar, rankOut) %=% RhsNames("rank")
    g(ratioVar, ratioOut) %=% RhsNames("ratio")
    g(nonReacControls, nonReacOut) %=% NonReacControls()

    vars <- c("year", rankVar, ratioVar, lhs, reacControls, nonReacControls)
    varsOut <- c("year", rankOut, ratioOut, lhsOut, reacOut, nonReacOut)
    # Newly added
    vars <- c(vars, "p25privatesectorstartingsalary",
              "p75privatesectorstartingsalary",
              "Percentofgraduatesinprivatesectorreportingsalaryinformation")
    varsOut <- c(varsOut, "Private Sector 25% Salary",
                 "Private Sector 75% Salary",
                 "% Private Sector Reporting")
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
    sumTable <- xtable(out, caption="Summary Statistics", label="tab:summary",
                       align=align, display=display)
    path <- file.path(PATH, "Initial", "Tables", "summary.tex")
    print(sumTable, type="latex", file=path)
}


# --------------- #
#    SCRIPTING    #
# --------------- #
RfScript <- function() {
    # Script function for the analysis
    data <- read.csv(file.path(dirname(getwd()), "data", "lawData.csv"))
    #data <- read.csv('../data/lawData.csv')
    SumStats(data)
    RatioGrapher(data)
    RatioGrapher(data, compact=TRUE)
    ScatterGrapher(data)
    #PersistenceCheck()
    DiffInDiffDriver(data)
    DiffInDiffDriver(data, compact=TRUE)
    DiffInDiffInDiff(data)
}

RfScript()
warnings()
