# Script for getting tuition growth rates (Real and Nominal)

library(DataCombine)
library(pastecs)

GrowthReal <- function(){
    df <- read.csv("lawData.csv")
    d <- df[c('year', 'Tuition', 'TuitionL')]
    d <- d[complete.cases(d),]
    d <- d[oder(d$year),]
    g <- d$Tuition / d$TuitionL
    tapply(g, d$year, mean)
}

GrowthNominal <- function(){
    df <- read.csv("lawData.csv")
    df <- slide(df, "Tuition_nominal", "year", "school",
                "Tuition_nominalL", -1)
    d <- df[c('year', 'Tuition_nominal', 'Tuition_nominalL')]
    d <- d[complete.cases(d),]
    d <- d[oder(d$year),]
    g <- d$Tuition_nominal / d$Tuition_nominalL
    tapply(g, d$year, mean)
}

gaggR <- GrowthReal()
gaggN <- GrowthNominal()


