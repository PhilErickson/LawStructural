library(MASS)
library(VGAM)

dfull <- read.csv("lawData.csv")
d <- d[c("OverallRank", "OverallRankL", "MedianLSATL",
         "UndergraduatemedianGPAL")]
d <- d[complete.cases(d),]
d <- d[d$OverallRank < 195,]
m <- polr(as.factor(OverallRank) ~ OverallRankL + MedianLSATL +
          UndergraduatemedianGPAL, data=d, Hess=TRUE)
# mfull picks up too much noise
mfull <- polr(as.factor(OverallRank) ~ as.factor(OverallRankL) + MedianLSATL +
              UndergraduatemedianGPAL, data=d, Hess=TRUE)

mt <- vglm(OverallRank ~ OverallRankL + MedianLSATL + UndergraduatemedianGPAL,
           tobit(Upper=194, Lower=1), data=d)

mlm <- lm(OverallRank ~ OverallRankL + MedianLSATL + UndergraduatemedianGPAL,
          data=d)

pm <- as.numeric(predict(m))
pmfull <- as.numeric(predict(mfull))
pt <- predict(mt)[,1]
pmlm <- predict(mlm)
ptr <- round(pt)

msem <- sum((d$OverallRank - pm)^2) / length(pm)
msemfull <- sum((d$OverallRank - pmfull)^2)
msemfull <- msemfull / length(pmfull)
mset <- sum((d$OverallRank - pt)^2) / length(pt)
msemlm <- sum((d$OverallRank - pmlm)^2) / length(pmlm)
msetr <- sum((d$OverallRank - ptr)^2) / length(ptr)


plot(d$OverallRankL,d$OverallRank)
points(d$OverallRankL, pm, col="red")
points(d$OverallRankL, pmfull, col="blue")
points(d$OverallRankL, pt, col="green")
points(d$OverallRankL, pmlm, col="purple")
points(d$OverallRankL, ptr, col="orange")

plot(density(d$OverallRank))
lines(density(pm), col="red")
lines(density(pmfull), col="blue")
lines(density(pt), col="green")
lines(density(pmlm), col="purple")
lines(density(ptr), col="orange")
