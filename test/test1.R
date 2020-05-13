library(salsocustom)
#load("Z:/Coins project/MCMC_Alexander/clustering_results.Rdata")
N <- 500L
p <- diag(rep(1, N))
p[1, 2] <- 1
p[2, 1] <- 1
salso(p, N)
