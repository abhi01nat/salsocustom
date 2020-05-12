library(salsocustom)
load("Z:/Coins project/MCMC_Alexander/clustering_results.Rdata")
p <- diag(c(1, 1, 1))
salso(p, 3L, 0.5, 6L, 3L)