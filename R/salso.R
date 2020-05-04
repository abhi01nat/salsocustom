#' Sequentially Allocated Latent Structure Optimisation
#'
#' Heuristic clustering to minimise the expected Binder loss function with respect to a given co-clustering probability matrix. 
#'
#' @param p co-clustering probability matrix, also called the expected adjacency matrix
#' @param maxClust maximum number of clusters
#' @param Const_Binder relative penalty in the Binder loss function for false-positives vs false-negatives
#' @param nPerm number of random permutations to start optimising from
#' @param nScans number of scans for each permutation
#' @return list containing a vector of cluster labels and the associated binder loss function
salso <- function(p, maxClust, Const_Binder = 0.5, nPerm = 10000L, nScans = 3L) {
    if (any(!is.numeric(p)) | any(!is.finite(p)) | nrow(p) != ncol(p) | any(p != t(p)) | any(p > 1) | any(p < 0) | sum(diag(p)) != nrow(p) ){
        stop("p must be a symmetric matrix with values between 0 and 1, and 1s on the diagonal.")
    }
    if (!is.numeric(maxClust) | !is.finite(maxClust) | !is.integer(maxClust) | maxClust <= 0) {
        stop("maxClust must be a positive integer.")
    }
    if (!is.numeric(Const_Binder) | !is.finite(Const_Binder) | Const_Binder < 0 | Const_Binder > 1) {
        stop("Const_Binder must be a number between 0 and 1.")
    }
    if (!is.numeric(nPerm) | !is.finite(nPerm) | !is.integer(nPerm) | nPerm <= 0) {
        stop("nPerm must be a positive integer.")
    }
    if (!is.numeric(nScans) | !is.finite(nScans) | !is.integer(nScans) | nScans <= 0) {
        stop("nScans must be a positive integer.")
    }
    return (.salso(p, maxClust, Const_Binder, nPerm, nScans))
}