#' Sequentially Allocated Latent Structure Optimisation
#'
#' Heuristic clustering to minimise the expected Binder loss function with respect to a given co-clustering probability matrix. 
#'
#' @param epam Expected Pairwise Allocation Matrix, i.e., the matrix whose entries \eqn{E_{ij}} is the posterior probability that items \eqn{i} and \eqn{j} are together. 
#' @param maxClusts Maximum number of clusters. The actual number of clusters searched may be lower. If set to 0L, the maximum is automatically limited by the number of items. 
#' @param Const_Binder Relative penalty in the Binder loss function for false-positives vis-a-vis false-negatives. Must be a real number in the interval [0, 1]. 
#' @param batchSize Number of permutations scanned per thread. If set to 0L, the thread will continue to scan permutations until it times out (in which case \code{timeLimit} cannot be 0L).
#' @param nScans Number of scans for each permutation. 
#' @param maxThreads Maximum number of threads to use. If set to 0L (default), the maximum number of threads will be determined by the runtime. Set to 1L for no parallelisation. The actual number of threads used may be lower than \code{maxThreads}.
#' @param timeLimit Maximum computation time for each thread in milliseconds. The actual computational time may be higher, since the time limit is only checked at the end of each iteration. If set to 0L, the thread will never time out (in which case \code{batchSize} cannot be 0L).
#' @return A list containing the following items:
#' \enumerate{
#' \item \code{Labels} - the vector of cluster labels
#' \item \code{BinderLoss} - the associated binder loss function
#' \item \code{NumClusts} - the number of clusters found
#' \item \code{NumPermutations} - the number of permutations actually scanned
#' \item \code{WallClockTime} - cumulative wall-clock time used by all threads in milliseconds
#' \item \code{TimeLimitReached} - whether the computation time limit was reached in any of the threads
#' \item \code{NumThreads} - actual number of threads used.
#' }
salso <- function(epam, maxClusts=0L, Const_Binder = 0.5, batchSize = 1000L, nScans = 10L, maxThreads = 0L, timeLimit = 300000L) {
    if (any(!is.numeric(epam)) | any(!is.finite(epam)) | nrow(epam) != ncol(epam) | any(epam != t(epam)) | any(epam > 1) | any(epam < 0) | sum(diag(epam)) != nrow(epam) ){
        stop("p must be a symmetric matrix with values between 0 and 1, and 1s on the diagonal.")
    }
    if (!is.numeric(maxClusts) | !is.finite(maxClusts) | !is.integer(maxClusts) | maxClusts < 0) {
        stop("maxClusts must be a nonnegative integer. ")
    }
    if (!is.numeric(Const_Binder) | !is.finite(Const_Binder) | Const_Binder < 0 | Const_Binder > 1) {
        stop("Const_Binder must be a number between 0 and 1.")
    }
    if (!is.numeric(batchSize) | !is.finite(batchSize) | !is.integer(batchSize) | batchSize < 0) {
        stop("batchSize must be a nonnegative integer.")
    }
    if (!is.numeric(nScans) | !is.finite(nScans) | !is.integer(nScans) | nScans <= 0) {
        stop("nScans must be a positive integer.")
    }
    if (!is.numeric(maxThreads) | !is.finite(maxThreads) | !is.integer(maxThreads) | maxThreads < 0) {
        stop("maxThreads must be a nonnegative integer.")
    }
    if (!is.numeric(timeLimit) | !is.finite(timeLimit) | !is.integer(timeLimit) | timeLimit < 0) {
        stop("timeLimit must be a nonnegative integer.")
    }
    if (timeLimit == 0L & batchSize == 0L) {
        stop("batchSize and timeLimit cannot both be 0.")
    }
    return (.salso(epam, maxClusts, Const_Binder, batchSize, nScans, maxThreads, timeLimit))
}