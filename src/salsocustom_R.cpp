#include "salsocustom.h"

// [[Rcpp::export(name= ".salso")]]
Rcpp::List salsoRcpp (const Rcpp::NumericMatrix& epam, int maxClusts,  double Const_Binder, int batchSize, int nScans, int maxThreads, int timeLimit) {
    salso_result_t tmpResult = salsoCpp(Rcpp::as<arma::mat>(epam), maxClusts, Const_Binder, batchSize, nScans, maxThreads, timeLimit);
    return Rcpp::List::create(Rcpp::_["Labels"] = tmpResult.labels, 
                            Rcpp::_["BinderLoss"] = tmpResult.binderLoss,
                            Rcpp::_["NumClusts"] = tmpResult.numClusts,
                            Rcpp::_["NumPermutations"] = tmpResult.nIters, 
                            Rcpp::_["WallClockTime"] = tmpResult.wallClockTime,
                            Rcpp::_["TimeLimitReached"] = tmpResult.timeLimitReached,
                            Rcpp::_["NumThreads"] = tmpResult.numThreads);
}