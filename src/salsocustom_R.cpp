#include "salsocustom.h"

// [[Rcpp::export(name= ".salso")]]
Rcpp::List salso_Rcpp_unsafe (const Rcpp::NumericMatrix& p, int maxClust,  double Const_Binder = 0.5, int nPerm = 10000, int nScans = 3) {
    salso_result tmp_result = salso_cpp(Rcpp::as<arma::mat>(p), maxClust, Const_Binder, nPerm, nScans);
    return Rcpp::List::create(Rcpp::_["Labels"] = tmp_result.label + 1, // have our labels start from 1
                            Rcpp::_["BinderLoss"] = tmp_result.binder_loss);
}