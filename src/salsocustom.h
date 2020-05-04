#pragma once

#include <algorithm>
#include <random>
#include <limits>

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
//#define ARMA_NO_DEBUG

typedef arma::uword ind_t; // type for all counters, indices, and labels

struct salso_result {
	arma::urowvec label;
	double binder_loss;
	salso_result (arma::uword N, double L) : label (N), binder_loss (L) {}
}; 
salso_result salso_cpp (const arma::mat& p, ind_t maxClust, double Const_Binder = 0.5, ind_t nPerm = 10000, ind_t nScans = 3);

struct best_clustering_t {
	arma::uword index;
	double binder_loss;
}; // type that stores the output of minimise_binder

#define BINDERS_TILE_SIZE 64
#define ceild(n,d)  std::ceil(((double)(n))/((double)(d)))
#define floord(n,d) std::floor(((double)(n))/((double)(d)))

best_clustering_t minimise_binder (const arma::mat& p, const arma::umat& CI, double Const_Binder = 0.5);

arma::urowvec randperm (ind_t N);