#pragma once
#include <algorithm>
#include <numeric>
#include <vector>
#include <random>
#include <limits>
#include <omp.h>

#ifndef HAS_RCPP
#include <armadillo>
// [[Rcpp::depends(RcppArmadillo)]]
#define ARMA_NO_DEBUG
#else
#include <RcppArmadillo>
#endif

typedef size_t ind_t; // type for all counters, indices, and labels
static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
extern double negative_infinity;

struct salso_result {
	arma::Row<ind_t> label;
	ind_t numClust;
	double binderLoss;
	salso_result (ind_t numElem, ind_t numClust, double L) : label (numElem), numClust (numClust), binderLoss (L) {}
};
salso_result salso_cpp (const arma::mat& p, ind_t maxClust, double Const_Binder = 0.5, ind_t batchSize = 100, ind_t nScans = 10);

std::vector<ind_t> randperm (ind_t N);