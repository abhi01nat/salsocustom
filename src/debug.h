#pragma once
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#ifdef HAS_RCPP
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

std::string to_string (const std::vector<size_t> v);

extern std::ostream& message_stream;