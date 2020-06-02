#include "debug.h"

#ifndef HAS_RCPP
std::ostream& messageStream = std::cout;
#else
std::ostream& messageStream = Rcpp::Rcout;
#endif

std::string to_string (const std::vector<unsigned int> v) {
	std::stringstream s;
	size_t N = v.size ();
	for (size_t i = 0; i < N; ++i) {
		s << v[i] << ' ';
	}
	return s.str ();
}

