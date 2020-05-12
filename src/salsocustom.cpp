#include "salsocustom.h"
#include "debug.h"

double negative_infinity = -std::numeric_limits<double>::infinity ();

double binder_single (const arma::mat& p, const std::vector<ind_t>& CI) {

	ind_t N = CI.size ();
	double Binder_f = 0;
	for (ind_t j = 0; j < N - 1; ++j) {
#pragma omp simd
		for (ind_t k = j + 1; k < N; ++k) {
			if (CI [j] == CI [k]) Binder_f += p (j, k);
		}
	}
	return Binder_f;
}

salso_result salso_cpp (const arma::mat& epam, ind_t maxClust, double Const_Binder, ind_t batchSize, ind_t nScans) {

	arma::mat p = epam - Const_Binder; // we never use epam directly
	ind_t N = p.n_cols; // number of items
	salso_result result (N, 0, negative_infinity);
	maxClust = std::min (maxClust, N); // never need more than N clusters
	int numThreads;
#pragma omp parallel 
	{
#pragma omp single 
		{
			numThreads = omp_get_num_threads ();
			if (numThreads == 1) {
				message_stream << "Using 1 thread.\n";
			}
			else {
				message_stream << "Using " << numThreads << " threads.\n";
			}
		}

		salso_result partialResult (N, 0, negative_infinity); // partial result from each thread
		std::vector<ind_t> cl (N, 0); // cluster label vectors
		ind_t currNumClust, tryNumClust; // number of clusters so far, number of clusters to try for next item
		
		// iterate over random orderings of of [1, ..., N]
		for (ind_t iter = 0; iter < batchSize; ++iter) {
			/* BEGIN ITERATION */

			arma::uvec itemOrder (randperm (N)); // Generate the random item ordering for this iteration. We will assign cluster labels to items in this order.
			arma::mat pOrd = p(itemOrder, itemOrder); // Permute the co-clustering probability matrix to match the item ordering for this iteration.
			// message_stream << "Permutation: "<< itemOrder;

			/* FIRST SEQUENTIAL ALLOCATION
			Sequentially allocate labels to all N items as follows:
			1. Assign the 0th item to the 0th cluster.
			2. For k = 1, ..., N-1, given a clustering of the first k-1 items:
				a. Create all possible clusterings of the first k items by varying the kth item label only.
				b. The kth item label can be any of the labels already present, or a new label.
				b. Find the best among these clusterings.
			*/
			std::fill (cl.begin (), cl.end (), 0); // cl stores our current best labelling
			std::vector<std::vector<ind_t>> labelIndex (maxClust); // labelIndex[t] stores the indices of the items with label t
			labelIndex[0].push_back (0); // item 0 gets label 0 to begin with
			currNumClust = 1; // currently only using 1 cluster			

			/* FIRST SEQUENTIAL ALLOCATION */
			for (ind_t k = 1; k < N; ++k) { 
				/* ITERATE OVER ITEMS */

				tryNumClust = std::min (currNumClust + 1, maxClust); 
				double bestLabelScore = negative_infinity, tmpLabelScore;
				ind_t bestLabel;

				for (ind_t t = 0; t < tryNumClust; ++t) { 
					/* ITERATE OVER CANDIDATE LABELS */

					tmpLabelScore = arma::accu (pOrd.unsafe_col (k).elem (arma::uvec(labelIndex[t]))); // change in Binder score if we use label t for item k
					if (tmpLabelScore > bestLabelScore) { // want to maximise the change
						bestLabelScore = tmpLabelScore;
						bestLabel = t;
					}
					/* END CURRENT LABEL */
				} 
				labelIndex[bestLabel].push_back (k); // item k has label bestLabel
				cl[k] = bestLabel; // item k has label bestLabel
				if (bestLabel == tryNumClust - 1) currNumClust++; // item was assigned a label not currently in our set of labels

				/* END CURRENT ITEM */
			} 
			/* END FIRST SEQUENTIAL ALLOCATION */

			/* SWEETENING SCANS
			1. Perform step 2 nScans times.
			2. For k = 0, ..., N-1:
				a. Consider all possible clusterings of the N items obtained by varying the kth item label only.
				b. The kth item label can be any of the labels already present, or a new label.
				c. Find the best among these clusterings.
			3. At the end of a scan if there is no change in clustering over the previous scan, stop scanning.
			*/

			double thisScanDeltaBinder = 0; // change in Binder score from this scan
			for (ind_t currScan = 0; currScan < nScans; ++currScan) {
				/* BEGIN SCAN */
				
				for (ind_t k = 0; k < N; ++k) { 
					/* ITERATE OVER ITEMS */

					tryNumClust = std::min (currNumClust + 1, maxClust);
					double bestLabelScore = negative_infinity, tmpLabelScore;
					ind_t bestLabel;
					for (ind_t t = 0; t < tryNumClust; ++t) { 
						/* ITERATE OVER CANDIDATE LABELS */

						if (t != cl[k]) tmpLabelScore = arma::accu (pOrd.unsafe_col (k).elem (arma::uvec (labelIndex[t]))); // change in Binder score if we change to label t for item k (upto additive constant)
						else tmpLabelScore = 0; // no change in Binder score if no change in label
						
						if (tmpLabelScore > bestLabelScore) { // want to maximise the change
							bestLabelScore = tmpLabelScore;
							bestLabel = t;
						}

						/* END CURRENT LABEL */
					} 
					if (bestLabel == cl[k]) continue; // no change in label
					labelIndex[cl[k]].erase (std::find (labelIndex[cl[k]].begin (), labelIndex[cl[k]].end (), k)); // remove current label for item k
					labelIndex[bestLabel].push_back (k); // make label bestLabel
					cl[k] = bestLabel; // item k has label bestLabel
					if (bestLabel == tryNumClust - 1) currNumClust++; // item was assigned a label not currently in our set of labels
					thisScanDeltaBinder += bestLabelScore;

					/* END CURRENT ITEM */
				} 
				if (thisScanDeltaBinder == 0) break; // no change in Binder score from the scan

				/* END SCAN */
			} 
			double currIterbinderLoss = binder_single (pOrd, cl);
			if (currIterbinderLoss > partialResult.binderLoss) { // if the current iteration yielded a better clustering
#pragma omp simd
				for (ind_t k = 0; k < N; ++k) partialResult.label [itemOrder [k]] = cl [k]; // undo the permutation on the labels
				partialResult.binderLoss = currIterbinderLoss;
				partialResult.numClust = currNumClust;
			}
			// message_stream << "Current iteration labels: " << partialResult.label;

			/* END ITERATION */
		}
#pragma omp critical 
		{
			result = (partialResult.binderLoss > result.binderLoss ? partialResult : result);
		}
	}

	// canonicalise labels
	std::vector<ind_t> sortedLabels (N, 0), labelPerm(result.numClust, 0);
	for (ind_t i = 0, c = 0; i < N; ++i) {
		if (labelPerm[result.label[i]] == 0) {
			labelPerm[result.label[i]] = ++c;
		}
	}
#pragma omp simd
	for (ind_t i = 0; i < N; ++i) sortedLabels[i] = labelPerm[result.label[i]];
	result.label = std::move(sortedLabels);
	return result;
}

std::vector<ind_t> randperm (ind_t N) {
	std::random_device rd;
	std::mt19937 mt (rd ());
	std::vector<ind_t> ans (N);
	std::iota (ans.begin (), ans.end (), 0); // sequentially fill values
	std::shuffle (ans.begin (), ans.end (), mt);
	return ans;
}

