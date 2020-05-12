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

salso_result_t salsoCpp (const arma::mat& epam, ind_t maxClusts, double Const_Binder, ind_t batchSize, ind_t nScans, unsigned int maxThreads, unsigned int timeLimit) {

	arma::mat p = epam - Const_Binder; // we rarely use epam directly
	ind_t N = p.n_cols; // number of items
	salso_result_t result (N);
	if (maxClusts == 0) maxClusts = N;
	else maxClusts = std::min (maxClusts, N); // never need more than N clusters

	if (maxThreads > 0) omp_set_num_threads(maxThreads); 
	int numThreads;
#pragma omp parallel 
	{
#pragma omp single 
		{
			message_stream << "Begin clustering ";
			numThreads = omp_get_num_threads ();
			result.numThreads = numThreads;
			if (numThreads == 1) {
				message_stream << "using 1 thread.\n";
			}
			else {
				message_stream << "using " << numThreads << " threads.\n";
			}
			message_stream << "Number of permutations to search: " << numThreads * batchSize << '\n';
		}
    auto timeStart = std::chrono::high_resolution_clock::now();
  
    salso_result_t partialResult (N); // partial result from each thread
		std::vector<ind_t> cl (N, 0); // cluster label vectors
		
		// iterate over random orderings of of [1, ..., N]
		while (true) {
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
			std::vector<std::vector<ind_t>> labelIndices (maxClusts); // labelIndices[t] stores the indices of the items with label t
			labelIndices[0].push_back (0); // item 0 gets label 0 to begin with
			ind_t currNumClusts = 1, trynumClusts; // currently only using 1 cluster			

			/* FIRST SEQUENTIAL ALLOCATION */
			for (ind_t k = 1; k < N; ++k) { 
				/* ITERATE OVER ITEMS */

				trynumClusts = std::min (currNumClusts + 1, maxClusts); 
				double bestLabelScore = negative_infinity, tmpLabelScore;
				ind_t bestLabel;

				for (ind_t t = 0; t < trynumClusts; ++t) { 
					/* ITERATE OVER CANDIDATE LABELS */

					tmpLabelScore = arma::accu (pOrd.unsafe_col (k).elem (arma::uvec(labelIndices[t]))); // change in Binder score if we use label t for item k
					if (tmpLabelScore > bestLabelScore) { // want to maximise the change
						bestLabelScore = tmpLabelScore;
						bestLabel = t;
					}
					/* END CURRENT LABEL */
				} 
				labelIndices[bestLabel].push_back (k); // item k has label bestLabel
				cl[k] = bestLabel; // item k has label bestLabel
				if (bestLabel == trynumClusts - 1) currNumClusts++; // item was assigned a label not currently in our set of labels

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

					trynumClusts = std::min (currNumClusts + 1, maxClusts);
					double bestLabelScore = negative_infinity, tmpLabelScore;
					ind_t bestLabel;
					for (ind_t t = 0; t < trynumClusts; ++t) { 
						/* ITERATE OVER CANDIDATE LABELS */

						if (t != cl[k]) tmpLabelScore = arma::accu (pOrd.unsafe_col (k).elem (arma::uvec (labelIndices[t]))); // change in Binder score if we change to label t for item k (upto additive constant)
						else tmpLabelScore = 0; // no change in Binder score if no change in label
						
						if (tmpLabelScore > bestLabelScore) { // want to maximise the change
							bestLabelScore = tmpLabelScore;
							bestLabel = t;
						}

						/* END CURRENT LABEL */
					} 
					if (bestLabel == cl[k]) continue; // no change in label
					labelIndices[cl[k]].erase (std::find (labelIndices[cl[k]].begin (), labelIndices[cl[k]].end (), k)); // remove current label for item k
					labelIndices[bestLabel].push_back (k); // make label bestLabel
					cl[k] = bestLabel; // item k has label bestLabel
					if (bestLabel == trynumClusts - 1) currNumClusts++; // item was assigned a label not currently in our set of labels
					thisScanDeltaBinder += bestLabelScore;

					/* END CURRENT ITEM */
				} 
				if (thisScanDeltaBinder == 0) break; // no change in Binder score from the scan

				/* END SCAN */
			} 
			double currIterbinderLoss = binder_single (pOrd, cl);
			if (currIterbinderLoss > partialResult.binderLoss) { // if the current iteration yielded a better clustering
#pragma omp simd
				for (ind_t k = 0; k < N; ++k) partialResult.labels [itemOrder [k]] = cl [k]; // undo the permutation on the labels
				partialResult.binderLoss = currIterbinderLoss;
				partialResult.numClusts = currNumClusts;
			}
			// message_stream << "Current iteration labels: " << partialResult.labels;
			
			/* LOOP EXIT CONDITIONS AND BOOKKEEPING */
			++partialResult.nIters;
			auto timeNow = std::chrono::high_resolution_clock::now ();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow - timeStart).count ();
			
			/* CHECK ITERATION COUNT */
			if (partialResult.nIters >= batchSize && batchSize > 0){
			  partialResult.wallClockTime = duration;
			  if (timeLimit > 0 && duration >= timeLimit) partialResult.timeLimitReached = true;
			  break;
			}
			
			/* CHECK TIMEOUT */
			if (timeLimit > 0 && duration >= timeLimit) {
			  partialResult.wallClockTime = duration;
			  partialResult.timeLimitReached = true;
			  break;
			}

			/* END ITERATION */
		}

#pragma omp critical 
		{
      result.nIters += partialResult.nIters;
      result.wallClockTime += partialResult.wallClockTime;
      result.timeLimitReached |= partialResult.timeLimitReached;
			if (partialResult.binderLoss > result.binderLoss) {
			  result.labels = std::move(partialResult.labels);
			  result.numClusts = partialResult.numClusts;
			  result.binderLoss = partialResult.binderLoss;
			}
		}
	}

	/* CANONICALISE LABELS STARTING AT 1 */
	std::vector<ind_t> sortedLabels (N, 0), labelPerm(result.numClusts, 0);
	for (ind_t i = 0, c = 0; i < N; ++i) {
		if (labelPerm[result.labels[i]] == 0) {
			labelPerm[result.labels[i]] = ++c;
		}
	}
#pragma omp simd
	for (ind_t i = 0; i < N; ++i) sortedLabels[i] = labelPerm[result.labels[i]];
	result.labels = std::move(sortedLabels);
	
	/* ADJUST THE BINDER LOSS TO CORRECTLY REFLECT THE SCORE */
	result.binderLoss = -result.binderLoss + (1 - Const_Binder) * arma::accu(epam);
	
	/* OUTPUT IF NOT IN R */
#ifndef HAS_RCPP
	message_stream << "Cluster labels:\n" << result.labels;
	message_stream << "Finished clustering, found " << result.numClusts << " clusters.\n";
	message_stream << "Normalised binder loss: " << result.binderLoss << '\n';
	message_stream << "Number of permutations scanned: " << result.nIters << '\n';
	message_stream << "Time limit reached: " << result.timeLimitReached;
#endif
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

