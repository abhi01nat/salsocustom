#include "salsocustom.h"

salso_result salso_cpp (const arma::mat& p, ind_t maxClust, double Const_Binder, ind_t nPerm, ind_t nScans) {

	ind_t N = p.n_cols; // number of items
	salso_result result (N, INFINITY);
#pragma omp parallel 
	{
		salso_result partialResult (N, INFINITY); // partial result from each thread

		// iterate over random orderings of of [1, ..., N]
#pragma omp for
		for (ind_t iter = 0; iter < nPerm; ++iter) {

			 /*
			 1. Generate the random item ordering for this iteration. We will assign cluster labels to items in this order.
			 2. Permute the co-clustering probability matrix to match the item ordering for this iteration.
			 */
			arma::urowvec itemOrder = randperm (N);
			arma::mat pOrd = p.submat (itemOrder, itemOrder);

			/* 
			Sequentially cluster all N items as follows:
			1. Assign the 0th item to the 0th cluster.
			2. For k = 1, ..., N-1, given a clustering of the first k-1 items:
				a. Create all possible clusterings of the first k items by varying the kth item label only.
				b. The kth item label can be any of the labels already present, or a new label.
				b. Find the best among these clusterings.
			*/
			arma::urowvec cl (N); // cluster label vector
			cl (0) = 0;
			ind_t currNumClust; // number of clusters so far
			currNumClust = 1;
			ind_t tryNumClust; // number of candidate clusters for next item
			best_clustering_t best_clustering;
			for (ind_t k = 1; k < N; ++k) {
				tryNumClust = std::min (std::min (currNumClust + 1, maxClust), N); 
				arma::umat C = arma::repmat (cl.subvec (0, k), tryNumClust, 1); // candidate clusterings
#pragma omp simd
				for (ind_t t = 0; t < tryNumClust; ++t) C (t, k) = t;
				best_clustering = minimise_binder (pOrd.submat (0, 0, k, k), C, Const_Binder); 
				cl (k) = C (best_clustering.index, k);
				if (best_clustering.index == tryNumClust - 1) currNumClust = tryNumClust; // item k was added to a new cluster
			}

			/*
			1. Perform step 2 nScans times.
			2. For k = 0, ..., N-1:
				a. Consider all possible clusterings of the N items obtained by varying the kth item label only. 
				b. The kth item label can be any of the labels already present, or a new label.
				c. Find the best among these clusterings. 
			3. At the end of a scan if there is no change in clustering over the previous scan, stop scanning.
			*/
			double prevScanBinder = std::numeric_limits<double>::infinity();
			for (ind_t currScan = 0; currScan < nScans; ++currScan) {
				for (ind_t k = 0; k < N; ++k) {
					tryNumClust = std::min (std::min (currNumClust + 1, maxClust), N);
					arma::umat C = arma::repmat (cl, tryNumClust, 1); // candidate clusterings
#pragma omp simd
					for (ind_t t = 0; t < tryNumClust; ++t) C (t, k) = t;
					best_clustering = minimise_binder (pOrd, C, Const_Binder); // best clustering
					cl (k) = C (best_clustering.index, k);
					if (best_clustering.index == tryNumClust - 1) currNumClust = tryNumClust; // item k was added to a new cluster
				}
				if (best_clustering.binder_loss < prevScanBinder) {
					prevScanBinder = best_clustering.binder_loss;
				}
				else {
					break; // no improvement over previous scan
				}
			}

			if (prevScanBinder < partialResult.binder_loss) { // if the current iteration yielded a better clustering
#pragma omp simd
				for (ind_t k = 0; k < N; ++k) partialResult.label (itemOrder (k)) = cl (k); // undo the permutation on the labels
				partialResult.binder_loss = prevScanBinder;
			}
		}
#pragma omp critical 
		{
			result = (partialResult.binder_loss < result.binder_loss ? partialResult : result);
		}
	}
	return result;
}


best_clustering_t  minimise_binder (const arma::mat& p, const arma::umat& CI, double Const_Binder) {

	arma::uword N = CI.n_cols;
	arma::uword iter = CI.n_rows;

	//VERBOSE_DEBUG (" Compute Binder_f    ");

	arma::vec Binder_f (iter, arma::fill::zeros);
	arma::mat tmp = p - Const_Binder;
	{
		/*
		Tiling the iterations:
		numVertTiles is the number of tiles that the iterations are split into (vertical tiles)
		t1 ranges over the number of tiles
		t5 ranges between the lower bound and upper bound of the current vertical tile
		*/
		arma::uword numVertTiles, t1, currVertTileLb, currVertTileUb, t5;
		numVertTiles = ceild (iter - 1, BINDERS_TILE_SIZE);

		/*
		Tiling the items to compare:
		t2 ranges over the number of tiles that the items are split into (horizontal tiles)
		t3 ranges within the current horizontal tile, these are items on one side of the comparison
		t4 tiles the items on the other side of the comparison
		*/
		arma::uword numHorizTiles, t2, currHorizTileLb, currHorizTileUb, t3, t4;
		numHorizTiles = ceild (N - 1, BINDERS_TILE_SIZE);

#pragma omp parallel for private(currVertTileLb,currVertTileUb,t2,t3,t4,t5)
		// iterate over the vertical tiles
		for (t1 = 0; t1 < numVertTiles; ++t1) {
			// compute the bounds of the current vertical tile
			currVertTileLb = BINDERS_TILE_SIZE * t1;
			currVertTileUb = std::min (iter - 1, BINDERS_TILE_SIZE * t1 + (BINDERS_TILE_SIZE - 1));

			//iterate over the horizontal tiles
			for (t2 = 0; t2 < numHorizTiles; t2++) {
				// compute the bounds of the current horizontal tile
				currHorizTileLb = BINDERS_TILE_SIZE * t2;
				currHorizTileUb = std::min (N - 1, BINDERS_TILE_SIZE * t2 + (BINDERS_TILE_SIZE - 1));

				// iterate within the current horizontal tile
				for (t3 = currHorizTileLb; t3 <= currHorizTileUb; ++t3) {
					// iterate over the tiling of items to compare to
					for (t4 = t3; t4 <= N - 1 - 7; t4 += 8) {
#pragma ivdep
#pragma vector always
						for (t5 = currVertTileLb; t5 <= currVertTileUb; ++t5) { //iterate within the current vertical tile
							if (CI (t5, t4 + 0) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4 + 0);
							if (CI (t5, t4 + 1) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4 + 1);
							if (CI (t5, t4 + 2) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4 + 2);
							if (CI (t5, t4 + 3) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4 + 3);
							if (CI (t5, t4 + 4) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4 + 4);
							if (CI (t5, t4 + 5) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4 + 5);
							if (CI (t5, t4 + 6) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4 + 6);
							if (CI (t5, t4 + 7) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4 + 7);
						}
					}
					// iterate over the left over items
					for (; t4 <= N - 1; ++t4) {
#pragma ivdep
#pragma vector always
						for (t5 = currVertTileLb; t5 <= currVertTileUb; ++t5) {
							if (CI (t5, t4) == CI (t5, t3)) Binder_f (t5) += tmp (t3, t4);
						}
					}
				}
			}
		}

	}

	//VERBOSE_DEBUG (" Return the maximum  ");
	arma::uword Binder_ind = arma::index_max (Binder_f); // first occurence of the maximum

	return best_clustering_t { Binder_ind, -Binder_f (Binder_ind) }; // note the negative sign, this is the Binder loss upto an additive constant
}

arma::urowvec randperm (ind_t N) {
	std::random_device rd;
	std::mt19937 mt (rd ());
	arma::urowvec ans (N);
#pragma omp simd
	for (ind_t i = 0; i < N; ++i) ans (i) = i;
	std::shuffle (ans.begin (), ans.end (), mt);
	return ans;
}