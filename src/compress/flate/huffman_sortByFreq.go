// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// Sort sorts data.
// It makes one call to data.Len to determine n, and O(n*log(n)) calls to
// data.Less and data.Swap. The sort is not guaranteed to be stable.
func sortByFreq(data []literalNode) {
	n := len(data)
	quickSortByFreq(data, 0, n, maxDepth(n))
}

func quickSortByFreq(data []literalNode, a, b, maxDepth int) {
	for b-a > 12 { // Use ShellSort for slices <= 12 elements
		if maxDepth == 0 {
			heapSort(data, a, b)
			return
		}
		maxDepth--
		mlo, mhi := doPivotByFreq(data, a, b)
		// Avoiding recursion on the larger subproblem guarantees
		// a stack depth of at most lg(b-a).
		if mlo-a < b-mhi {
			quickSortByFreq(data, a, mlo, maxDepth)
			a = mhi // i.e., quickSortByFreq(data, mhi, b)
		} else {
			quickSortByFreq(data, mhi, b, maxDepth)
			b = mlo // i.e., quickSortByFreq(data, a, mlo)
		}
	}
	if b-a > 1 {
		// Do ShellSort pass with gap 6
		// It could be written in this simplified form cause b-a <= 12
		for i := a + 6; i < b; i++ {
			if data[i].freq == data[i-6].freq && data[i].literal < data[i-6].literal || data[i].freq < data[i-6].freq {
				data[i], data[i-6] = data[i-6], data[i]
			}
		}
		insertionSortByFreq(data, a, b)
	}
}

func doPivotByFreq(data []literalNode, lo, hi int) (midlo, midhi int) {
	m := int(uint(lo+hi) >> 1) // Written like this to avoid integer overflow.
	if hi-lo > 40 {
		// Tukey's ``Ninther,'' median of three medians of three.
		s := (hi - lo) / 8
		medianOfThreeSortByFreq(data, lo, lo+s, lo+2*s)
		medianOfThreeSortByFreq(data, m, m-s, m+s)
		medianOfThreeSortByFreq(data, hi-1, hi-1-s, hi-1-2*s)
	}
	medianOfThreeSortByFreq(data, lo, m, hi-1)

	// Invariants are:
	//	data[lo] = pivot (set up by ChoosePivot)
	//	data[lo < i < a] < pivot
	//	data[a <= i < b] <= pivot
	//	data[b <= i < c] unexamined
	//	data[c <= i < hi-1] > pivot
	//	data[hi-1] >= pivot
	pivot := lo
	a, c := lo+1, hi-1

	for ; a < c && (data[a].freq == data[pivot].freq && data[a].literal < data[pivot].literal || data[a].freq < data[pivot].freq); a++ {
	}
	b := a
	for {
		for ; b < c && (data[pivot].freq == data[b].freq && data[pivot].literal > data[b].literal || data[pivot].freq > data[b].freq); b++ { // data[b] <= pivot
		}
		for ; b < c && (data[pivot].freq == data[c-1].freq && data[pivot].literal < data[c-1].literal || data[pivot].freq < data[c-1].freq); c-- { // data[c-1] > pivot
		}
		if b >= c {
			break
		}
		// data[b] > pivot; data[c-1] <= pivot
		data[b], data[c-1] = data[c-1], data[b]
		b++
		c--
	}
	// If hi-c<3 then there are duplicates (by property of median of nine).
	// Let's be a bit more conservative, and set border to 5.
	protect := hi-c < 5
	if !protect && hi-c < (hi-lo)/4 {
		// Lets test some points for equality to pivot
		dups := 0
		if data[pivot].freq == data[hi-1].freq && data[pivot].literal > data[hi-1].literal || data[pivot].freq > data[hi-1].freq { // data[hi-1] = pivot
			data[c], data[hi-1] = data[hi-1], data[c]
			c++
			dups++
		}
		if data[b-1].freq == data[pivot].freq && data[b-1].literal > data[pivot].literal || data[b-1].freq > data[pivot].freq { // data[b-1] = pivot
			b--
			dups++
		}
		// m-lo = (hi-lo)/2 > 6
		// b-lo > (hi-lo)*3/4-1 > 8
		// ==> m < b ==> data[m] <= pivot
		if data[m].freq == data[pivot].freq && data[m].literal > data[pivot].literal || data[m].freq > data[pivot].freq { // data[m] = pivot
			data[m], data[b-1] = data[b-1], data[m]
			b--
			dups++
		}
		// if at least 2 points are equal to pivot, assume skewed distribution
		protect = dups > 1
	}
	if protect {
		// Protect against a lot of duplicates
		// Add invariant:
		//	data[a <= i < b] unexamined
		//	data[b <= i < c] = pivot
		for {
			for ; a < b && (data[b-1].freq == data[pivot].freq && data[b-1].literal > data[pivot].literal || data[b-1].freq > data[pivot].freq); b-- { // data[b] == pivot
			}
			for ; a < b && (data[a].freq == data[pivot].freq && data[a].literal < data[pivot].literal || data[a].freq < data[pivot].freq); a++ { // data[a] < pivot
			}
			if a >= b {
				break
			}
			// data[a] == pivot; data[b-1] < pivot
			data[a], data[b-1] = data[b-1], data[a]
			a++
			b--
		}
	}
	// Swap pivot into middle
	data[pivot], data[b-1] = data[b-1], data[pivot]
	return b - 1, c
}

// Insertion sort
func insertionSortByFreq(data []literalNode, a, b int) {
	for i := a + 1; i < b; i++ {
		for j := i; j > a && (data[j].freq == data[j-1].freq && data[j].literal < data[j-1].literal || data[j].freq < data[j-1].freq); j-- {
			data[j], data[j-1] = data[j-1], data[j]
		}
	}
}

// quickSortByFreq, loosely following Bentley and McIlroy,
// ``Engineering a Sort Function,'' SP&E November 1993.

// medianOfThreeSortByFreq moves the median of the three values data[m0], data[m1], data[m2] into data[m1].
func medianOfThreeSortByFreq(data []literalNode, m1, m0, m2 int) {
	// sort 3 elements
	if data[m1].freq == data[m0].freq && data[m1].literal < data[m0].literal || data[m1].freq < data[m0].freq {
		data[m1], data[m0] = data[m0], data[m1]
	}
	// data[m0] <= data[m1]
	if data[m2].freq == data[m1].freq && data[m2].literal < data[m1].literal || data[m2].freq < data[m1].freq {
		data[m2], data[m1] = data[m1], data[m2]
		// data[m0] <= data[m2] && data[m1] < data[m2]
		if data[m1].freq == data[m0].freq && data[m1].literal < data[m0].literal || data[m1].freq < data[m0].freq {
			data[m1], data[m0] = data[m0], data[m1]
		}
	}
	// now data[m0] <= data[m1] <= data[m2]
}
