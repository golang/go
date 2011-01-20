// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The sort package provides primitives for sorting arrays
// and user-defined collections.
package sort

// A type, typically a collection, that satisfies sort.Interface can be
// sorted by the routines in this package.  The methods require that the
// elements of the collection be enumerated by an integer index.
type Interface interface {
	// Len is the number of elements in the collection.
	Len() int
	// Less returns whether the element with index i should sort
	// before the element with index j.
	Less(i, j int) bool
	// Swap swaps the elements with indexes i and j.
	Swap(i, j int)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Insertion sort
func insertionSort(data Interface, a, b int) {
	for i := a + 1; i < b; i++ {
		for j := i; j > a && data.Less(j, j-1); j-- {
			data.Swap(j, j-1)
		}
	}
}

// Quicksort, following Bentley and McIlroy,
// ``Engineering a Sort Function,'' SP&E November 1993.

// Move the median of the three values data[a], data[b], data[c] into data[a].
func medianOfThree(data Interface, a, b, c int) {
	m0 := b
	m1 := a
	m2 := c
	// bubble sort on 3 elements
	if data.Less(m1, m0) {
		data.Swap(m1, m0)
	}
	if data.Less(m2, m1) {
		data.Swap(m2, m1)
	}
	if data.Less(m1, m0) {
		data.Swap(m1, m0)
	}
	// now data[m0] <= data[m1] <= data[m2]
}

func swapRange(data Interface, a, b, n int) {
	for i := 0; i < n; i++ {
		data.Swap(a+i, b+i)
	}
}

func doPivot(data Interface, lo, hi int) (midlo, midhi int) {
	m := lo + (hi-lo)/2 // Written like this to avoid integer overflow.
	if hi-lo > 40 {
		// Tukey's ``Ninther,'' median of three medians of three.
		s := (hi - lo) / 8
		medianOfThree(data, lo, lo+s, lo+2*s)
		medianOfThree(data, m, m-s, m+s)
		medianOfThree(data, hi-1, hi-1-s, hi-1-2*s)
	}
	medianOfThree(data, lo, m, hi-1)

	// Invariants are:
	//	data[lo] = pivot (set up by ChoosePivot)
	//	data[lo <= i < a] = pivot
	//	data[a <= i < b] < pivot
	//	data[b <= i < c] is unexamined
	//	data[c <= i < d] > pivot
	//	data[d <= i < hi] = pivot
	//
	// Once b meets c, can swap the "= pivot" sections
	// into the middle of the array.
	pivot := lo
	a, b, c, d := lo+1, lo+1, hi, hi
	for b < c {
		if data.Less(b, pivot) { // data[b] < pivot
			b++
			continue
		}
		if !data.Less(pivot, b) { // data[b] = pivot
			data.Swap(a, b)
			a++
			b++
			continue
		}
		if data.Less(pivot, c-1) { // data[c-1] > pivot
			c--
			continue
		}
		if !data.Less(c-1, pivot) { // data[c-1] = pivot
			data.Swap(c-1, d-1)
			c--
			d--
			continue
		}
		// data[b] > pivot; data[c-1] < pivot
		data.Swap(b, c-1)
		b++
		c--
	}

	n := min(b-a, a-lo)
	swapRange(data, lo, b-n, n)

	n = min(hi-d, d-c)
	swapRange(data, c, hi-n, n)

	return lo + b - a, hi - (d - c)
}

func quickSort(data Interface, a, b int) {
	for b-a > 7 {
		mlo, mhi := doPivot(data, a, b)
		// Avoiding recursion on the larger subproblem guarantees
		// a stack depth of at most lg(b-a).
		if mlo-a < b-mhi {
			quickSort(data, a, mlo)
			a = mhi // i.e., quickSort(data, mhi, b)
		} else {
			quickSort(data, mhi, b)
			b = mlo // i.e., quickSort(data, a, mlo)
		}
	}
	if b-a > 1 {
		insertionSort(data, a, b)
	}
}

func Sort(data Interface) { quickSort(data, 0, data.Len()) }


func IsSorted(data Interface) bool {
	n := data.Len()
	for i := n - 1; i > 0; i-- {
		if data.Less(i, i-1) {
			return false
		}
	}
	return true
}


// Convenience types for common cases

// IntArray attaches the methods of Interface to []int, sorting in increasing order.
type IntArray []int

func (p IntArray) Len() int           { return len(p) }
func (p IntArray) Less(i, j int) bool { return p[i] < p[j] }
func (p IntArray) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Sort is a convenience method.
func (p IntArray) Sort() { Sort(p) }


// Float64Array attaches the methods of Interface to []float64, sorting in increasing order.
type Float64Array []float64

func (p Float64Array) Len() int           { return len(p) }
func (p Float64Array) Less(i, j int) bool { return p[i] < p[j] }
func (p Float64Array) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Sort is a convenience method.
func (p Float64Array) Sort() { Sort(p) }


// StringArray attaches the methods of Interface to []string, sorting in increasing order.
type StringArray []string

func (p StringArray) Len() int           { return len(p) }
func (p StringArray) Less(i, j int) bool { return p[i] < p[j] }
func (p StringArray) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Sort is a convenience method.
func (p StringArray) Sort() { Sort(p) }


// Convenience wrappers for common cases

// SortInts sorts an array of ints in increasing order.
func SortInts(a []int) { Sort(IntArray(a)) }
// SortFloat64s sorts an array of float64s in increasing order.
func SortFloat64s(a []float64) { Sort(Float64Array(a)) }
// SortStrings sorts an array of strings in increasing order.
func SortStrings(a []string) { Sort(StringArray(a)) }


// IntsAreSorted tests whether an array of ints is sorted in increasing order.
func IntsAreSorted(a []int) bool { return IsSorted(IntArray(a)) }
// Float64sAreSorted tests whether an array of float64s is sorted in increasing order.
func Float64sAreSorted(a []float64) bool { return IsSorted(Float64Array(a)) }
// StringsAreSorted tests whether an array of strings is sorted in increasing order.
func StringsAreSorted(a []string) bool { return IsSorted(StringArray(a)) }
