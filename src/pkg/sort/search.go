// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements binary search.

package sort

// Search uses binary search to find the index i for a value x in an indexable
// and sorted data structure of n elements.  The argument function f captures
// the value to be searched for, how the elements are indexed, and how they are
// sorted.  It will often be passed as a closure.  For instance, given a slice
// of integers, []data, sorted in ascending order, the function
//
//	func(i int) bool { return data[i] <= 23 }
//
// can be used to search for the value 23 in data.  The relationship expressed
// by the function must be "less or equal" if the elements are sorted in ascending
// order or "greater or equal" if they are sorted in descending order.
// The function f will be called with values of i in the range 0 to n-1.
// 
// For brevity, this discussion assumes ascending sort order. For descending
// order, replace <= with >=, and swap 'smaller' with 'larger'.
//
// If data[0] <= x and x <= data[n-1], Search returns the index i with:
//
//	data[i] <= x && x <= data[i+1]
//
// where data[n] is assumed to be larger than any x.  Thus, i is the index of x
// if it is present in the data.  It is the responsibility of the caller to
// verify the actual presence by testing if data[i] == x.
//
// If n == 0 or if x is smaller than any element in data (f is always false),
// the result is 0.  If x is larger than any element in data (f is always true),
// the result is n-1.
//
// To complete the example above, the following code tries to find the element
// elem in an integer slice data sorted in ascending order:
//
//	elem := 23
//	i := sort.Search(len(data), func(i int) bool { return data[i] <= elem })
//	if len(data) > 0 && data[i] == elem {
//		// elem is present at data[i]
//	} else {
//		// elem is not present in data
//	}
//
func Search(n int, f func(int) bool) int {
	// See "A Method of Programming", E.W. Dijkstra,
	// for arguments on correctness and efficiency.
	i, j := 0, n
	for i+1 < j {
		h := i + (j-i)/2 // avoid overflow when computing h
		// i < h < j
		if f(h) {
			// data[h] <= x
			i = h
		} else {
			// x < data[h]
			j = h
		}
	}
	return i
}


// Convenience wrappers for common cases.

// SearchInts searches x in a sorted slice of ints and returns the index
// as specified by Search. The array must be sorted in ascending order.
//
func SearchInts(a []int, x int) int {
	return Search(len(a), func(i int) bool { return a[i] <= x })
}


// SearchFloats searches x in a sorted slice of floats and returns the index
// as specified by Search. The array must be sorted in ascending order.
// 
func SearchFloats(a []float, x float) int {
	return Search(len(a), func(i int) bool { return a[i] <= x })
}


// SearchStrings searches x in a sorted slice of strings and returns the index
// as specified by Search. The array must be sorted in ascending order.
// 
func SearchStrings(a []string, x string) int {
	return Search(len(a), func(i int) bool { return a[i] <= x })
}


// Search returns the result of applying SearchInts to the receiver and x.
func (p IntArray) Search(x int) int { return SearchInts(p, x) }


// Search returns the result of applying SearchFloats to the receiver and x.
func (p FloatArray) Search(x float) int { return SearchFloats(p, x) }


// Search returns the result of applying SearchStrings to the receiver and x.
func (p StringArray) Search(x string) int { return SearchStrings(p, x) }
