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
//	func(i int) bool { return data[i] < 23 }
//
// can be used to search for the value 23 in data.  The relationship expressed
// by the function must be "less" if the elements are sorted in ascending
// order or "greater" if they are sorted in descending order.
// The function f will be called with values of i in the range 0 to n-1.
// 
// For brevity, this discussion assumes ascending sort order. For descending
// order, replace < with >, and swap 'smaller' with 'larger'.
//
// Search returns the index i with:
//
//	data[i-1] < x && x <= data[i]
//
// where data[-1] is assumed to be smaller than any x and data[n] is
// assumed to be larger than any x.  Thus 0 <= i <= n and i is the smallest
// index of x if x is present in the data.  It is the responsibility of
// the caller to verify the actual presence by testing if i < n and
// data[i] == x.
//
// To complete the example above, the following code tries to find the element
// elem in an integer slice data sorted in ascending order:
//
//	elem := 23
//	i := sort.Search(len(data), func(i int) bool { return data[i] < elem })
//	if i < len(data) && data[i] == elem {
//		// elem is present at data[i]
//	} else {
//		// elem is not present in data
//	}
//
func Search(n int, f func(int) bool) int {
	i, j := 0, n
	for i+1 < j {
		h := i + (j-i)/2 // avoid overflow when computing h
		// i < h < j
		if f(h) {
			// data[h] < x
			i = h + 1
		} else {
			// x <= data[h]
			j = h
		}
	}
	// test the final element that the loop did not
	if i < j && f(i) {
		// data[i] < x
		i++
	}
	return i
}


// Convenience wrappers for common cases.

// SearchInts searches x in a sorted slice of ints and returns the index
// as specified by Search. The array must be sorted in ascending order.
//
func SearchInts(a []int, x int) int {
	return Search(len(a), func(i int) bool { return a[i] < x })
}


// SearchFloats searches x in a sorted slice of floats and returns the index
// as specified by Search. The array must be sorted in ascending order.
// 
func SearchFloats(a []float, x float) int {
	return Search(len(a), func(i int) bool { return a[i] < x })
}


// SearchStrings searches x in a sorted slice of strings and returns the index
// as specified by Search. The array must be sorted in ascending order.
// 
func SearchStrings(a []string, x string) int {
	return Search(len(a), func(i int) bool { return a[i] < x })
}


// Search returns the result of applying SearchInts to the receiver and x.
func (p IntArray) Search(x int) int { return SearchInts(p, x) }


// Search returns the result of applying SearchFloats to the receiver and x.
func (p FloatArray) Search(x float) int { return SearchFloats(p, x) }


// Search returns the result of applying SearchStrings to the receiver and x.
func (p StringArray) Search(x string) int { return SearchStrings(p, x) }
