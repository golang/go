// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

// Slice sorts the slice x given the provided less function.
// It panics if x is not a slice.
//
// The sort is not guaranteed to be stable: equal elements
// may be reversed from their original order.
// For a stable sort, use SliceStable.
//
// The less function must satisfy the same requirements as
// the Interface type's Less method.
func Slice(x any, less func(i, j int) bool) {
	rv := reflectValueOf(x)
	swap := reflectSwapper(x)
	length := rv.Len()
	quickSort_func(lessSwap{less, swap}, 0, length, maxDepth(length))
}

// SliceStable sorts the slice x using the provided less
// function, keeping equal elements in their original order.
// It panics if x is not a slice.
//
// The less function must satisfy the same requirements as
// the Interface type's Less method.
func SliceStable(x any, less func(i, j int) bool) {
	rv := reflectValueOf(x)
	swap := reflectSwapper(x)
	stable_func(lessSwap{less, swap}, rv.Len())
}

// SliceIsSorted reports whether the slice x is sorted according to the provided less function.
// It panics if x is not a slice.
func SliceIsSorted(x any, less func(i, j int) bool) bool {
	rv := reflectValueOf(x)
	n := rv.Len()
	for i := n - 1; i > 0; i-- {
		if less(i, i-1) {
			return false
		}
	}
	return true
}
