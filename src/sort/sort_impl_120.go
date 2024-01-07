// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.21

package sort

func intsImpl(x []int)         { Sort(IntSlice(x)) }
func float64sImpl(x []float64) { Sort(Float64Slice(x)) }
func stringsImpl(x []string)   { Sort(StringSlice(x)) }

func intsAreSortedImpl(x []int) bool         { return IsSorted(IntSlice(x)) }
func float64sAreSortedImpl(x []float64) bool { return IsSorted(Float64Slice(x)) }
func stringsAreSortedImpl(x []string) bool   { return IsSorted(StringSlice(x)) }
