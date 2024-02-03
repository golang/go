// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21

// Starting with Go 1.21, we can leverage the new generic functions from the
// slices package to implement some `sort` functions faster. However, until
// the bootstrap compiler uses Go 1.21 or later, we keep a fallback version
// in sort_impl_120.go that retains the old implementation.

package sort

import "slices"

func intsImpl(x []int)         { slices.Sort(x) }
func float64sImpl(x []float64) { slices.Sort(x) }
func stringsImpl(x []string)   { slices.Sort(x) }

func intsAreSortedImpl(x []int) bool         { return slices.IsSorted(x) }
func float64sAreSortedImpl(x []float64) bool { return slices.IsSorted(x) }
func stringsAreSortedImpl(x []string) bool   { return slices.IsSorted(x) }
