// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

type Interface interface {
	Len() int
	Less(i, j int) bool
	Swap(i, j int)
}

func Sort(data Interface) {
	for i := 1; i < data.Len(); i++ {
		for j := i; j > 0 && data.Less(j, j-1); j-- {
			data.Swap(j, j-1)
		}
	}
}

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

type IntSlice []int

func (p IntSlice) Len() int           { return len(p) }
func (p IntSlice) Less(i, j int) bool { return p[i] < p[j] }
func (p IntSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

type Float64Slice []float64

func (p Float64Slice) Len() int           { return len(p) }
func (p Float64Slice) Less(i, j int) bool { return p[i] < p[j] }
func (p Float64Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

type StringSlice []string

func (p StringSlice) Len() int           { return len(p) }
func (p StringSlice) Less(i, j int) bool { return p[i] < p[j] }
func (p StringSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Convenience wrappers for common cases

func SortInts(a []int)         { Sort(IntSlice(a)) }
func SortFloat64s(a []float64) { Sort(Float64Slice(a)) }
func SortStrings(a []string)   { Sort(StringSlice(a)) }

func IntsAreSorted(a []int) bool         { return IsSorted(IntSlice(a)) }
func Float64sAreSorted(a []float64) bool { return IsSorted(Float64Slice(a)) }
func StringsAreSorted(a []string) bool   { return IsSorted(StringSlice(a)) }
