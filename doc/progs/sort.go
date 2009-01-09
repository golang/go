// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

export type SortInterface interface {
	Len() int;
	Less(i, j int) bool;
	Swap(i, j int);
}

export func Sort(data SortInterface) {
	for i := 1; i < data.Len(); i++ {
		for j := i; j > 0 && data.Less(j, j-1); j-- {
			data.Swap(j, j-1);
		}
	}
}

export func IsSorted(data SortInterface) bool {
	n := data.Len();
	for i := n - 1; i > 0; i-- {
		if data.Less(i, i - 1) {
			return false;
		}
	}
	return true;
}

// Convenience types for common cases

export type IntArray []int

func (p IntArray) Len() int            { return len(p); }
func (p IntArray) Less(i, j int) bool  { return p[i] < p[j]; }
func (p IntArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


export type FloatArray []float

func (p FloatArray) Len() int            { return len(p); }
func (p FloatArray) Less(i, j int) bool  { return p[i] < p[j]; }
func (p FloatArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


export type StringArray []string

func (p StringArray) Len() int            { return len(p); }
func (p StringArray) Less(i, j int) bool  { return p[i] < p[j]; }
func (p StringArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


// Convenience wrappers for common cases

export func SortInts(a []int)        { Sort(IntArray(a)); }
export func SortFloats(a []float)    { Sort(FloatArray(a)); }
export func SortStrings(a []string)  { Sort(StringArray(a)); }


export func IntsAreSorted(a []int) bool       { return IsSorted(IntArray(a)); }
export func FloatsAreSorted(a []float) bool   { return IsSorted(FloatArray(a)); }
export func StringsAreSorted(a []string) bool { return IsSorted(StringArray(a)); }
