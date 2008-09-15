// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

export type SortInterface interface {
	len() int;
	less(i, j int) bool;
	swap(i, j int);
}

export func Sort(data SortInterface) {
	// Bubble sort for brevity
	for i := 0; i < data.len(); i++ {
		for j := i; j < data.len(); j++ {
			if data.less(j, i) {
				data.swap(i, j)
			}
		}
	}
}

export func IsSorted(data SortInterface) bool {
	n := data.len();
	for i := n - 1; i > 0; i-- {
		if data.less(i, i - 1) {
			return false;
		}
	}
	return true;
}

// Convenience types for common cases

export type IntArray struct {
	data *[]int;
}

func (p *IntArray) len() int            { return len(p.data); }
func (p *IntArray) less(i, j int) bool  { return p.data[i] < p.data[j]; }
func (p *IntArray) swap(i, j int)       { p.data[i], p.data[j] = p.data[j], p.data[i]; }


export type FloatArray struct {
	data *[]float;
}

func (p *FloatArray) len() int            { return len(p.data); }
func (p *FloatArray) less(i, j int) bool  { return p.data[i] < p.data[j]; }
func (p *FloatArray) swap(i, j int)       { p.data[i], p.data[j] = p.data[j], p.data[i]; }


export type StringArray struct {
	data *[]string;
}

func (p *StringArray) len() int            { return len(p.data); }
func (p *StringArray) less(i, j int) bool  { return p.data[i] < p.data[j]; }
func (p *StringArray) swap(i, j int)       { p.data[i], p.data[j] = p.data[j], p.data[i]; }


// Convenience wrappers for common cases

export func SortInts(a *[]int)        { Sort(&IntArray{a}); }
export func SortFloats(a *[]float)    { Sort(&FloatArray{a}); }
export func SortStrings(a *[]string)  { Sort(&StringArray{a}); }


export func IntsAreSorted(a *[]int) bool       { return IsSorted(&IntArray{a}); }
export func FloatsAreSorted(a *[]float) bool   { return IsSorted(&FloatArray{a}); }
export func StringsAreSorted(a *[]string) bool { return IsSorted(&StringArray{a}); }
