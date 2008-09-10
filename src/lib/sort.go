// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Sort

export type SortInterface interface {
	len() int;
	less(i, j int) bool;
	swap(i, j int);
}


func Pivot(data SortInterface, a, b int) int {
	// if we have at least 10 elements, find a better median
	// by selecting the median of 3 elements and putting it
	// at position a
	if b - a >= 10 {
		m0 := (a + b) / 2;
		m1 := a;
		m2 := b - 1;
		// bubble sort on 3 elements
		if data.less(m1, m0) { data.swap(m1, m0); }
		if data.less(m2, m1) { data.swap(m2, m1); }
		if data.less(m1, m0) { data.swap(m1, m0); }
		// "m0 <= m1 <= m2"
	}
	
	m := a;
	for i := a + 1; i < b; i++ {
		if data.less(i, a) {
			m++;
			data.swap(i, m);
		}
	}
	data.swap(a, m);
	
	return m;
}


func Quicksort(data SortInterface, a, b int) {
	if a + 1 < b {
		m := Pivot(data, a, b);
		Quicksort(data, 0, m);
		Quicksort(data, m + 1, b);
	}
}


export func Sort(data SortInterface) {
	Quicksort(data, 0, data.len());
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
