// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

export type SortInterface interface {
	Len() int;
	Less(i, j int) bool;
	Swap(i, j int);
}

func min(a, b int) int {
	if a < b {
		return a;
	}
	return b;
}

// Insertion sort
func InsertionSort(data SortInterface, a, b int) {
	for i := a+1; i < b; i++ {
		for j := i; j > a && data.Less(j, j-1); j-- {
			data.Swap(j, j-1);
		}
	}
}

// Quicksort, following Bentley and McIlroy,
// ``Engineering a Sort Function,'' SP&E November 1993.

// Move the median of the three values data[a], data[b], data[c] into data[a].
func MedianOfThree(data SortInterface, a, b, c int) {
	m0 := b;
	m1 := a;
	m2 := c;

	// bubble sort on 3 elements
	if data.Less(m1, m0) { data.Swap(m1, m0); }
	if data.Less(m2, m1) { data.Swap(m2, m1); }
	if data.Less(m1, m0) { data.Swap(m1, m0); }
	// now data[m0] <= data[m1] <= data[m2]
}

func SwapRange(data SortInterface, a, b, n int) {
	for i := 0; i < n; i++ {
		data.Swap(a+i, b+i);
	}
}

func Pivot(data SortInterface, lo, hi int) (midlo, midhi int) {
	m := (lo+hi)/2;
	if hi - lo > 40 {
		// Tukey's ``Ninther,'' median of three medians of three.
		s := (hi - lo) / 8;
		MedianOfThree(data, lo, lo+s, lo+2*s);
		MedianOfThree(data, m, m-s, m+s);
		MedianOfThree(data, hi-1, hi-1-s, hi-1-2*s);
	}
	MedianOfThree(data, lo, m, hi-1);

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
	pivot := lo;
	a, b, c, d := lo+1, lo+1, hi, hi;
	for b < c {
		if data.Less(b, pivot) {	// data[b] < pivot
			b++;
			continue;
		}
		if !data.Less(pivot, b) {	// data[b] = pivot
			data.Swap(a, b);
			a++;
			b++;
			continue;
		}
		if data.Less(pivot, c-1) {	// data[c-1] > pivot
			c--;
			continue;
		}
		if !data.Less(c-1, pivot) {	// data[c-1] = pivot
			data.Swap(c-1, d-1);
			c--;
			d--;
			continue;
		}
		// data[b] > pivot; data[c-1] < pivot
		data.Swap(b, c-1);
		b++;
		c--;
	}

	n := min(b-a, a-lo);
	SwapRange(data, lo, b-n, n);

	n = min(hi-d, d-c);
	SwapRange(data, c, hi-n, n);

	return lo+b-a, hi-(d-c);
}

func Quicksort(data SortInterface, a, b int) {
	if b - a > 7 {
		mlo, mhi := Pivot(data, a, b);
		Quicksort(data, a, mlo);
		Quicksort(data, mhi, b);
	} else if b - a > 1 {
		InsertionSort(data, a, b);
	}
}

export func Sort(data SortInterface) {
	Quicksort(data, 0, data.Len());
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

export type IntArray struct {
	data *[]int;
}

func (p *IntArray) Len() int            { return len(p.data); }
func (p *IntArray) Less(i, j int) bool  { return p.data[i] < p.data[j]; }
func (p *IntArray) Swap(i, j int)       { p.data[i], p.data[j] = p.data[j], p.data[i]; }


export type FloatArray struct {
	data *[]float;
}

func (p *FloatArray) Len() int            { return len(p.data); }
func (p *FloatArray) Less(i, j int) bool  { return p.data[i] < p.data[j]; }
func (p *FloatArray) Swap(i, j int)       { p.data[i], p.data[j] = p.data[j], p.data[i]; }


export type StringArray struct {
	data *[]string;
}

func (p *StringArray) Len() int            { return len(p.data); }
func (p *StringArray) Less(i, j int) bool  { return p.data[i] < p.data[j]; }
func (p *StringArray) Swap(i, j int)       { p.data[i], p.data[j] = p.data[j], p.data[i]; }


// Convenience wrappers for common cases

export func SortInts(a *[]int)        { Sort(&IntArray{a}); }
export func SortFloats(a *[]float)    { Sort(&FloatArray{a}); }
export func SortStrings(a *[]string)  { Sort(&StringArray{a}); }


export func IntsAreSorted(a *[]int) bool       { return IsSorted(&IntArray{a}); }
export func FloatsAreSorted(a *[]float) bool   { return IsSorted(&FloatArray{a}); }
export func StringsAreSorted(a *[]string) bool { return IsSorted(&StringArray{a}); }
