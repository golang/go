// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

// Max returns the maximum of two values of some ordered type.
func Max[T Ordered](a, b T) T {
	if a > b {
		return a
	}
	return b
}

// Min returns the minimum of two values of some ordered type.
func Min[T Ordered](a, b T) T {
	if a < b {
		return a
	}
	return b
}

// Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func Equal[Elem comparable](s1, s2 []Elem) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if v1 != v2 {
			isNaN := func(f Elem) bool { return f != f }
			if !isNaN(v1) || !isNaN(v2) {
				return false
			}
		}
	}
	return true
}

// EqualFn reports whether two slices are equal using a comparison
// function on each element.
func EqualFn[Elem any](s1, s2 []Elem, eq func(Elem, Elem) bool) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if !eq(v1, v2) {
			return false
		}
	}
	return true
}

// Map turns a []Elem1 to a []Elem2 using a mapping function.
func Map[Elem1, Elem2 any](s []Elem1, f func(Elem1) Elem2) []Elem2 {
	r := make([]Elem2, len(s))
	for i, v := range s {
		r[i] = f(v)
	}
	return r
}

// Reduce reduces a []Elem1 to a single value of type Elem2 using
// a reduction function.
func Reduce[Elem1, Elem2 any](s []Elem1, initializer Elem2, f func(Elem2, Elem1) Elem2) Elem2 {
	r := initializer
	for _, v := range s {
		r = f(r, v)
	}
	return r
}

// Filter filters values from a slice using a filter function.
func Filter[Elem any](s []Elem, f func(Elem) bool) []Elem {
	var r []Elem
	for _, v := range s {
		if f(v) {
			r = append(r, v)
		}
	}
	return r
}

// Max returns the maximum element in a slice of some ordered type.
// If the slice is empty it returns the zero value of the element type.
func SliceMax[Elem Ordered](s []Elem) Elem {
	if len(s) == 0 {
		var zero Elem
		return zero
	}
	return Reduce(s[1:], s[0], Max[Elem])
}

// Min returns the minimum element in a slice of some ordered type.
// If the slice is empty it returns the zero value of the element type.
func SliceMin[Elem Ordered](s []Elem) Elem {
	if len(s) == 0 {
		var zero Elem
		return zero
	}
	return Reduce(s[1:], s[0], Min[Elem])
}

// Append adds values to the end of a slice, returning a new slice.
// This is like the predeclared append function; it's an example
// of how to write it using generics. We used to write code like
// this before append was added to the language, but we had to write
// a separate copy for each type.
func Append[T any](s []T, t ...T) []T {
	lens := len(s)
	tot := lens + len(t)
	if tot <= cap(s) {
		s = s[:tot]
	} else {
		news := make([]T, tot, tot+tot/2)
		Copy(news, s)
		s = news
	}
	Copy(s[lens:tot], t)
	return s
}

// Copy copies values from t to s, stopping when either slice is full,
// returning the number of values copied. This is like the predeclared
// copy function; it's an example of how to write it using generics.
func Copy[T any](s, t []T) int {
	i := 0
	for ; i < len(s) && i < len(t); i++ {
		s[i] = t[i]
	}
	return i
}
