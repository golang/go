// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Ordered Size comparisons can be performed (> , < , <= , >=)
type Ordered interface {
	Integer | Float | ~string
}

type Integer interface {
	Signed | Unsigned
}

type Signed interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type Unsigned interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

// Float we do not recommend adding float64
type Float interface {
	~float32
	//~float32 | ~float64
}

// Dim returns the maximum of x-y or 0.
//
// Special cases are:
//
//	Dim(+Inf, +Inf) = NaN
//	Dim(-Inf, -Inf) = NaN
//	Dim(x, NaN) = Dim(NaN, x) = NaN
func Dim(x, y float64) float64 {
	// The special cases result in NaN after the subtraction:
	//      +Inf - +Inf = NaN
	//      -Inf - -Inf = NaN
	//       NaN - y    = NaN
	//         x - NaN  = NaN
	v := x - y
	if v <= 0 {
		// v is negative or 0
		return 0
	}
	// v is positive or NaN
	return v
}

// Max returns the larger of x or y.
//
// Special cases are:
//
//	Max(x, +Inf) = Max(+Inf, x) = +Inf
//	Max(x, NaN) = Max(NaN, x) = NaN
//	Max(+0, ±0) = Max(±0, +0) = +0
//	Max(-0, -0) = -0
//
// Note that this differs from the built-in function max when called
// with NaN and +Inf.
func Max(x, y float64) float64 {
	if haveArchMax {
		return archMax(x, y)
	}
	return max(x, y)
}

// MaxAll returns the larger of x or y.(All Ordered types except float64)
// If it is of type float64, Max should be used instead of MaxAll
// If add float64 type to MaxAll, it's inevitably need reflection
//  to determine the type of the value and then do something different
// So we do not recommend adding float64
func MaxAll[T Ordered](x, y T) T {
	if x > y {
		return x
	}
	return y
}

func max(x, y float64) float64 {
	// special cases
	switch {
	case IsInf(x, 1) || IsInf(y, 1):
		return Inf(1)
	case IsNaN(x) || IsNaN(y):
		return NaN()
	case x == 0 && x == y:
		if Signbit(x) {
			return y
		}
		return x
	}
	if x > y {
		return x
	}
	return y
}

// Min returns the smaller of x or y.
//
// Special cases are:
//
//	Min(x, -Inf) = Min(-Inf, x) = -Inf
//	Min(x, NaN) = Min(NaN, x) = NaN
//	Min(-0, ±0) = Min(±0, -0) = -0
//
// Note that this differs from the built-in function min when called
// with NaN and -Inf.
func Min(x, y float64) float64 {
	if haveArchMin {
		return archMin(x, y)
	}
	return min(x, y)
}

// MinAll returns the smaller of x or y.(All Ordered types except float64)
// If it is of type float64, Min should be used instead of MinAll
// If add float64 type to MinAll, it's inevitably need reflection
//  to determine the type of the value and then do something different
// So we do not recommend adding float64
func MinAll[T Ordered](x, y T) T {
	if x < y {
		return x
	}
	return y
}

func min(x, y float64) float64 {
	// special cases
	switch {
	case IsInf(x, -1) || IsInf(y, -1):
		return Inf(-1)
	case IsNaN(x) || IsNaN(y):
		return NaN()
	case x == 0 && x == y:
		if Signbit(x) {
			return x
		}
		return y
	}
	if x < y {
		return x
	}
	return y
}
