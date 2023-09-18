// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

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

// rune and byte data types are added implicitly due to them being alias for
// int32 and uint8 respectively.
type Number interface {
	int | int8 | int16 | int32 | int64 |
		uint | uint8 | uint16 | uint32 | uint64 |
		float32 | float64
}

// Returns the highest number among multiple arguments passed in
//
// Special cases are:
//
//	If +Inf is among the arguments, return +Inf
//	If NaN is among the arguments, return NaN
//	If both +Inf and NaN are among the arguments, return +Inf
func MaxMany[T Number](numbers ...T) float64 {
	var maximum T
	var hasNaN bool
	for index, number := range numbers {
		numberFloat := float64(number)
		switch {
		case index == 0:
			maximum = number
		case IsInf(numberFloat, 1):
			return Inf(1)
		case IsNaN(numberFloat):
			hasNaN = true
		case number > maximum:
			maximum = number
		}
	}
	if hasNaN {
		return NaN()
	}
	return float64(maximum)
}

// Returns the lowest number among multiple arguments passed in
//
// Special cases are:
//
//	If -Inf is among the arguments, return -Inf
//	If NaN is among the arguments, return NaN
//	If both -Inf and NaN are among the arguments, return -Inf
func MinMany[T Number](numbers ...T) float64 {
	var minimum T
	var hasNaN bool
	for index, number := range numbers {
		numberFloat := float64(number)
		switch {
		case index == 0:
			minimum = number
		case IsInf(numberFloat, 1):
			return Inf(-1)
		case IsNaN(numberFloat):
			hasNaN = true
		case number < minimum:
			minimum = number
		}
	}
	if hasNaN {
		return NaN()
	}
	return float64(minimum)
}
