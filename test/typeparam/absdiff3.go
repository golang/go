// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// absdiff example using a function argument rather than attaching an
// Abs method to a structure containing base types.

package main

import (
	"fmt"
	"math"
)

type Numeric interface {
	OrderedNumeric | Complex
}

// absDifference computes the absolute value of the difference of
// a and b, where the absolute value is determined by the abs function.
func absDifference[T Numeric](a, b T, abs func(a T) T) T {
	return abs(a - b)
}

// OrderedNumeric matches numeric types that support the < operator.
type OrderedNumeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

func Abs[T OrderedNumeric](a T) T {
	if a < 0 {
		return -a
	}
	return a
}

// Complex matches the two complex types, which do not have a < operator.
type Complex interface {
	~complex64 | ~complex128
}

func realimag(x any) (re, im float64) {
	switch z := x.(type) {
	case complex64:
		re = float64(real(z))
		im = float64(imag(z))
	case complex128:
		re = real(z)
		im = imag(z)
	default:
		panic("unknown complex type")
	}
	return
}

func ComplexAbs[T Complex](a T) T {
	// TODO use direct conversion instead of realimag once #50937 is fixed
	r, i := realimag(a)
	// r := float64(real(a))
	// i := float64(imag(a))
	d := math.Sqrt(r*r + i*i)
	return T(complex(d, 0))
}

// OrderedAbsDifference returns the absolute value of the difference
// between a and b, where a and b are of an ordered type.
func OrderedAbsDifference[T OrderedNumeric](a, b T) T {
	return absDifference(a, b, Abs[T])
}

// ComplexAbsDifference returns the absolute value of the difference
// between a and b, where a and b are of a complex type.
func ComplexAbsDifference[T Complex](a, b T) T {
	return absDifference(a, b, ComplexAbs[T])
}

func main() {
	if got, want := OrderedAbsDifference(1.0, -2.0), 3.0; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}
	if got, want := OrderedAbsDifference(-1.0, 2.0), 3.0; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}
	if got, want := OrderedAbsDifference(-20, 15), 35; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}

	if got, want := ComplexAbsDifference(5.0+2.0i, 2.0-2.0i), 5+0i; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}
	if got, want := ComplexAbsDifference(2.0-2.0i, 5.0+2.0i), 5+0i; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}
}
