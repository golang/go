// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// absdiff example in which an Abs method is attached to a generic type, which is a
// structure with a single field that may be a list of possible basic types.

package main

import (
	"fmt"
	"math"
)

type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~complex64 | ~complex128
}

// numericAbs matches a struct containing a numeric type that has an Abs method.
type numericAbs[T Numeric] interface {
	~struct{ Value_ T }
	Abs() T
	Value() T
}

// absDifference computes the absolute value of the difference of
// a and b, where the absolute value is determined by the Abs method.
func absDifference[T Numeric, U numericAbs[T]](a, b U) T {
	d := a.Value() - b.Value()
	dt := U{Value_: d}
	return dt.Abs()
}

// orderedNumeric matches numeric types that support the < operator.
type orderedNumeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

// Complex matches the two complex types, which do not have a < operator.
type Complex interface {
	~complex64 | ~complex128
}

// orderedAbs is a helper type that defines an Abs method for
// a struct containing an ordered numeric type.
type orderedAbs[T orderedNumeric] struct {
	Value_ T
}

func (a orderedAbs[T]) Abs() T {
	if a.Value_ < 0 {
		return -a.Value_
	}
	return a.Value_
}

// Field accesses through type parameters are disabled
// until we have a more thorough understanding of the
// implications on the spec. See issue #51576.
// Use accessor method instead.

func (a orderedAbs[T]) Value() T {
	return a.Value_
}

// complexAbs is a helper type that defines an Abs method for
// a struct containing a complex type.
type complexAbs[T Complex] struct {
	Value_ T
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

func (a complexAbs[T]) Abs() T {
	// TODO use direct conversion instead of realimag once #50937 is fixed
	r, i := realimag(a.Value_)
	// r := float64(real(a.Value))
	// i := float64(imag(a.Value))
	d := math.Sqrt(r*r + i*i)
	return T(complex(d, 0))
}

func (a complexAbs[T]) Value() T {
	return a.Value_
}

// OrderedAbsDifference returns the absolute value of the difference
// between a and b, where a and b are of an ordered type.
func OrderedAbsDifference[T orderedNumeric](a, b T) T {
	return absDifference(orderedAbs[T]{a}, orderedAbs[T]{b})
}

// ComplexAbsDifference returns the absolute value of the difference
// between a and b, where a and b are of a complex type.
func ComplexAbsDifference[T Complex](a, b T) T {
	return absDifference(complexAbs[T]{a}, complexAbs[T]{b})
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
