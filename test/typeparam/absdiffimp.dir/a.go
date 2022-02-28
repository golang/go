// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~complex64 | ~complex128
}

// numericAbs matches numeric types with an Abs method.
type numericAbs[T any] interface {
	Numeric
	Abs() T
}

// AbsDifference computes the absolute value of the difference of
// a and b, where the absolute value is determined by the Abs method.
func absDifference[T numericAbs[T]](a, b T) T {
	d := a - b
	return d.Abs()
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

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// // orderedAbs is a helper type that defines an Abs method for
// // ordered numeric types.
// type orderedAbs[T orderedNumeric] T
//
// func (a orderedAbs[T]) Abs() orderedAbs[T] {
// 	if a < 0 {
// 		return -a
// 	}
// 	return a
// }
//
// // complexAbs is a helper type that defines an Abs method for
// // complex types.
// type complexAbs[T Complex] T
//
// func (a complexAbs[T]) Abs() complexAbs[T] {
// 	r := float64(real(a))
// 	i := float64(imag(a))
// 	d := math.Sqrt(r*r + i*i)
// 	return complexAbs[T](complex(d, 0))
// }
//
// // OrderedAbsDifference returns the absolute value of the difference
// // between a and b, where a and b are of an ordered type.
// func OrderedAbsDifference[T orderedNumeric](a, b T) T {
// 	return T(absDifference(orderedAbs[T](a), orderedAbs[T](b)))
// }
//
// // ComplexAbsDifference returns the absolute value of the difference
// // between a and b, where a and b are of a complex type.
// func ComplexAbsDifference[T Complex](a, b T) T {
// 	return T(absDifference(complexAbs[T](a), complexAbs[T](b)))
// }
