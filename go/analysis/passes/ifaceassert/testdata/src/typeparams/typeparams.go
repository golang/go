// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import "io"

type SourceReader[Source any] interface {
	Read(p Source) (n int, err error)
}

func GenericInterfaceAssertionTest[T io.Reader]() {
	var (
		a SourceReader[[]byte]
		b SourceReader[[]int]
		r io.Reader
	)
	_ = a.(io.Reader)
	_ = b.(io.Reader) // want `^impossible type assertion: no type can implement both typeparams.SourceReader\[\[\]int\] and io.Reader \(conflicting types for Read method\)$`

	_ = r.(SourceReader[[]byte])
	_ = r.(SourceReader[[]int]) // want `^impossible type assertion: no type can implement both io.Reader and typeparams.SourceReader\[\[\]int\] \(conflicting types for Read method\)$`
	_ = r.(T)                   // not actually an iface assertion, so checked by the type checker.

	switch a.(type) {
	case io.Reader:
	default:
	}

	switch b.(type) {
	case io.Reader: // want `^impossible type assertion: no type can implement both typeparams.SourceReader\[\[\]int\] and io.Reader \(conflicting types for Read method\)$`

	default:
	}
}

// Issue 50658: Check for type parameters in type switches.
type Float interface {
	float32 | float64
}

type Doer[F Float] interface {
	Do() F
}

func Underlying[F Float](v Doer[F]) string {
	switch v.(type) {
	case Doer[float32]:
		return "float32!"
	case Doer[float64]:
		return "float64!"
	default:
		return "<unknown>"
	}
}

func DoIf[F Float]() {
	// This is a synthetic function to create a non-generic to generic assignment.
	// This function does not make much sense.
	var v Doer[float32]
	if t, ok := v.(Doer[F]); ok {
		t.Do()
	}
}

func IsASwitch[F Float, U Float](v Doer[F]) bool {
	switch v.(type) {
	case Doer[U]:
		return true
	}
	return false
}

func IsA[F Float, U Float](v Doer[F]) bool {
	_, is := v.(Doer[U])
	return is
}

func LayeredTypes[F Float]() {
	// This is a synthetic function cover more isParameterized cases.
	type T interface {
		foo() struct{ _ map[T][2]chan *F }
	}
	type V interface {
		foo() struct{ _ map[T][2]chan *float32 }
	}
	var t T
	var v V
	t, _ = v.(T)
	_ = t
}

type X[T any] struct{}

func (x X[T]) m(T) {}

func InstancesOfGenericMethods() {
	var x interface{ m(string) }
	// _ = x.(X[int])    // BAD. Not enabled as it does not type check.
	_ = x.(X[string]) // OK
}
