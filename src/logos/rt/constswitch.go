// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"reflect"
	"unsafe"
)

// Reinterpret returns the value at p viewed as an S. A `const switch` exact
// case rebinds each variable whose type names the switched type to the case
// type through it. The checker admits the rebinding only when the two types
// are identical after substitution, so the layouts agree (SPEC Sec 2.5).
func Reinterpret[S, T any](p *T) S {
	return *(*S)(unsafe.Pointer(p))
}

// TildeIs reports whether the underlying type of T has the same kind as the
// basic type B. It is the run-time test of a `~B` case of a `const switch`
// (SPEC Sec 2.5).
func TildeIs[T, B any]() bool {
	return reflect.TypeFor[T]().Kind() == reflect.TypeFor[B]().Kind()
}

// Implements reports whether T implements the interface I. It is the
// run-time test of an interface case of a `const switch` (SPEC Sec 2.5).
func Implements[T, I any]() bool {
	return reflect.TypeFor[T]().Implements(reflect.TypeFor[I]())
}
