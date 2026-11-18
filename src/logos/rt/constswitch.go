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
// Every legitimate compiler call proves S == T, so the safe pointer assertion
// below preserves those calls and panics deterministically otherwise,
// without an unsafe cast or checkptr fatal.
func Reinterpret[S, T any](p *T) S {
	return *any(p).(*S)
}

// PtrAs returns p viewed as a *S. A `const switch` case binds each variable
// whose type names the switched type to a pointer of the case type, and every
// use of the variable is rewritten to dereference the pointer, so reads and
// writes reach the caller's storage directly. The checker admits the binding
// only when the two types are identical after substitution, so the layouts
// agree (SPEC Sec 2.5); the assertion preserves legitimate calls and panics
// deterministically otherwise, without an unsafe cast or checkptr fatal.
func PtrAs[S, T any](p *T) *S {
	return any(p).(*S)
}

// UnderPtrAs returns p viewed as a *S. A `const switch` tilde case binds
// variables as for PtrAs, but the case's basic type keeps the layout and not
// the identity of the type argument (Port vs int), so an identity assertion
// cannot see through it; the tilde match guarantees the layouts agree, making
// the pointer cast sound (SPEC Sec 2.5). Sizes and alignment match exactly,
// so checkptr admits it.
func UnderPtrAs[S, T any](p *T) *S {
	return (*S)(unsafe.Pointer(p))
}

// Box returns v as an I. Inside a `const switch` interface case, a value of
// the switched type keeps its concrete layout and gains I's method set where
// the checker admitted it; the comma-ok assert returns the zero I rather than
// panicking when v carries a nil interface value (SPEC Sec 2.5).
func Box[I, T any](v T) I {
	r, _ := any(v).(I)
	return r
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
