// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file shows some examples of generic constraint interfaces.

package p

type MyInt int

type (
	// Arbitrary types may be embedded like interfaces.
	_ interface{int}
	_ interface{~int}

	// Types may be combined into a union.
	union interface{int|~string}

	// Union terms must describe disjoint (non-overlapping) type sets.
	_ interface{int|int /* ERROR overlapping terms int */ }
	_ interface{int|~ /* ERROR overlapping terms ~int */ int }
	_ interface{~int|~ /* ERROR overlapping terms ~int */ int }
	_ interface{~int|MyInt /* ERROR overlapping terms p.MyInt and ~int */ }
	_ interface{int|any}
	_ interface{int|~string|union}
	_ interface{int|~string|interface{int}}
	_ interface{union|int}   // interfaces (here: union) are ignored when checking for overlap
	_ interface{union|union} // ditto

	// For now we do not permit interfaces with methods in unions.
	_ interface{~ /* ERROR invalid use of ~ */ any}
	_ interface{int|interface /* ERROR cannot use .* in union */ { m() }}
)

type (
	// Tilde is not permitted on defined types or interfaces.
	foo int
	bar any
	_ interface{foo}
	_ interface{~ /* ERROR invalid use of ~ */ foo }
	_ interface{~ /* ERROR invalid use of ~ */ bar }
)

// Stand-alone type parameters are not permitted as elements or terms in unions.
type (
	_[T interface{ *T } ] struct{}        // ok
	_[T interface{ int | *T } ] struct{}  // ok
	_[T interface{ T /* ERROR term cannot be a type parameter */ } ] struct{}
	_[T interface{ ~T /* ERROR type in term ~T cannot be a type parameter */ } ] struct{}
	_[T interface{ int|T /* ERROR term cannot be a type parameter */ }] struct{}
)

// Multiple embedded union elements are intersected. The order in which they
// appear in the interface doesn't matter since intersection is a symmetric
// operation.

type myInt1 int
type myInt2 int

func _[T interface{ myInt1|myInt2; ~int }]() T { return T(0) }
func _[T interface{ ~int; myInt1|myInt2 }]() T { return T(0) }

// Here the intersections are empty - there's no type that's in the type set of T.
func _[T interface{ myInt1|myInt2; int }]() T { return T(0 /* ERROR cannot convert */ ) }
func _[T interface{ int; myInt1|myInt2 }]() T { return T(0 /* ERROR cannot convert */ ) }

// Union elements may be interfaces as long as they don't define
// any methods or embed comparable.

type (
	Integer interface{ ~int|~int8|~int16|~int32|~int64 }
	Unsigned interface{ ~uint|~uint8|~uint16|~uint32|~uint64 }
	Floats interface{ ~float32|~float64 }
	Complex interface{ ~complex64|~complex128 }
	Number interface{ Integer|Unsigned|Floats|Complex }
	Ordered interface{ Integer|Unsigned|Floats|~string }

	_ interface{ Number | error /* ERROR cannot use error in union */ }
	_ interface{ Ordered | comparable /* ERROR cannot use comparable in union */ }
)
