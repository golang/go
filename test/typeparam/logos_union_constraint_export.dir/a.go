// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package a declares constraint interfaces and never uses them: no type
// parameter list of this package names any of them, so every use is in
// the importing package.
package a

type Signed interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type Unsigned interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

// Integer is a union of unions.
type Integer interface {
	Signed | Unsigned
}

type Float interface {
	~float32 | ~float64
}

// Ordered is a union that nests Integer, which nests Signed and
// Unsigned.
type Ordered interface {
	Integer | Float | ~string
}

// Number is a closed term list without ~ terms.
type Number interface {
	int | float64
}

// Labeled mixes a method with a term list.
type Labeled interface {
	~int | ~string
	Label() string
}

// Wrapped embeds a declared constraint and adds a method.
type Wrapped interface {
	Ordered
	Wrap() string
}
