// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package a declares constraint interfaces and never uses them.
package a

type Signed interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type Unsigned interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

type Integer interface {
	Signed | Unsigned
}

type Ordered interface {
	Integer | ~float32 | ~float64 | ~string
}

type Number interface {
	int | float64
}

type Labeled interface {
	~int | ~string
	Label() string
}
