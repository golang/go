// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package arena provides lifetime-scoped allocation regions for loGo/s.
//
// Open returns a Region, and New, Make, and Append allocate into it. The
// compiler releases a region's memory at the end of the block that owns its
// Open call. With GOEXPERIMENT unset, every function in this package behaves
// exactly as its plain heap equivalent and placement never changes program
// meaning.
package arena

import "unsafe"

// Region is a handle to an allocation region.
//
// The zero Region is the heap: every function treats a zero handle as an
// ordinary heap allocation, so a Region that outlives its owner block keeps
// heap semantics.
type Region struct {
	h   unsafe.Pointer
	gen uintptr
}

// Open opens a new region and returns its handle. Close happens at the end
// of the block that owns the Open call; the program never calls Close.
func Open() Region { return Region{} }

// New allocates a zeroed T and returns its address.
func New[T any](r Region) *T { return new(T) }

// Make allocates a slice of n zeroed T elements. It panics like make for a
// negative n.
func Make[T any](r Region, n int) []T { return make([]T, n) }

// Append appends the values v to s, growing s as append does.
func Append[T any](r Region, s []T, v ...T) []T { return append(s, v...) }
