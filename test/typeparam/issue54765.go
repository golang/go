// errorcheck

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that not-in-heap types cannot be used as type
// arguments. (pointer-to-nih types are okay though.)

//go:build cgo
// +build cgo

package p

import (
	"runtime/cgo"
	"sync/atomic"
)

var _ atomic.Pointer[cgo.Incomplete]  // ERROR "cannot use incomplete \(or unallocatable\) type as a type argument: runtime/cgo\.Incomplete"
var _ atomic.Pointer[*cgo.Incomplete] // ok

func implicit(ptr *cgo.Incomplete) {
	g(ptr)  // ERROR "cannot use incomplete \(or unallocatable\) type as a type argument: runtime/cgo\.Incomplete"
	g(&ptr) // ok
}

func g[T any](_ *T) {}
