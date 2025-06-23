// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build purego

// Package alias implements memory aliasing tests.
package alias

// This is the Google App Engine standard variant based on reflect
// because the unsafe package and cgo are disallowed.

import "reflect"

// AnyOverlap reports whether x and y share memory at any (not necessarily
// corresponding) index. The memory beyond the slice length is ignored.
func AnyOverlap(x, y []byte) bool {
	return len(x) > 0 && len(y) > 0 &&
		reflect.ValueOf(&x[0]).Pointer() <= reflect.ValueOf(&y[len(y)-1]).Pointer() &&
		reflect.ValueOf(&y[0]).Pointer() <= reflect.ValueOf(&x[len(x)-1]).Pointer()
}

// InexactOverlap reports whether x and y share memory at any non-corresponding
// index. The memory beyond the slice length is ignored. Note that x and y can
// have different lengths and still not have any inexact overlap.
//
// InexactOverlap can be used to implement the requirements of the crypto/cipher
// AEAD, Block, BlockMode and Stream interfaces.
func InexactOverlap(x, y []byte) bool {
	if len(x) == 0 || len(y) == 0 || &x[0] == &y[0] {
		return false
	}
	return AnyOverlap(x, y)
}
