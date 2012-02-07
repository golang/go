// true

// WAS: errchk cgo $D/$F.go
// but this fails (cgo succeeds) on OS X Snow Leopard
// with Xcode 4.2 and gcc version 4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2336.1.00).

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1800: cgo not reporting line numbers.

package main

// #include <stdio.h>
import "C"

func f() {
	C.printf(nil) // ERROR "go:15.*unexpected type"
}
