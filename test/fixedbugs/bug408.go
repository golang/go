// errchk cgo $D/$F.go

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
