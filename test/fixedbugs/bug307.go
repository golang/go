// $G $D/$F.go

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Valid program, gccgo reported an error.
// bug307.go:14:6: error: complex arguments must have identical types

package main

func main() {
	var f float64
	_ = complex(1/f, 0)
}
