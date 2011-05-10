// $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1787.

package main

import "unsafe"

const x = unsafe.Sizeof([8]byte{})

func main() {
	var b [x]int
	_ = b
}

/*
bug338.go:14: array bound must be non-negative
*/
