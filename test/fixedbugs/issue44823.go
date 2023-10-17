// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 44823: miscompilation with store combining.

package main

import "encoding/binary"

//go:noinline
func Id(a [8]byte) (x [8]byte) {
	binary.LittleEndian.PutUint64(x[:], binary.LittleEndian.Uint64(a[:]))
	return
}

var a = [8]byte{1, 2, 3, 4, 5, 6, 7, 8}

func main() {
	x := Id(a)
	if x != a {
		panic("FAIL")
	}
}
