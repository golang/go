// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test _may_ fail on arm, but requires the host to
// trap unaligned loads. This is generally done with
//
// echo "4" > /proc/cpu/alignment

package main

type T struct {
	U uint16
	V T2
}

type T2 struct {
	pad    [4096]byte
	A, B byte
}

var s, t = new(T), new(T)

func main() {
	var u, v *T2 = &s.V, &t.V
	u.B = v.B
}
