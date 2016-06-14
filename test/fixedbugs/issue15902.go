// run

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure we don't use 4-byte unaligned writes
// to zero memory on architectures that don't support them.

package main

type T struct {
	a byte
	b [10]byte
}

//go:noinline
func f(t *T) {
	// t will be aligned, so &t.b won't be.
	t.b = [10]byte{}
}

var t T

func main() {
	f(&t)
}
