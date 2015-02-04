// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	s := "foo"
	b := []byte(s)
	m := make(map[string]int)
	// Test that map index can be used in range
	// and that slicebytetostringtmp is not used in this context.
	for m[string(b)] = range s {
	}
	b[0] = 'b'
	if m["foo"] != 2 {
		panic("bad")
	}
}
