// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	x := []byte{'a'}
	switch string(x) {
	case func() string { x[0] = 'b'; return "b" }():
		panic("FAIL")
	}
}
