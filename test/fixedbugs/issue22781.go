// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime/debug"

type T struct {
	// >= 16 bytes to avoid tiny alloc.
	a, b int
}

func main() {
	debug.SetGCPercent(1)
	for i := 0; i < 100000; i++ {
		m := make(map[*T]struct{}, 0)
		for j := 0; j < 20; j++ {
			// During the call to mapassign_fast64, the key argument
			// was incorrectly treated as a uint64. If the stack was
			// scanned during that call, the only pointer to k was
			// missed, leading to *k being collected prematurely.
			k := new(T)
			m[k] = struct{}{}
		}
	}
}
