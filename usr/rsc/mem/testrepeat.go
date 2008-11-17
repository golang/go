// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"allocator"
)

var footprint int64
func bigger() {
	if footprint < allocator.footprint {
		footprint = allocator.footprint;
		println("Footprint", footprint);
	}
}

func main() {
	for i := 0; i < 1<<16; i++ {
		for j := 1; j <= 1<<22; j<<=1 {
			if i == 0 {
				println("First alloc:", j);
			}
			b := allocator.malloc(j);
			allocator.free(b);
			bigger();
		}
		if i%(1<<10) == 0 {
			println(i);
		}
		if i == 0 {
			println("Primed", i);
			allocator.frozen = true;
		}
	}
}
