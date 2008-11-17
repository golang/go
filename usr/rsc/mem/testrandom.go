// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"allocator";
	"rand"
)

var footprint int64;
var allocated int64;
func bigger() {
	if footprint < allocator.footprint {
		footprint = allocator.footprint;
		println("Footprint", footprint, " for ", allocated);
		if footprint > 1e9 {
			panicln("too big");
		}
	}
}

// Prime the data structures by allocating one of
// each block in order.  After this, there should be
// little reason to ask for more memory from the OS.
func prime() {
	for i := 0; i < 16; i++ {
		b := allocator.malloc(1<<uint(i));
		allocator.free(b);
	}
	for i := 0; i < 256; i++ {
		b := allocator.malloc(i<<12);
		allocator.free(b);
	}
}

func main() {
//	prime();
	var blocks [1] struct { base *byte; siz int; };
	for i := 0; i < 1 << 20; i++ {
		if i%(1<<10) == 0 {
			println(i);
		}
		b := rand.rand() % len(blocks);
		if blocks[b].base != nil {
		//	println("Free", blocks[b].siz, blocks[b].base);
			allocator.free(blocks[b].base);
			blocks[b].base = nil;
			allocated -= int64(blocks[b].siz);
			continue
		}
		siz := rand.rand() >> (11 + rand.urand32() % 20);
		base := allocator.malloc(siz);
		blocks[b].base = base;
		blocks[b].siz = siz;
		allocated += int64(siz);
	//	println("Alloc", siz, base);
		allocator.memset(base, 0xbb, siz);
		bigger();
	}
}
