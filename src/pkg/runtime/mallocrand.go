// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Random malloc test.

package main

import (
	"flag"
	"math/rand"
	"runtime"
	"unsafe"
)

var chatty = flag.Bool("v", false, "chatty")

var footprint uint64
var allocated uint64

func bigger() {
	memstats := new(runtime.MemStats)
	runtime.ReadMemStats(memstats)
	if f := memstats.Sys; footprint < f {
		footprint = f
		if *chatty {
			println("Footprint", footprint, " for ", allocated)
		}
		if footprint > 1e9 {
			println("too big")
			panic("fail")
		}
	}
}

// Prime the data structures by allocating one of
// each block in order.  After this, there should be
// little reason to ask for more memory from the OS.
func prime() {
	for i := 0; i < 16; i++ {
		b := runtime.Alloc(1 << uint(i))
		runtime.Free(b)
	}
	for i := uintptr(0); i < 256; i++ {
		b := runtime.Alloc(i << 12)
		runtime.Free(b)
	}
}

func memset(b *byte, c byte, n uintptr) {
	np := uintptr(n)
	for i := uintptr(0); i < np; i++ {
		*(*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(b)) + i)) = c
	}
}

func main() {
	flag.Parse()
	//	prime()
	var blocks [1]struct {
		base *byte
		siz  uintptr
	}
	for i := 0; i < 1<<10; i++ {
		if i%(1<<10) == 0 && *chatty {
			println(i)
		}
		b := rand.Int() % len(blocks)
		if blocks[b].base != nil {
			//	println("Free", blocks[b].siz, blocks[b].base)
			runtime.Free(blocks[b].base)
			blocks[b].base = nil
			allocated -= uint64(blocks[b].siz)
			continue
		}
		siz := uintptr(rand.Int() >> (11 + rand.Uint32()%20))
		base := runtime.Alloc(siz)
		//	ptr := uintptr(syscall.BytePtr(base))+uintptr(siz/2)
		//	obj, size, ref, ok := allocator.find(ptr)
		//	if obj != base || *ref != 0 || !ok {
		//		println("find", siz, obj, ref, ok)
		//		panic("fail")
		//	}
		blocks[b].base = base
		blocks[b].siz = siz
		allocated += uint64(siz)
		//	println("Alloc", siz, base)
		memset(base, 0xbb, siz)
		bigger()
	}
}
