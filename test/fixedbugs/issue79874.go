// run

//go:build (linux || darwin) && !(386 || arm || mips || mipsle)

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/binary"
	"fmt"
	"syscall"
	"unsafe"
)

const l = 1 << 34

//go:noinline
func bug(s []byte) []byte {
	if len(s) < l+8 {
		panic("too short")
	}
	return s[min(l, len(s)):]
}

func main() {
	// This code is a bit tricky because I have two contradictory constraints:
	// 1. I need a slice >4GB big, ideally more (to test with non byte size element).
	// 2. I can't allocate 4GB of ram in a test, let alone the 16GB I ended up using for real.
	pageSize := syscall.Getpagesize()

	// Allocate a bunch of zeros, because this MAP_ANON mapping lack the PROT_WRITE permission
	// the kernel will use a single shared aliased zero page to back up this memory.
	// We still pay on the order of 32MB for the page table entries but it's acceptable.
	s, err := syscall.Mmap(-1, 0, l+pageSize, syscall.PROT_READ, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		panic(err)
	}

	// Make the tail page writable. Use unsafe.Slice rather than s[l:] because s[l:] goes through the slicemask path under test.
	if err := syscall.Mprotect(unsafe.Slice(&s[l], pageSize), syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
		panic(err)
	}

	// Write without using s[l:] otherwise the same bug happens here and in bug making the test pass even if the bug is present.
	const sentinel uint64 = 0x1122334455667788
	for i := 0; i < 8; i++ {
		s[l+i] = byte(sentinel >> (8 * i))
	}

	// Finally test the bug.
	if v := binary.LittleEndian.Uint64(bug(s)); v != sentinel {
		panic(fmt.Sprintf("got %x, want %x", v, sentinel))
	}
}
