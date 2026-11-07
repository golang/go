// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"math/rand"
	"syscall"
	"testing"
)

// The input ends at an inaccessible page, so any overread faults.
func TestCastagnoliPMULLNoOverread(t *testing.T) {
	if !hasPMULLCRC32() {
		t.Skip("PMULL and CRC32 not available")
	}
	ps := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 3*ps, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap: %v", err)
	}
	defer syscall.Munmap(mem)
	if err := syscall.Mprotect(mem[2*ps:], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	rand.Read(mem[:2*ps])
	slicingTable := slicingMakeTable(Castagnoli)
	end := 2 * ps
	for n := 0; n <= 2*ps; n++ {
		p := mem[end-n : end : end]
		if got, want := ^castagnoliUpdatePMULL(^uint32(7), p), slicingUpdate(7, slicingTable, p); got != want {
			t.Fatalf("length %d: got %#x, want %#x", n, got, want)
		}
	}
}
