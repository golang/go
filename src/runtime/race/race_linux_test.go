// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && race

package race_test

import (
	"sync/atomic"
	"syscall"
	"testing"
	"unsafe"
)

func TestAtomicMmap(t *testing.T) {
	// Test that atomic operations work on "external" memory. Previously they crashed (#16206).
	// Also do a sanity correctness check: under race detector atomic operations
	// are implemented inside of race runtime.
	mem, err := syscall.Mmap(-1, 0, 1<<20, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap failed: %v", err)
	}
	defer syscall.Munmap(mem)
	a := (*uint64)(unsafe.Pointer(&mem[0]))
	if *a != 0 {
		t.Fatalf("bad atomic value: %v, want 0", *a)
	}
	atomic.AddUint64(a, 1)
	if *a != 1 {
		t.Fatalf("bad atomic value: %v, want 1", *a)
	}
	atomic.AddUint64(a, 1)
	if *a != 2 {
		t.Fatalf("bad atomic value: %v, want 2", *a)
	}
}
