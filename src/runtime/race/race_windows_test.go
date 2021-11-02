// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows && race

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
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	VirtualAlloc := kernel32.NewProc("VirtualAlloc")
	VirtualFree := kernel32.NewProc("VirtualFree")
	const (
		MEM_COMMIT     = 0x00001000
		MEM_RESERVE    = 0x00002000
		MEM_RELEASE    = 0x8000
		PAGE_READWRITE = 0x04
	)
	mem, _, err := syscall.Syscall6(VirtualAlloc.Addr(), 4, 0, 1<<20, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE, 0, 0)
	if err != 0 {
		t.Fatalf("VirtualAlloc failed: %v", err)
	}
	defer syscall.Syscall(VirtualFree.Addr(), 3, mem, 1<<20, MEM_RELEASE)
	a := (*uint64)(unsafe.Pointer(mem))
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
