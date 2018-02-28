// +build linux darwin
// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that if a slice access causes a fault, a deferred func
// sees the most recent value of the variables it accesses.
// This is true today; the role of the test is to ensure it stays true.
//
// In the test, memcopy is the function that will fault, during dst[i] = src[i].
// The deferred func recovers from the error and returns, making memcopy
// return the current value of n. If n is not being flushed to memory
// after each modification, the result will be a stale value of n.
//
// The test is set up by mmapping a 64 kB block of memory and then
// unmapping a 16 kB hole in the middle of it. Running memcopy
// on the resulting slice will fault when it reaches the hole.

package main

import (
	"log"
	"runtime/debug"
	"syscall"
	"unsafe"
)

func memcopy(dst, src []byte) (n int, err error) {
	defer func() {
		err = recover().(error)
	}()

	for i := 0; i < len(dst) && i < len(src); i++ {
		dst[i] = src[i]
		n++
	}
	return
}

func main() {
	// Turn the eventual fault into a panic, not a program crash,
	// so that memcopy can recover.
	debug.SetPanicOnFault(true)

	size := syscall.Getpagesize()

	// Map 16 pages of data with a 4-page hole in the middle.
	data, err := syscall.Mmap(-1, 0, 16*size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		log.Fatalf("mmap: %v", err)
	}

	other := make([]byte, 16*size)

	// Note: Cannot call syscall.Munmap, because Munmap checks
	// that you are unmapping a whole region returned by Mmap.
	// We are trying to unmap just a hole in the middle.
	if _, _, err := syscall.Syscall(syscall.SYS_MUNMAP, uintptr(unsafe.Pointer(&data[8*size])), uintptr(4*size), 0); err != 0 {
		log.Fatalf("munmap: %v", err)
	}

	// Check that memcopy returns the actual amount copied
	// before the fault (8*size - 5, the offset we skip in the argument).
	n, err := memcopy(data[5:], other)
	if err == nil {
		log.Fatal("no error from memcopy across memory hole")
	}
	if n != 8*size-5 {
		log.Fatalf("memcopy returned %d, want %d", n, 8*size-5)
	}
}
