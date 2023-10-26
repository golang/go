// run

//go:build linux || darwin

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
)

func memcopy(dst, src []byte) (n int, err error) {
	defer func() {
		if r, ok := recover().(error); ok {
			err = r
		}
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

	// Create a hole in the mapping that's PROT_NONE.
	// Note that we can't use munmap here because the Go runtime
	// could create a mapping that ends up in this hole otherwise,
	// invalidating the test.
	hole := data[len(data)/2 : 3*(len(data)/4)]
	if err := syscall.Mprotect(hole, syscall.PROT_NONE); err != nil {
		log.Fatalf("mprotect: %v", err)
	}

	// Check that memcopy returns the actual amount copied
	// before the fault.
	const offset = 5
	n, err := memcopy(data[offset:], make([]byte, len(data)))
	if err == nil {
		log.Fatal("no error from memcopy across memory hole")
	}
	if expect := len(data)/2 - offset; n != expect {
		log.Fatalf("memcopy returned %d, want %d", n, expect)
	}
}
