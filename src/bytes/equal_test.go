// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// +build linux

package bytes_test

import (
	. "bytes"
	"syscall"
	"testing"
	"unsafe"
)

// This file tests the situation where memeq is checking
// data very near to a page boundary. We want to make sure
// equal does not read across the boundary and cause a page
// fault where it shouldn't.

// This test runs only on linux. The code being tested is
// not OS-specific, so it does not need to be tested on all
// operating systems.

func TestEqualNearPageBoundary(t *testing.T) {
	pagesize := syscall.Getpagesize()
	b := make([]byte, 4*pagesize)
	i := pagesize
	for ; uintptr(unsafe.Pointer(&b[i]))%uintptr(pagesize) != 0; i++ {
	}
	syscall.Mprotect(b[i-pagesize:i], 0)
	syscall.Mprotect(b[i+pagesize:i+2*pagesize], 0)
	defer syscall.Mprotect(b[i-pagesize:i], syscall.PROT_READ|syscall.PROT_WRITE)
	defer syscall.Mprotect(b[i+pagesize:i+2*pagesize], syscall.PROT_READ|syscall.PROT_WRITE)

	// both of these should fault
	//pagesize += int(b[i-1])
	//pagesize += int(b[i+pagesize])

	for j := 0; j < pagesize; j++ {
		b[i+j] = 'A'
	}
	for j := 0; j <= pagesize; j++ {
		Equal(b[i:i+j], b[i+pagesize-j:i+pagesize])
		Equal(b[i+pagesize-j:i+pagesize], b[i:i+j])
	}
}
