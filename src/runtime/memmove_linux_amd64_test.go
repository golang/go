// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// TestMemmoveOverflow maps 3GB of memory and calls memmove on
// the corresponding slice.
func TestMemmoveOverflow(t *testing.T) {
	t.Parallel()
	// Create a temporary file.
	tmp, err := os.CreateTemp("", "go-memmovetest")
	if err != nil {
		t.Fatal(err)
	}
	_, err = tmp.Write(make([]byte, 65536))
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()

	// Set up mappings.
	base, _, errno := syscall.Syscall6(syscall.SYS_MMAP,
		0xa0<<32, 3<<30, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS, ^uintptr(0), 0)
	if errno != 0 {
		t.Skipf("could not create memory mapping: %s", errno)
	}
	syscall.Syscall(syscall.SYS_MUNMAP, base, 3<<30, 0)

	for off := uintptr(0); off < 3<<30; off += 65536 {
		_, _, errno := syscall.Syscall6(syscall.SYS_MMAP,
			base+off, 65536, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED|syscall.MAP_FIXED, tmp.Fd(), 0)
		if errno != 0 {
			t.Skipf("could not map a page at requested 0x%x: %s", base+off, errno)
		}
		defer syscall.Syscall(syscall.SYS_MUNMAP, base+off, 65536, 0)
	}

	s := unsafe.Slice((*byte)(unsafe.Pointer(base)), 3<<30)
	n := copy(s[1:], s)
	if n != 3<<30-1 {
		t.Fatalf("copied %d bytes, expected %d", n, 3<<30-1)
	}
	n = copy(s, s[1:])
	if n != 3<<30-1 {
		t.Fatalf("copied %d bytes, expected %d", n, 3<<30-1)
	}
}
