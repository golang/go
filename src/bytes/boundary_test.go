// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
//go:build linux

package bytes_test

import (
	. "bytes"
	"syscall"
	"testing"
)

// This file tests the situation where byte operations are checking
// data very near to a page boundary. We want to make sure those
// operations do not read across the boundary and cause a page
// fault where they shouldn't.

// These tests run only on linux. The code being tested is
// not OS-specific, so it does not need to be tested on all
// operating systems.

// dangerousSlice returns a slice which is immediately
// preceded and followed by a faulting page.
func dangerousSlice(t *testing.T) []byte {
	pagesize := syscall.Getpagesize()
	b, err := syscall.Mmap(0, 0, 3*pagesize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANONYMOUS|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap failed %s", err)
	}
	err = syscall.Mprotect(b[:pagesize], syscall.PROT_NONE)
	if err != nil {
		t.Fatalf("mprotect low failed %s\n", err)
	}
	err = syscall.Mprotect(b[2*pagesize:], syscall.PROT_NONE)
	if err != nil {
		t.Fatalf("mprotect high failed %s\n", err)
	}
	return b[pagesize : 2*pagesize]
}

func TestEqualNearPageBoundary(t *testing.T) {
	t.Parallel()
	b := dangerousSlice(t)
	for i := range b {
		b[i] = 'A'
	}
	for i := 0; i <= len(b); i++ {
		Equal(b[:i], b[len(b)-i:])
		Equal(b[len(b)-i:], b[:i])
	}
}

func TestIndexByteNearPageBoundary(t *testing.T) {
	t.Parallel()
	b := dangerousSlice(t)
	for i := range b {
		idx := IndexByte(b[i:], 1)
		if idx != -1 {
			t.Fatalf("IndexByte(b[%d:])=%d, want -1\n", i, idx)
		}
	}
}

func TestIndexNearPageBoundary(t *testing.T) {
	t.Parallel()
	q := dangerousSlice(t)
	if len(q) > 64 {
		// Only worry about when we're near the end of a page.
		q = q[len(q)-64:]
	}
	b := dangerousSlice(t)
	if len(b) > 256 {
		// Only worry about when we're near the end of a page.
		b = b[len(b)-256:]
	}
	for j := 1; j < len(q); j++ {
		q[j-1] = 1 // difference is only found on the last byte
		for i := range b {
			idx := Index(b[i:], q[:j])
			if idx != -1 {
				t.Fatalf("Index(b[%d:], q[:%d])=%d, want -1\n", i, j, idx)
			}
		}
		q[j-1] = 0
	}

	// Test differing alignments and sizes of q which always end on a page boundary.
	q[len(q)-1] = 1 // difference is only found on the last byte
	for j := 0; j < len(q); j++ {
		for i := range b {
			idx := Index(b[i:], q[j:])
			if idx != -1 {
				t.Fatalf("Index(b[%d:], q[%d:])=%d, want -1\n", i, j, idx)
			}
		}
	}
	q[len(q)-1] = 0
}

func TestCountNearPageBoundary(t *testing.T) {
	t.Parallel()
	b := dangerousSlice(t)
	for i := range b {
		c := Count(b[i:], []byte{1})
		if c != 0 {
			t.Fatalf("Count(b[%d:], {1})=%d, want 0\n", i, c)
		}
		c = Count(b[:i], []byte{0})
		if c != i {
			t.Fatalf("Count(b[:%d], {0})=%d, want %d\n", i, c, i)
		}
	}
}
