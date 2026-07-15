// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || darwin

package cryptotest

import (
	"syscall"
	"testing"
)

// BoundarySlices allocates a pair of slices of the given size one at the start
// of a page, another at the end. Any access beyond or before the slice
// boundaries should cause a fault.
func BoundarySlices(t *testing.T, size int) (start, end []byte) {
	pageSize := syscall.Getpagesize()
	needPages := 2 + (2*size+pageSize-1)/pageSize
	b, err := syscall.Mmap(0, 0, needPages*pageSize, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap failed: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(b) })
	if err := syscall.Mprotect(b[:pageSize], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect low failed: %v", err)
	}
	if err := syscall.Mprotect(b[len(b)-pageSize:], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect high failed: %v", err)
	}
	return b[pageSize : pageSize+size : pageSize+size],
		b[len(b)-pageSize-size : len(b)-pageSize : len(b)-pageSize]
}
