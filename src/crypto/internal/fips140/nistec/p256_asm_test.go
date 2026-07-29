// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64 || ppc64le || s390x) && !purego && (linux || darwin)

package nistec

import (
	"syscall"
	"testing"
	"unsafe"
)

// dangerousObjs allocates a pair of T one at the start of a page, another at
// the end. Any access beyond or before the page boundary should cause a fault.
func dangerousObjs[T any](t *testing.T) (start *T, end *T) {
	startBuf, endBuf := boundarySlices(t, int(unsafe.Sizeof(*new(T))))
	return (*T)(unsafe.Pointer(&startBuf[0])), (*T)(unsafe.Pointer(&endBuf[0]))
}

// boundarySlices is a copy of [crypto/internal/cryptotest.BoundarySlices].
func boundarySlices(t *testing.T, size int) (start, end []byte) {
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

func TestP256SelectAffinePageBoundary(t *testing.T) {
	var out p256AffinePoint
	begintp, endtp := dangerousObjs[p256AffineTable](t)
	for i := 0; i < 31; i++ {
		p256SelectAffine(&out, begintp, i)
		p256SelectAffine(&out, endtp, i)
	}
}

func TestP256SelectPageBoundary(t *testing.T) {
	var out P256Point
	begintp, endtp := dangerousObjs[p256Table](t)
	for i := 0; i < 15; i++ {
		p256Select(&out, begintp, i)
		p256Select(&out, endtp, i)
	}
}
