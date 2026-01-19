// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64 || ppc64le || s390x) && !purego && linux

package nistec

import (
	"syscall"
	"testing"
	"unsafe"
)

// Lightly adapted from the bytes test package. Allocate a pair of T one at the start of a page, another at the
// end. Any access beyond or before the page boundary should cause a fault. This is linux specific.
func dangerousObjs[T any](t *testing.T) (start *T, end *T) {
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
	b = b[pagesize : 2*pagesize]
	end = (*T)(unsafe.Pointer(&b[len(b)-(int)(unsafe.Sizeof(*end))]))
	start = (*T)(unsafe.Pointer(&b[0]))
	return start, end
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
