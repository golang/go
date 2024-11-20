// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle_test

import (
	"crypto/subtle"
	"syscall"
	"testing"
)

// dangerousSlice returns a slice which is immediately
// preceded and followed by a faulting page.
// Copied from the bytes package tests.
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

func TestXORBytesBoundary(t *testing.T) {
	safe := make([]byte, syscall.Getpagesize()*2)
	spicy := dangerousSlice(t)
	for i := 1; i <= syscall.Getpagesize(); i++ {
		start := spicy[:i]
		end := spicy[len(spicy)-i:]
		subtle.XORBytes(end, safe, safe[:i])
		subtle.XORBytes(start, safe, safe[:i])
		subtle.XORBytes(safe, start, safe)
		subtle.XORBytes(safe, end, safe)
		subtle.XORBytes(safe, safe, start)
		subtle.XORBytes(safe, safe, end)
	}
}
