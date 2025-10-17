// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package scan_test

import (
	"internal/runtime/gc"
	"syscall"
	"testing"
	"unsafe"
)

func makeMem(t testing.TB, nPages int) ([]uintptr, func()) {
	mem, err := syscall.Mmap(-1, 0, int(gc.PageSize*nPages), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatalf("mmap failed: %s", err)
	}
	free := func() {
		syscall.Munmap(mem)
	}
	return unsafe.Slice((*uintptr)(unsafe.Pointer(unsafe.SliceData(mem))), len(mem)/8), free
}
