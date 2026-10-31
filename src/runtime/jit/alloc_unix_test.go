// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && unix

package jit_test

import (
	"syscall"
	"unsafe"
)

// allocExecutable allocates a page of executable memory and copies code into it.
func allocExecutable(code []byte) (uintptr, int, error) {
	pageSize := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, pageSize,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS)
	if err != nil {
		return 0, 0, err
	}
	copy(mem, code)
	err = syscall.Mprotect(mem, syscall.PROT_READ|syscall.PROT_EXEC)
	if err != nil {
		syscall.Munmap(mem)
		return 0, 0, err
	}
	return uintptr(unsafe.Pointer(&mem[0])), pageSize, nil
}

func freeExecutable(addr uintptr, size int) {
	mem := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)
	syscall.Munmap(mem)
}
