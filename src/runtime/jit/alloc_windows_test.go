// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && windows

package jit_test

import (
	"fmt"
	"syscall"
	"unsafe"
)

const (
	_MEM_COMMIT     = 0x1000
	_MEM_RESERVE    = 0x2000
	_MEM_RELEASE    = 0x8000
	_PAGE_READWRITE    = 0x04
	_PAGE_EXECUTE_READ = 0x20
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procVirtualAlloc = kernel32.NewProc("VirtualAlloc")
	procVirtualFree  = kernel32.NewProc("VirtualFree")
	procVirtualProtect = kernel32.NewProc("VirtualProtect")
)

// allocExecutable allocates a page of executable memory and copies code into it.
func allocExecutable(code []byte) (uintptr, int, error) {
	const pageSize = 4096
	// Allocate RW memory.
	addr, _, err := procVirtualAlloc.Call(0, uintptr(pageSize), _MEM_COMMIT|_MEM_RESERVE, _PAGE_READWRITE)
	if addr == 0 {
		return 0, 0, fmt.Errorf("VirtualAlloc: %w", err)
	}
	// Copy code into the allocated memory.
	dst := unsafe.Slice((*byte)(unsafe.Pointer(addr)), pageSize)
	copy(dst, code)
	// Change protection to RX (W^X).
	var oldProtect uint32
	ret, _, err := procVirtualProtect.Call(addr, uintptr(pageSize), _PAGE_EXECUTE_READ, uintptr(unsafe.Pointer(&oldProtect)))
	if ret == 0 {
		procVirtualFree.Call(addr, 0, _MEM_RELEASE)
		return 0, 0, fmt.Errorf("VirtualProtect: %w", err)
	}
	return addr, pageSize, nil
}

func freeExecutable(addr uintptr, size int) {
	procVirtualFree.Call(addr, 0, _MEM_RELEASE)
}
