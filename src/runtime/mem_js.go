// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysAlloc(n uintptr, sysStat *uint64) unsafe.Pointer {
	p := sysReserve(nil, n)
	sysMap(p, n, sysStat)
	return p
}

func sysUnused(v unsafe.Pointer, n uintptr) {
}

func sysUsed(v unsafe.Pointer, n uintptr) {
}

// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//go:nosplit
func sysFree(v unsafe.Pointer, n uintptr, sysStat *uint64) {
	mSysStatDec(sysStat, n)
}

func sysFault(v unsafe.Pointer, n uintptr) {
}

var reserveEnd uintptr

func sysReserve(v unsafe.Pointer, n uintptr) unsafe.Pointer {
	// TODO(neelance): maybe unify with mem_plan9.go, depending on how https://github.com/WebAssembly/design/blob/master/FutureFeatures.md#finer-grained-control-over-memory turns out

	if reserveEnd < lastmoduledatap.end {
		reserveEnd = lastmoduledatap.end
	}
	if uintptr(v) < reserveEnd {
		v = unsafe.Pointer(reserveEnd)
	}
	reserveEnd = uintptr(v) + n

	current := currentMemory()
	needed := int32(reserveEnd/sys.DefaultPhysPageSize + 1)
	if current < needed {
		if growMemory(needed-current) == -1 {
			return nil
		}
	}

	return v
}

func currentMemory() int32
func growMemory(pages int32) int32

func sysMap(v unsafe.Pointer, n uintptr, sysStat *uint64) {
	mSysStatInc(sysStat, n)
}
