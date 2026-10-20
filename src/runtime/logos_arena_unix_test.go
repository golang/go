// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena && unix

package runtime_test

import (
	"runtime"
	"syscall"
	"testing"
	"unsafe"
)

// M18-S1 (#59): the cgocheck2 write checks accept arena addresses that reach
// non-Go memory. The checks throw rather than panic, so a failure kills the
// test binary.
func TestLogosArenaCgoWriteChecksToNonGoMemory(t *testing.T) {
	type holder struct{ p *int }
	buf, err := syscall.Mmap(-1, 0, 4096, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap: %v", err)
	}
	defer syscall.Munmap(buf)
	nonGo := unsafe.Pointer(&buf[0])
	if runtime.LogosArenaIsGoPointer(nonGo) {
		t.Fatal("mmap memory is a Go pointer for the cgo checks")
	}
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(m)
		leaf := (*int)(runtime.LogosArenaAlloc(0, 8))
		h := (*holder)(runtime.LogosArenaAllocScan((*int)(nil), unsafe.Sizeof(holder{})))
		h.p = leaf
		runtime.LogosArenaCgoCheckPtrWrite((*unsafe.Pointer)(nonGo), unsafe.Pointer(leaf))
		runtime.LogosArenaCgoCheckMemmove(holder{}, nonGo, unsafe.Pointer(h))
	})
}
