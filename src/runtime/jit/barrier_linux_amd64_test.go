// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && linux && amd64

package jit

import (
	"encoding/binary"
	"runtime"
	"syscall"
	"testing"
	"unsafe"
)

func TestABIBarrierBranches(t *testing.T) {
	// Exercise the actual emitted conditional branch with both states. Redirect
	// its address literal to a private flag; never modify runtime.writeBarrier.
	flag := new(uint64)
	b := &SliceBuffer{Bytes: make([]byte, 128)}
	f := EmitWriteBarrierCheck(b, 0)
	b.Write([]byte{0xb8, 42, 0, 0, 0, 0xc3})
	PatchWriteBarrierCheck(b, f, b.Offset())
	b.Write([]byte{0xb8, 17, 0, 0, 0, 0xc3})
	binary.LittleEndian.PutUint64(b.Bytes[2:], uint64(uintptr(unsafe.Pointer(flag))))
	code, err := syscall.Mmap(-1, 0, 4096, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(code)
	copy(code, b.Bytes[:b.Offset()])
	if err := syscall.Mprotect(code, syscall.PROT_READ|syscall.PROT_EXEC); err != nil {
		t.Fatal(err)
	}
	pc := uintptr(unsafe.Pointer(&code[0]))
	handle := Register(Region{Start: pc, End: pc + uintptr(b.Offset()), Unwind: UnwindSkip})
	defer handle.Unregister()
	header := new(FunctionHeader)
	header.SetEntry(pc)
	fn := Function[func() int](header)
	for _, value := range []uint64{0, 1, 0} {
		*flag = value
		want := 42
		if value != 0 {
			want = 17
		}
		if got := fn(); got != want {
			t.Fatalf("flag %d: got %d, want %d", value, got, want)
		}
	}
	runtime.KeepAlive(flag)
}
