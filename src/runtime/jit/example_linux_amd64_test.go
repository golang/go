// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && linux && amd64

package jit_test

import (
	"encoding/binary"
	"fmt"
	"runtime/jit"
	"syscall"
	"unsafe"
)

// This complete consumer is independent of any JIT compiler or arena library.
// More complex functions must additionally publish exact call-site stack maps.
func ExampleFunction() {
	c := jit.GetCallingConventionForFunc((func() uint64)(nil))
	result := c.Outputs[0].Parts[0].Location.Register
	b := &jit.SliceBuffer{Bytes: make([]byte, 64)}
	// MOV imm64 to the toolchain's integer result register.
	instruction := []byte{0x48 | result.Number>>3, 0xb8 | result.Number&7, 0, 0, 0, 0, 0, 0, 0, 0}
	binary.LittleEndian.PutUint64(instruction[2:], 42)
	b.Write(instruction)
	jit.EmitLeafReturn(b)
	code, err := syscall.Mmap(-1, 0, 4096, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		panic(err)
	}
	defer syscall.Munmap(code)
	copy(code, b.Bytes[:b.Offset()])
	if err := syscall.Mprotect(code, syscall.PROT_READ|syscall.PROT_EXEC); err != nil {
		panic(err)
	}
	pc := uintptr(unsafe.Pointer(&code[0]))
	handle := jit.Register(jit.Region{Start: pc, End: pc + uintptr(b.Offset()), Unwind: jit.UnwindSkip})
	defer handle.Unregister()
	context := new(jit.FunctionHeader)
	context.SetEntry(pc)
	fn := jit.Function[func() uint64](context)
	fmt.Println(fn())
	// Output: 42
}
