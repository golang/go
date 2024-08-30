// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/binary"
	"runtime"
	"runtime/debug"
	"unsafe"
)

func main() {
	debug.SetPanicOnFault(true)
	defer func() {
		if err := recover(); err == nil {
			panic("not panicking")
		}
		pc, _, _, _ := runtime.Caller(10)
		f := runtime.FuncForPC(pc)
		if f == nil || f.Name() != "main.f" {
			if f == nil {
				println("no func for ", unsafe.Pointer(pc))
			} else {
				println("found func:", f.Name())
			}
			panic("cannot find main.f on stack")
		}
	}()
	f(20)
}

func f(n int) {
	if n > 0 {
		f(n - 1)
	}
	var f struct {
		x uintptr
	}

	// We want to force a seg fault, to get a crash at a PC value != 0.
	// Not all systems make the data section non-executable.
	ill := make([]byte, 64)
	switch runtime.GOARCH {
	case "386", "amd64":
		ill = append(ill[:0], 0x89, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00) // MOVL AX, 0
	case "arm":
		binary.LittleEndian.PutUint32(ill[0:4], 0xe3a00000) // MOVW $0, R0
		binary.LittleEndian.PutUint32(ill[4:8], 0xe5800000) // MOVW R0, (R0)
	case "arm64":
		binary.LittleEndian.PutUint32(ill, 0xf90003ff) // MOVD ZR, (ZR)
	case "ppc64":
		binary.BigEndian.PutUint32(ill, 0xf8000000) // MOVD R0, (R0)
	case "ppc64le":
		binary.LittleEndian.PutUint32(ill, 0xf8000000) // MOVD R0, (R0)
	case "mips", "mips64":
		binary.BigEndian.PutUint32(ill, 0xfc000000) // MOVV R0, (R0)
	case "mipsle", "mips64le":
		binary.LittleEndian.PutUint32(ill, 0xfc000000) // MOVV R0, (R0)
	case "s390x":
		ill = append(ill[:0], 0xa7, 0x09, 0x00, 0x00)         // MOVD $0, R0
		ill = append(ill, 0xe3, 0x00, 0x00, 0x00, 0x00, 0x24) // MOVD R0, (R0)
	case "riscv64":
		binary.LittleEndian.PutUint32(ill, 0x00003023) // MOV X0, (X0)
	default:
		// Just leave it as 0 and hope for the best.
	}

	f.x = uintptr(unsafe.Pointer(&ill[0]))
	p := &f
	fn := *(*func())(unsafe.Pointer(&p))
	syncIcache(f.x)
	fn()
}
