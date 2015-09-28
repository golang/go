// run

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// windows doesn't work, because Windows exception handling
// delivers signals based on the current PC, and that current PC
// doesn't go into the Go runtime.
// +build !windows

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

	// We want to force an illegal instruction, to get a crash
	// at a PC value != 0.
	// Not all systems make the data section non-executable.
	ill := make([]byte, 64)
	switch runtime.GOARCH {
	case "386", "amd64":
		binary.LittleEndian.PutUint16(ill, 0x0b0f) // ud2
	case "arm":
		binary.LittleEndian.PutUint32(ill, 0xe7f000f0) // no name, but permanently undefined
	case "arm64":
		binary.LittleEndian.PutUint32(ill, 0xd4207d00) // brk #1000
	case "ppc64":
		binary.BigEndian.PutUint32(ill, 0x7fe00008) // trap
	case "ppc64le":
		binary.LittleEndian.PutUint32(ill, 0x7fe00008) // trap
	case "mips64":
		binary.BigEndian.PutUint32(ill, 0x00000034) // trap
	case "mips64le":
		binary.LittleEndian.PutUint32(ill, 0x00000034) // trap
	default:
		// Just leave it as 0 and hope for the best.
	}

	f.x = uintptr(unsafe.Pointer(&ill[0]))
	fn := *(*func())(unsafe.Pointer(&f))
	fn()
}
