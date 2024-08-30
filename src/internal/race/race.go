// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

package race

import (
	"runtime"
	"unsafe"
)

const Enabled = true

func Acquire(addr unsafe.Pointer) {
	runtime.RaceAcquire(addr)
}

func Release(addr unsafe.Pointer) {
	runtime.RaceRelease(addr)
}

func ReleaseMerge(addr unsafe.Pointer) {
	runtime.RaceReleaseMerge(addr)
}

func Disable() {
	runtime.RaceDisable()
}

func Enable() {
	runtime.RaceEnable()
}

func Read(addr unsafe.Pointer) {
	runtime.RaceRead(addr)
}

func Write(addr unsafe.Pointer) {
	runtime.RaceWrite(addr)
}

func ReadRange(addr unsafe.Pointer, len int) {
	runtime.RaceReadRange(addr, len)
}

func WriteRange(addr unsafe.Pointer, len int) {
	runtime.RaceWriteRange(addr, len)
}

func Errors() int {
	return runtime.RaceErrors()
}
