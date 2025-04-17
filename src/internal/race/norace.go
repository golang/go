// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !race

package race

import (
	"internal/abi"
	"unsafe"
)

const Enabled = false

func Acquire(addr unsafe.Pointer) {
}

func Release(addr unsafe.Pointer) {
}

func ReleaseMerge(addr unsafe.Pointer) {
}

func Disable() {
}

func Enable() {
}

func Read(addr unsafe.Pointer) {
}

func ReadPC(addr unsafe.Pointer, callerpc, pc uintptr) {
}

func ReadObjectPC(t *abi.Type, addr unsafe.Pointer, callerpc, pc uintptr) {
}

func Write(addr unsafe.Pointer) {
}

func WritePC(addr unsafe.Pointer, callerpc, pc uintptr) {
}

func WriteObjectPC(t *abi.Type, addr unsafe.Pointer, callerpc, pc uintptr) {
}

func ReadRange(addr unsafe.Pointer, len int) {
}

func WriteRange(addr unsafe.Pointer, len int) {
}

func Errors() int { return 0 }
