// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !msan

package msan

import (
	"unsafe"
)

const Enabled = false

func Read(addr unsafe.Pointer, sz uintptr) {
}

func Write(addr unsafe.Pointer, sz uintptr) {
}

func Malloc(addr unsafe.Pointer, sz uintptr) {
}

func Free(addr unsafe.Pointer, sz uintptr) {
}

func Move(dst, src unsafe.Pointer, sz uintptr) {
}
