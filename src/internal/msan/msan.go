// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build msan

package msan

import (
	"unsafe"
)

const Enabled = true

func Read(addr unsafe.Pointer, sz uintptr) {
	read(addr, sz)
}

func Write(addr unsafe.Pointer, sz uintptr) {
	write(addr, sz)
}

func Malloc(addr unsafe.Pointer, sz uintptr) {
	malloc(addr, sz)
}

func Free(addr unsafe.Pointer, sz uintptr) {
	free(addr, sz)
}

func Move(dst, src unsafe.Pointer, sz uintptr) {
	move(dst, src, sz)
}

// Import private msan functions from runtime.
//
//go:linkname read runtime.msanread
func read(addr unsafe.Pointer, sz uintptr)

//go:linkname write runtime.msanwrite
func write(addr unsafe.Pointer, sz uintptr)

//go:linkname malloc runtime.msanmalloc
func malloc(addr unsafe.Pointer, sz uintptr)

//go:linkname free runtime.msanfree
func free(addr unsafe.Pointer, sz uintptr)

//go:linkname move runtime.msanmove
func move(dst, src unsafe.Pointer, sz uintptr)
