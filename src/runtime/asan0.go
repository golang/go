// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !asan

// Dummy ASan support API, used when not built with -asan.

package runtime

import (
	"unsafe"
)

const asanenabled = false
const asanenabledBit = 0

// Because asanenabled is false, none of these functions should be called.

func asanread(addr unsafe.Pointer, sz uintptr)            { throw("asan") }
func asanwrite(addr unsafe.Pointer, sz uintptr)           { throw("asan") }
func asanunpoison(addr unsafe.Pointer, sz uintptr)        { throw("asan") }
func asanpoison(addr unsafe.Pointer, sz uintptr)          { throw("asan") }
func asanregisterglobals(addr unsafe.Pointer, sz uintptr) { throw("asan") }
func lsanregisterrootregion(unsafe.Pointer, uintptr)      { throw("asan") }
func lsanunregisterrootregion(unsafe.Pointer, uintptr)    { throw("asan") }
func lsandoleakcheck()                                    { throw("asan") }
