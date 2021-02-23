// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !msan

// Dummy MSan support API, used when not built with -msan.

package runtime

import (
	"unsafe"
)

const msanenabled = false

// Because msanenabled is false, none of these functions should be called.

func msanread(addr unsafe.Pointer, sz uintptr)     { throw("msan") }
func msanwrite(addr unsafe.Pointer, sz uintptr)    { throw("msan") }
func msanmalloc(addr unsafe.Pointer, sz uintptr)   { throw("msan") }
func msanfree(addr unsafe.Pointer, sz uintptr)     { throw("msan") }
func msanmove(dst, src unsafe.Pointer, sz uintptr) { throw("msan") }
