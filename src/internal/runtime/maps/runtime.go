// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps

import (
	"internal/abi"
	"unsafe"
)

// Functions below pushed from runtime.

//go:linkname fatal
func fatal(s string)

//go:linkname rand
func rand() uint64

//go:linkname typedmemmove
func typedmemmove(typ *abi.Type, dst, src unsafe.Pointer)

//go:linkname typedmemclr
func typedmemclr(typ *abi.Type, ptr unsafe.Pointer)

//go:linkname newarray
func newarray(typ *abi.Type, n int) unsafe.Pointer

//go:linkname newobject
func newobject(typ *abi.Type) unsafe.Pointer
