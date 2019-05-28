// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytealg

import (
	"internal/cpu"
	"unsafe"
)

// Offsets into internal/cpu records for use in assembly.
const (
	offsetX86HasSSE2   = unsafe.Offsetof(cpu.X86.HasSSE2)
	offsetX86HasSSE42  = unsafe.Offsetof(cpu.X86.HasSSE42)
	offsetX86HasAVX2   = unsafe.Offsetof(cpu.X86.HasAVX2)
	offsetX86HasPOPCNT = unsafe.Offsetof(cpu.X86.HasPOPCNT)

	offsetS390xHasVX = unsafe.Offsetof(cpu.S390X.HasVX)
)

// MaxLen is the maximum length of the string to be searched for (argument b) in Index.
var MaxLen int
