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
	x86_HasSSE2   = unsafe.Offsetof(cpu.X86.HasSSE2)
	x86_HasSSE42  = unsafe.Offsetof(cpu.X86.HasSSE42)
	x86_HasAVX2   = unsafe.Offsetof(cpu.X86.HasAVX2)
	x86_HasPOPCNT = unsafe.Offsetof(cpu.X86.HasPOPCNT)
	s390x_HasVX   = unsafe.Offsetof(cpu.S390X.HasVX)
)

// MaxLen is the maximum length of the string to be searched for (argument b) in Index.
var MaxLen int
