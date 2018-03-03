// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytealg

import (
	"internal/cpu"
	"unsafe"
)

// Note: there's no equal_generic.go because every platform must implement at least memequal_varlen in assembly.

// Because equal_native.go is unconditional, it's a good place to compute asm constants.
// TODO: find a better way to do this?

// Offsets into internal/cpu records for use in assembly.
const x86_HasSSE2 = unsafe.Offsetof(cpu.X86.HasSSE2)
const x86_HasAVX2 = unsafe.Offsetof(cpu.X86.HasAVX2)
const s390x_HasVX = unsafe.Offsetof(cpu.S390X.HasVX)

//go:noescape
func Equal(a, b []byte) bool

// The compiler generates calls to runtime.memequal and runtime.memequal_varlen.
// In addition, the runtime calls runtime.memequal explicitly.
// Those functions are implemented in this package.
