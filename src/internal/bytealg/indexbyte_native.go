// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 amd64 amd64p32 s390x arm arm64 ppc64 ppc64le mips mipsle mips64 mips64le

package bytealg

import (
	"internal/cpu"
	"unsafe"
)

// Offsets into internal/cpu records for use in assembly
// TODO: find a better way to do this?
const x86_HasAVX2 = unsafe.Offsetof(cpu.X86.HasAVX2)
const s390x_HasVX = unsafe.Offsetof(cpu.S390X.HasVX)

//go:noescape
func IndexByte(b []byte, c byte) int

//go:noescape
func IndexByteString(s string, c byte) int
