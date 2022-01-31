// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package goarch contains GOARCH-specific constants.
package goarch

// The next line makes 'go generate' write the zgoarch*.go files with
// per-arch information, including constants named $GOARCH for every
// GOARCH. The constant is 1 on the current system, 0 otherwise; multiplying
// by them is useful for defining GOARCH-specific constants.
//
//go:generate go run gengoarch.go

type ArchFamilyType int

const (
	AMD64 ArchFamilyType = iota
	ARM
	ARM64
	I386
	MIPS
	MIPS64
	PPC64
	RISCV64
	S390X
	WASM
)

// PtrSize is the size of a pointer in bytes - unsafe.Sizeof(uintptr(0)) but as an ideal constant.
// It is also the size of the machine's native word size (that is, 4 on 32-bit systems, 8 on 64-bit).
const PtrSize = 4 << (^uintptr(0) >> 63)

// ArchFamily is the architecture family (AMD64, ARM, ...)
const ArchFamily ArchFamilyType = _ArchFamily

// BigEndian reports whether the architecture is big-endian.
const BigEndian = IsArmbe|IsArm64be|IsMips|IsMips64|IsPpc|IsPpc64|IsS390|IsS390x|IsSparc|IsSparc64 == 1

// DefaultPhysPageSize is the default physical page size.
const DefaultPhysPageSize = _DefaultPhysPageSize

// PCQuantum is the minimal unit for a program counter (1 on x86, 4 on most other systems).
// The various PC tables record PC deltas pre-divided by PCQuantum.
const PCQuantum = _PCQuantum

// Int64Align is the required alignment for a 64-bit integer (4 on 32-bit systems, 8 on 64-bit).
const Int64Align = PtrSize

// MinFrameSize is the size of the system-reserved words at the bottom
// of a frame (just above the architectural stack pointer).
// It is zero on x86 and PtrSize on most non-x86 (LR-based) systems.
// On PowerPC it is larger, to cover three more reserved words:
// the compiler word, the link editor word, and the TOC save word.
const MinFrameSize = _MinFrameSize

// StackAlign is the required alignment of the SP register.
// The stack must be at least word aligned, but some architectures require more.
const StackAlign = _StackAlign
