// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

import (
	"internal/goarch"
	"internal/goos"
)

type ArchFamilyType = goarch.ArchFamilyType

const (
	AMD64   = goarch.AMD64
	ARM     = goarch.ARM
	ARM64   = goarch.ARM64
	I386    = goarch.I386
	MIPS    = goarch.MIPS
	MIPS64  = goarch.MIPS64
	PPC64   = goarch.PPC64
	RISCV64 = goarch.RISCV64
	S390X   = goarch.S390X
	WASM    = goarch.WASM
)

// ArchFamily is the architecture family (AMD64, ARM, ...)
const ArchFamily ArchFamilyType = goarch.ArchFamily

// AIX requires a larger stack for syscalls.
const StackGuardMultiplier = StackGuardMultiplierDefault*(1-goos.GoosAix) + 2*goos.GoosAix

// BigEndian reports whether the architecture is big-endian.
const BigEndian = goarch.BigEndian

// DefaultPhysPageSize is the default physical page size.
const DefaultPhysPageSize = goarch.DefaultPhysPageSize

// PCQuantum is the minimal unit for a program counter (1 on x86, 4 on most other systems).
// The various PC tables record PC deltas pre-divided by PCQuantum.
const PCQuantum = goarch.PCQuantum

// Int64Align is the required alignment for a 64-bit integer (4 on 32-bit systems, 8 on 64-bit).
const Int64Align = goarch.PtrSize

// MinFrameSize is the size of the system-reserved words at the bottom
// of a frame (just above the architectural stack pointer).
// It is zero on x86 and PtrSize on most non-x86 (LR-based) systems.
// On PowerPC it is larger, to cover three more reserved words:
// the compiler word, the link editor word, and the TOC save word.
const MinFrameSize = goarch.MinFrameSize

// StackAlign is the required alignment of the SP register.
// The stack must be at least word aligned, but some architectures require more.
const StackAlign = goarch.StackAlign

const GOARCH = goarch.GOARCH

const (
	Goarch386         = goarch.Goarch386
	GoarchAmd64       = goarch.GoarchAmd64
	GoarchAmd64p32    = goarch.GoarchAmd64p32
	GoarchArm         = goarch.GoarchArm
	GoarchArmbe       = goarch.GoarchArmbe
	GoarchArm64       = goarch.GoarchArm64
	GoarchArm64be     = goarch.GoarchArm64be
	GoarchPpc64       = goarch.GoarchPpc64
	GoarchPpc64le     = goarch.GoarchPpc64le
	GoarchMips        = goarch.GoarchMips
	GoarchMipsle      = goarch.GoarchMipsle
	GoarchMips64      = goarch.GoarchMips64
	GoarchMips64le    = goarch.GoarchMips64le
	GoarchMips64p32   = goarch.GoarchMips64p32
	GoarchMips64p32le = goarch.GoarchMips64p32le
	GoarchPpc         = goarch.GoarchPpc
	GoarchRiscv       = goarch.GoarchRiscv
	GoarchRiscv64     = goarch.GoarchRiscv64
	GoarchS390        = goarch.GoarchS390
	GoarchS390x       = goarch.GoarchS390x
	GoarchSparc       = goarch.GoarchSparc
	GoarchSparc64     = goarch.GoarchSparc64
	GoarchWasm        = goarch.GoarchWasm
)

const GOOS = goos.GOOS

const (
	GoosAix       = goos.GoosAix
	GoosAndroid   = goos.GoosAndroid
	GoosDarwin    = goos.GoosDarwin
	GoosDragonfly = goos.GoosDragonfly
	GoosFreebsd   = goos.GoosFreebsd
	GoosHurd      = goos.GoosHurd
	GoosIllumos   = goos.GoosIllumos
	GoosIos       = goos.GoosIos
	GoosJs        = goos.GoosJs
	GoosLinux     = goos.GoosLinux
	GoosNacl      = goos.GoosNacl
	GoosNetbsd    = goos.GoosNetbsd
	GoosOpenbsd   = goos.GoosOpenbsd
	GoosPlan9     = goos.GoosPlan9
	GoosSolaris   = goos.GoosSolaris
	GoosWindows   = goos.GoosWindows
	GoosZos       = goos.GoosZos
)
