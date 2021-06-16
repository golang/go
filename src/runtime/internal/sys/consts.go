// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

import (
	"internal/goarch"
	"internal/goos"
)

// AIX requires a larger stack for syscalls.
const StackGuardMultiplier = StackGuardMultiplierDefault*(1-goos.IsAix) + 2*goos.IsAix

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

const (
	Goarch386         = goarch.Is386
	GoarchAmd64       = goarch.IsAmd64
	GoarchAmd64p32    = goarch.IsAmd64p32
	GoarchArm         = goarch.IsArm
	GoarchArmbe       = goarch.IsArmbe
	GoarchArm64       = goarch.IsArm64
	GoarchArm64be     = goarch.IsArm64be
	GoarchPpc64       = goarch.IsPpc64
	GoarchPpc64le     = goarch.IsPpc64le
	GoarchMips        = goarch.IsMips
	GoarchMipsle      = goarch.IsMipsle
	GoarchMips64      = goarch.IsMips64
	GoarchMips64le    = goarch.IsMips64le
	GoarchMips64p32   = goarch.IsMips64p32
	GoarchMips64p32le = goarch.IsMips64p32le
	GoarchPpc         = goarch.IsPpc
	GoarchRiscv       = goarch.IsRiscv
	GoarchRiscv64     = goarch.IsRiscv64
	GoarchS390        = goarch.IsS390
	GoarchS390x       = goarch.IsS390x
	GoarchSparc       = goarch.IsSparc
	GoarchSparc64     = goarch.IsSparc64
	GoarchWasm        = goarch.IsWasm
)

const (
	GoosAix       = goos.IsAix
	GoosAndroid   = goos.IsAndroid
	GoosDarwin    = goos.IsDarwin
	GoosDragonfly = goos.IsDragonfly
	GoosFreebsd   = goos.IsFreebsd
	GoosHurd      = goos.IsHurd
	GoosIllumos   = goos.IsIllumos
	GoosIos       = goos.IsIos
	GoosJs        = goos.IsJs
	GoosLinux     = goos.IsLinux
	GoosNacl      = goos.IsNacl
	GoosNetbsd    = goos.IsNetbsd
	GoosOpenbsd   = goos.IsOpenbsd
	GoosPlan9     = goos.IsPlan9
	GoosSolaris   = goos.IsSolaris
	GoosWindows   = goos.IsWindows
	GoosZos       = goos.IsZos
)
