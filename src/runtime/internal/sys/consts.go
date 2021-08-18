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
