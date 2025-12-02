// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package secret

import (
	"internal/cpu"
	"unsafe"
)

// exports for assembly testing functions
const (
	offsetX86HasAVX    = unsafe.Offsetof(cpu.X86.HasAVX)
	offsetX86HasAVX512 = unsafe.Offsetof(cpu.X86.HasAVX512)
)
