// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"unsafe"
)

// Offsets into internal/cpu records for use in assembly.
const (
	offsetX86HasAVX2 = unsafe.Offsetof(cpu.X86.HasAVX2)
)

var useAVXmemmove bool

func init() {
	// Let's remove stepping and reserved fields
	processor := processorVersionInfo & 0x0FFF3FF0

	isIntelBridgeFamily := isIntel &&
		processor == 0x206A0 ||
		processor == 0x206D0 ||
		processor == 0x306A0 ||
		processor == 0x306E0

	useAVXmemmove = cpu.X86.HasAVX && !isIntelBridgeFamily
}
