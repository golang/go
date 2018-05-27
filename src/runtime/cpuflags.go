// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"unsafe"
)

// Offsets into internal/cpu records for use in assembly.
const (
	offset_x86_HasAVX2 = unsafe.Offsetof(cpu.X86.HasAVX2)
	offset_x86_HasERMS = unsafe.Offsetof(cpu.X86.HasERMS)
	offset_x86_HasSSE2 = unsafe.Offsetof(cpu.X86.HasSSE2)

	offset_arm_HasIDIVA = unsafe.Offsetof(cpu.ARM.HasIDIVA)
)
