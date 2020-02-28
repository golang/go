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
	offsetX86HasAVX  = unsafe.Offsetof(cpu.X86.HasAVX)
	offsetX86HasAVX2 = unsafe.Offsetof(cpu.X86.HasAVX2)
	offsetX86HasERMS = unsafe.Offsetof(cpu.X86.HasERMS)
	offsetX86HasSSE2 = unsafe.Offsetof(cpu.X86.HasSSE2)

	offsetARMHasIDIVA = unsafe.Offsetof(cpu.ARM.HasIDIVA)
)

var (
	// Set in runtime.cpuinit.
	// TODO: deprecate these; use internal/cpu directly.
	x86HasPOPCNT bool
	x86HasSSE41  bool
	x86HasFMA    bool

	armHasVFPv4 bool

	arm64HasATOMICS bool
)
