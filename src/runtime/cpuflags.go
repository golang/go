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
	offsetX86HasAVX    = unsafe.Offsetof(cpu.X86.HasAVX)
	offsetX86HasAVX2   = unsafe.Offsetof(cpu.X86.HasAVX2)
	offsetX86HasERMS   = unsafe.Offsetof(cpu.X86.HasERMS)
	offsetX86HasRDTSCP = unsafe.Offsetof(cpu.X86.HasRDTSCP)

	offsetARMHasIDIVA = unsafe.Offsetof(cpu.ARM.HasIDIVA)

	offsetMIPS64XHasMSA = unsafe.Offsetof(cpu.MIPS64X.HasMSA)

	offsetLOONG64HasLSX = unsafe.Offsetof(cpu.Loong64.HasLSX)
)

var (
	// Set in runtime.cpuinit.
	// TODO: deprecate these; use internal/cpu directly.
	x86HasPOPCNT bool
	x86HasSSE41  bool
	x86HasFMA    bool

	armHasVFPv4 bool

	arm64HasATOMICS bool

	loong64HasLAMCAS bool
	loong64HasLAM_BH bool
	loong64HasLSX    bool
)
