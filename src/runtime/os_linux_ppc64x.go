// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

package runtime

import (
	"runtime/internal/sys"
)

const (
	// ISA level
	// Go currently requires POWER5 as a minimum for ppc64, so we need
	// to check for ISA 2.03 and beyond.
	_PPC_FEATURE_POWER5_PLUS = 0x00020000 // ISA 2.03 (POWER5+)
	_PPC_FEATURE_ARCH_2_05   = 0x00001000 // ISA 2.05 (POWER6)
	_PPC_FEATURE_POWER6_EXT  = 0x00000200 // mffgpr/mftgpr extension (POWER6x)
	_PPC_FEATURE_ARCH_2_06   = 0x00000100 // ISA 2.06 (POWER7)
	_PPC_FEATURE2_ARCH_2_07  = 0x80000000 // ISA 2.07 (POWER8)

	// Standalone capabilities
	_PPC_FEATURE_HAS_ALTIVEC = 0x10000000 // SIMD/Vector unit
	_PPC_FEATURE_HAS_VSX     = 0x00000080 // Vector scalar unit
)

type facilities struct {
	_         [sys.CacheLineSize]byte
	isPOWER5x bool // ISA 2.03
	isPOWER6  bool // ISA 2.05
	isPOWER6x bool // ISA 2.05 + mffgpr/mftgpr extension
	isPOWER7  bool // ISA 2.06
	isPOWER8  bool // ISA 2.07
	hasVMX    bool // Vector unit
	hasVSX    bool // Vector scalar unit
	_         [sys.CacheLineSize]byte
}

// cpu can be tested at runtime in go assembler code to check for
// a certain ISA level or hardware capability, for example:
//	  ·cpu+facilities_hasVSX(SB) for checking the availability of VSX
//	  or
//	  ·cpu+facilities_isPOWER7(SB) for checking if the processor implements
//	  ISA 2.06 instructions.
var cpu facilities

func archauxv(tag, val uintptr) {
	switch tag {
	case _AT_HWCAP:
		cpu.isPOWER5x = val&_PPC_FEATURE_POWER5_PLUS != 0
		cpu.isPOWER6 = val&_PPC_FEATURE_ARCH_2_05 != 0
		cpu.isPOWER6x = val&_PPC_FEATURE_POWER6_EXT != 0
		cpu.isPOWER7 = val&_PPC_FEATURE_ARCH_2_06 != 0
		cpu.hasVMX = val&_PPC_FEATURE_HAS_ALTIVEC != 0
		cpu.hasVSX = val&_PPC_FEATURE_HAS_VSX != 0
	case _AT_HWCAP2:
		cpu.isPOWER8 = val&_PPC_FEATURE2_ARCH_2_07 != 0
	}
}
