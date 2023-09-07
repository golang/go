// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"debug/elf"
	"debug/macho"
)

// RelocVariant is a linker-internal variation on a relocation.
type RelocVariant uint8

const (
	RV_NONE RelocVariant = iota
	RV_POWER_LO
	RV_POWER_HI
	RV_POWER_HA
	RV_POWER_DS

	// RV_390_DBL is a s390x-specific relocation variant that indicates that
	// the value to be placed into the relocatable field should first be
	// divided by 2.
	RV_390_DBL

	RV_CHECK_OVERFLOW RelocVariant = 1 << 7
	RV_TYPE_MASK      RelocVariant = RV_CHECK_OVERFLOW - 1
)

func RelocName(arch *sys.Arch, r objabi.RelocType) string {
	switch {
	case r >= objabi.MachoRelocOffset: // Mach-O
		nr := (r - objabi.MachoRelocOffset) >> 1
		switch arch.Family {
		case sys.AMD64:
			return macho.RelocTypeX86_64(nr).String()
		case sys.ARM64:
			return macho.RelocTypeARM64(nr).String()
		default:
			panic("unreachable")
		}
	case r >= objabi.ElfRelocOffset: // ELF
		nr := r - objabi.ElfRelocOffset
		switch arch.Family {
		case sys.AMD64:
			return elf.R_X86_64(nr).String()
		case sys.ARM:
			return elf.R_ARM(nr).String()
		case sys.ARM64:
			return elf.R_AARCH64(nr).String()
		case sys.I386:
			return elf.R_386(nr).String()
		case sys.Loong64:
			return elf.R_LARCH(nr).String()
		case sys.MIPS, sys.MIPS64:
			return elf.R_MIPS(nr).String()
		case sys.PPC64:
			return elf.R_PPC64(nr).String()
		case sys.S390X:
			return elf.R_390(nr).String()
		case sys.RISCV64:
			return elf.R_RISCV(nr).String()
		default:
			panic("unreachable")
		}
	}

	return r.String()
}
