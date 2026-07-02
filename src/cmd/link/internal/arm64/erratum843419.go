// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/internal/objabi"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"fmt"
)

// Cortex-A53 Erratum 843419
//
// On Cortex-A53 (and some other cores), an ADRP instruction placed at the last
// two word-aligned addresses of a 4KB page (page offset 0xFF8 or 0xFFC) may
// compute an incorrect address when followed by a specific instruction sequence.
//
// The triggering sequences (from ARM-EPM-048406 and LLVM lld AArch64ErrataFix.cpp):
//
// 3-instruction variant (ADRP at page offset 0xFFC):
//
//	Pos1 (0xFFC): ADRP Xn
//	Pos2 (0x000): load/store (not exclusive, not writeback to Xn)
//	Pos3 (0x004): unsigned-immediate load/store using Xn as base
//
// 4-instruction variant (ADRP at page offset 0xFF8 or 0xFFC):
//
//	Pos1 (0xFF8): ADRP Xn
//	Pos2 (0xFFC): load/store (not exclusive, not writeback to Xn)
//	Pos3 (0x000): non-branch instruction
//	Pos4 (0x004): unsigned-immediate load/store using Xn as base
//
// The Go internal linker emits ADRP instructions via R_ADDRARM64 and
// R_ARM64_PCREL_LDST* relocations. These relocations cover an ADRP+op pair
// (siz=8). After address assignment, we know where each ADRP will land. If the
// sequence matches the erratum pattern, we redirect the ADRP+op pair through a
// veneer at a safe address.
//
// GNU ld implements this as --fix-cortex-a53-843419 (enabled by default for
// aarch64). LLVM lld does the same.
//
// Limitation: the detection checks instruction sequences within a single
// symbol's data. If an ADRP is at the very end of a symbol and the following
// instructions are in the next symbol, the pattern will not be detected.
// In practice this is unlikely because Go functions do not end with ADRP
// instructions (they end with RET or branch instructions).

// ARM64 instruction encoding helpers, following LLVM lld's AArch64ErrataFix.cpp.

func isADRP(insn uint32) bool {
	return (insn & 0x9F000000) == 0x90000000
}

func isBranch(insn uint32) bool {
	// Unconditional branch (immediate): B, BL
	if (insn & 0x7C000000) == 0x14000000 {
		return true
	}
	// Compare and branch: CBZ, CBNZ
	if (insn & 0x7E000000) == 0x34000000 {
		return true
	}
	// Conditional branch: B.cond
	if (insn & 0xFF000010) == 0x54000000 {
		return true
	}
	// Test and branch: TBZ, TBNZ
	if (insn & 0x7E000000) == 0x36000000 {
		return true
	}
	// Unconditional branch (register): BR, BLR, RET
	if (insn & 0xFE1F0000) == 0xD61F0000 {
		return true
	}
	return false
}

// getRt returns the Rd/Rt field (bits [4:0]) of an instruction.
func getRt(insn uint32) uint32 {
	return insn & 0x1F
}

// getRn returns the Rn field (bits [9:5]) of an instruction.
func getRn(insn uint32) uint32 {
	return (insn >> 5) & 0x1F
}

// isLoadStoreClass reports whether insn is in the load/store instruction class
// (bit [27] == 1, bit [25] == 0).
func isLoadStoreClass(insn uint32) bool {
	return (insn & 0x0A000000) == 0x08000000
}

// isLoadStoreExclusive reports whether insn is a load/store exclusive
// (bits [31:24] == 0x08 pattern: bit[29:24] == 001000).
func isLoadStoreExclusive(insn uint32) bool {
	return (insn & 0x3F000000) == 0x08000000
}

// isLoadStoreRegisterUnsigned reports whether insn is a load/store with
// unsigned immediate offset (bits [29:24] == 111001 for unscaled or
// [31:24] matches the unsigned imm pattern).
func isLoadStoreRegisterUnsigned(insn uint32) bool {
	// LDR/STR (unsigned immediate): bits [29:24] == 111001
	return (insn & 0x3B000000) == 0x39000000
}

// hasWriteback reports whether a load/store instruction writes back to Rn.
// This covers pre-index and post-index addressing modes.
func hasWriteback(insn uint32) bool {
	// Load/store register (unscaled immediate, pre-index, post-index)
	// bits [29:24] == 111000, bits [11:10] distinguish:
	//   00 = unscaled, 01 = post-index, 11 = pre-index
	if (insn&0x3B200000) == 0x38000000 && (insn&0xC00) != 0 {
		return true
	}
	// Load/store pair (pre/post-index)
	// bits [29:23]: 0x28800000 pattern with bit[23] or writeback bit set
	if (insn & 0x3B800000) == 0x28800000 {
		return true // post-index pair
	}
	if (insn & 0x3B800000) == 0x29800000 {
		return true // pre-index pair
	}
	return false
}

// isValidPos2 checks whether insn is a valid instruction at position 2 of the
// erratum sequence: a load/store that is not exclusive and does not write back
// to register rn.
func isValidPos2(insn uint32, rn uint32) bool {
	if !isLoadStoreClass(insn) {
		return false
	}
	if isLoadStoreExclusive(insn) {
		return false
	}
	if hasWriteback(insn) && getRn(insn) == rn {
		return false
	}
	return true
}

// isValidPos4 checks whether insn is a valid instruction at position 3 or 4
// (the dependent instruction): an unsigned-immediate load/store using rn as
// the base register.
func isValidPos4(insn uint32, rn uint32) bool {
	return isLoadStoreRegisterUnsigned(insn) && getRn(insn) == rn
}

// isErratum843419Reloc reports whether rt is a relocation that produces an
// ADRP instruction on ARM64.
func isErratum843419Reloc(rt objabi.RelocType) bool {
	switch rt {
	case objabi.R_ADDRARM64,
		objabi.R_ARM64_PCREL_LDST8,
		objabi.R_ARM64_PCREL_LDST16,
		objabi.R_ARM64_PCREL_LDST32,
		objabi.R_ARM64_PCREL_LDST64:
		return true
	}
	return false
}

// is843419Sequence reads up to 4 instructions starting at the ADRP and checks
// whether they form an erratum 843419 triggering sequence. It requires access
// to instructions that may be in the NEXT symbol (across a page boundary), so
// it takes a function to read instruction words by virtual address.
//
// Returns true if the sequence matches.
func is843419Sequence(adrpInsn uint32, pageOffset uint64, readInsn func(off int) (uint32, bool)) bool {
	rn := getRt(adrpInsn) // destination register of ADRP

	// Read Pos2 (instruction after ADRP).
	pos2, ok := readInsn(4)
	if !ok {
		return false
	}
	if !isValidPos2(pos2, rn) {
		return false
	}

	// 3-instruction variant: ADRP at 0xFFC, Pos2 at next page, Pos3 is the
	// dependent load/store.
	if pageOffset == 0xFFC {
		pos3, ok := readInsn(8)
		if ok && isValidPos4(pos3, rn) {
			return true
		}
	}

	// 4-instruction variant: ADRP at 0xFF8 or 0xFFC, Pos3 is any non-branch,
	// Pos4 is the dependent load/store.
	pos3, ok := readInsn(8)
	if !ok {
		return false
	}
	if isBranch(pos3) {
		return false
	}
	pos4, ok := readInsn(12)
	if !ok {
		return false
	}
	return isValidPos4(pos4, rn)
}

// erratum843419Check checks whether the ADRP instruction for relocation ri in
// symbol s will land at a dangerous page offset (0xFF8 or 0xFFC within a 4KB
// page) AND the following instructions match the erratum 843419 triggering
// pattern. If so, it generates a veneer that performs the ADRP+op at a safe
// address and patches the original site to branch to the veneer.
func erratum843419Check(ctxt *ld.Link, ldr *loader.Loader, ri int, rs, s loader.Sym) {
	relocs := ldr.Relocs(s)
	r := relocs.At(ri)

	adrpAddr := ldr.SymValue(s) + int64(r.Off())
	pageOffset := uint64(adrpAddr) & 0xFFF

	// With -debugtramp=2, generate veneers for ALL ADRP relocations to stress
	// test the veneer mechanism. Otherwise, only check dangerous page offsets.
	forceVeneer := *ld.FlagDebugTramp >= 2
	if !forceVeneer && pageOffset != 0xFF8 && pageOffset != 0xFFC {
		return
	}

	sb := ldr.MakeSymbolUpdater(s)
	sdata := sb.Data()
	off := r.Off()

	if !forceVeneer {
		// Check whether the instruction sequence actually matches the erratum.
		ioff := int(off)
		adrpInsn := ctxt.Arch.ByteOrder.Uint32(sdata[ioff:])

		if !isADRP(adrpInsn) {
			return
		}

		readInsn := func(delta int) (uint32, bool) {
			pos := ioff + delta
			if pos+4 > len(sdata) {
				return 0, false
			}
			return ctxt.Arch.ByteOrder.Uint32(sdata[pos:]), true
		}

		if !is843419Sequence(adrpInsn, pageOffset, readInsn) {
			return
		}
	}

	rt := r.Type()
	add := r.Add()

	// Create a veneer symbol.
	name := fmt.Sprintf("%s+%d-erratum843419", ldr.SymName(s), off)
	veneer := ldr.LookupOrCreateSym(name, ldr.SymVersion(s))
	ldr.SetAttrReachable(veneer, true)
	if ldr.SymType(veneer) != 0 {
		return // already created
	}

	vb := ldr.MakeSymbolUpdater(veneer)
	ctxt.AddTramp(vb, ldr.SymType(s))
	// Align veneer to 16 bytes so its ADRP (at offset 0) never lands on a
	// page offset of 0xFF8 or 0xFFC. The highest 16-aligned offset in a 4KB
	// page is 0xFF0, which is safe.
	vb.SetAlign(16)

	// Build veneer: ADRP Xd, target + original_second_insn + B back
	vb.SetSize(12)
	P := make([]byte, 12)

	// Read original instructions from the symbol data.
	origADRP := ctxt.Arch.ByteOrder.Uint32(sdata[off:])
	origOP := ctxt.Arch.ByteOrder.Uint32(sdata[off+4:])

	// Veneer instruction 0: ADRP with cleared immediate (relocation fills it).
	adrpRd := origADRP & 0x1F
	ctxt.Arch.ByteOrder.PutUint32(P[0:], 0x90000000|adrpRd)
	// Veneer instruction 1: copy the original second instruction as-is.
	ctxt.Arch.ByteOrder.PutUint32(P[4:], origOP)
	// Veneer instruction 2: B (unconditional branch, placeholder for relocation).
	ctxt.Arch.ByteOrder.PutUint32(P[8:], 0x14000000)
	vb.SetData(P)

	// Relocation 0: ADRP+op pair in the veneer, same type and target as original.
	rAdrp, _ := vb.AddRel(rt)
	rAdrp.SetOff(0)
	rAdrp.SetSiz(8)
	rAdrp.SetSym(rs)
	rAdrp.SetAdd(add)

	// Relocation 1: B back to the instruction after the original pair.
	rBack, _ := vb.AddRel(objabi.R_CALLARM64)
	rBack.SetOff(8)
	rBack.SetSiz(4)
	rBack.SetSym(s)
	rBack.SetAdd(int64(off) + 8)

	// Patch original site: replace ADRP with B to veneer, replace second insn
	// with NOP. The B will be resolved by a R_CALLARM64 relocation.
	sb.MakeWritable()
	sb.SetUint32(ctxt.Arch, int64(off), 0x14000000)   // B (placeholder)
	sb.SetUint32(ctxt.Arch, int64(off)+4, 0xd503201f) // NOP

	// Replace the original relocation with R_CALLARM64 to branch to the veneer.
	rels := sb.Relocs()
	rel := rels.At(ri)
	rel.SetType(objabi.R_CALLARM64)
	rel.SetSym(veneer)
	rel.SetAdd(0)
	rel.SetSiz(4)

	if *ld.FlagDebugTramp > 0 && ctxt.Debugvlog > 0 {
		ctxt.Logf("erratum 843419 veneer for %s+%#x at page offset %#x\n",
			ldr.SymName(s), off, pageOffset)
	}
}
