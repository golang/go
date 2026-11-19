// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

// Static unwind info for runtime/cgo assembly trampolines that invoke C
// callbacks which may unwind the stack (e.g. via _Unwind_Backtrace or
// backtrace).
//
// Each covered function in cgoUnwindSyms must begin with a standard
// frame-pointer prologue and preserve all callee-saved registers:
//   - amd64: PUSHQ BP (1B); MOVQ SP, BP (3B)
//   - arm64: STP.W (R29, R30), -16(RSP) (4B); MOVD RSP, R29 (4B)
//
// Formats emitted:
//   - Darwin external link: __LD,__compact_unwind (.compact_unwind)
//   - Darwin internal link: __TEXT,__unwind_info (.unwind_info)
//   - ELF external link:    .eh_frame (no zero terminator; external linker merges)
//   - ELF internal link:    .eh_frame (with zero terminator) + .eh_frame_hdr
//                           (referenced by PT_GNU_EH_FRAME in asmbElf)

import (
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"cmp"
	"slices"
)

var cgoUnwindSyms = []string{
	"runtime/cgo.x_cgo_callers",
	"runtime/cgo.x_cgo_call_context",
}

// collectCgoUnwind emits static unwind sections for cgo callback trampolines
// once final text addresses are assigned.
func collectCgoUnwind(ctxt *Link) {
	if ctxt.Arch.Family != sys.AMD64 && ctxt.Arch.Family != sys.ARM64 {
		return
	}
	if !ctxt.IsDarwin() && !ctxt.IsELF {
		return
	}

	ldr := ctxt.loader
	var fns []loader.Sym
	for _, name := range cgoUnwindSyms {
		s := ldr.Lookup(name, 0)
		if s != 0 && ldr.AttrReachable(s) && ldr.SymType(s).IsText() && ldr.SymSize(s) > 0 {
			fns = append(fns, s)
		}
	}
	if len(fns) == 0 {
		return
	}
	slices.SortFunc(fns, func(a, b loader.Sym) int {
		return cmp.Compare(ldr.SymValue(a), ldr.SymValue(b))
	})

	if ctxt.IsDarwin() {
		emitMachoCgoUnwind(ctxt, fns)
	} else {
		emitELFCgoUnwind(ctxt, fns)
	}
}

func emitMachoCgoUnwind(ctxt *Link, fns []loader.Sym) {
	const (
		// From <mach-o/compact_unwind_encoding.h>.
		unwindX8664ModeRBPFrame  = 0x01000000
		unwindARM64ModeFrame     = 0x04000000
		unwindSectionVersion     = 1
		unwindSecondLevelRegular = 2
	)

	var encoding uint32
	switch ctxt.Arch.Family {
	case sys.AMD64:
		encoding = unwindX8664ModeRBPFrame
	case sys.ARM64:
		encoding = unwindARM64ModeFrame
	}

	ldr := ctxt.loader
	if ctxt.IsExternal() {
		// Emit __LD,__compact_unwind for the system linker to merge into __TEXT,__unwind_info.
		sb := ldr.CreateSymForUpdate(".compact_unwind", 0)
		sb.SetType(sym.SMACHO)
		sb.SetAlign(8)
		sb.SetReachable(true)
		for _, fn := range fns {
			// struct compact_unwind_entry: start(8), length(4), encoding(4), personality(8), lsda(8)
			sb.AddAddr(ctxt.Arch, fn)
			sb.AddUint32(ctxt.Arch, uint32(ldr.SymSize(fn)))
			sb.AddUint32(ctxt.Arch, encoding)
			sb.AddUint64(ctxt.Arch, 0)
			sb.AddUint64(ctxt.Arch, 0)
		}
		return
	}

	// Internal linking: emit __TEXT,__unwind_info with a single regular
	// second-level page. All offsets are relative to the __TEXT base.
	const (
		headerSize     = 7 * 4
		indexEntrySize = 3 * 4
		pageHeaderSize = 4 + 2 + 2
		pageOff        = headerSize + 2*indexEntrySize
	)

	textBase := uint64(*FlagTextAddr - int64(HEADR))
	type entry struct {
		funcOff  uint32
		encoding uint32
	}
	entries := make([]entry, 0, 2*len(fns))
	for i, fn := range fns {
		start := uint32(uint64(ldr.SymValue(fn)) - textBase)
		end := uint32(uint64(ldr.SymValue(fn)+ldr.SymSize(fn)) - textBase)
		entries = append(entries, entry{start, encoding})
		if i+1 < len(fns) {
			nextStart := uint32(uint64(ldr.SymValue(fns[i+1])) - textBase)
			if end < nextStart {
				// Mark the gap between functions as having no unwind info.
				entries = append(entries, entry{end, 0})
			}
		}
	}
	last := fns[len(fns)-1]
	lastEnd := uint32(uint64(ldr.SymValue(last)+ldr.SymSize(last)) - textBase)

	sb := ldr.CreateSymForUpdate(".unwind_info", 0)
	sb.SetType(sym.SELFROSECT)
	sb.SetAlign(4)
	sb.SetReachable(true)

	// unwind_info_section_header
	sb.AddUint32(ctxt.Arch, unwindSectionVersion)
	sb.AddUint32(ctxt.Arch, headerSize) // commonEncodingsArraySectionOffset
	sb.AddUint32(ctxt.Arch, 0)          // commonEncodingsArrayCount
	sb.AddUint32(ctxt.Arch, headerSize) // personalityArraySectionOffset
	sb.AddUint32(ctxt.Arch, 0)          // personalityArrayCount
	sb.AddUint32(ctxt.Arch, headerSize) // indexSectionOffset
	sb.AddUint32(ctxt.Arch, 2)          // indexCount

	// Top-level index: one range entry plus one end sentinel.
	sb.AddUint32(ctxt.Arch, entries[0].funcOff)
	sb.AddUint32(ctxt.Arch, pageOff)
	sb.AddUint32(ctxt.Arch, pageOff)
	sb.AddUint32(ctxt.Arch, lastEnd)
	sb.AddUint32(ctxt.Arch, 0)
	sb.AddUint32(ctxt.Arch, pageOff)

	// unwind_info_regular_second_level_page_header
	sb.AddUint32(ctxt.Arch, unwindSecondLevelRegular)
	sb.AddUint16(ctxt.Arch, pageHeaderSize)
	sb.AddUint16(ctxt.Arch, uint16(len(entries)))

	// unwind_info_regular_second_level_entry[]
	for _, e := range entries {
		sb.AddUint32(ctxt.Arch, e.funcOff)
		sb.AddUint32(ctxt.Arch, e.encoding)
	}
}

func emitELFCgoUnwind(ctxt *Link, fns []loader.Sym) {
	const (
		dwEhPeUdata4  = 0x03
		dwEhPeSdata4  = 0x0b
		dwEhPePcrel   = 0x10
		dwEhPeDatarel = 0x30

		ciePayloadLen = 20 // 4+20 = 24 bytes (8-byte aligned)
		fdePayloadLen = 28 // 4+28 = 32 bytes (8-byte aligned)
	)

	ldr := ctxt.loader
	eh := ldr.CreateSymForUpdate(".eh_frame", 0)
	eh.SetType(sym.SELFROSECT)
	eh.SetAlign(8)
	eh.SetReachable(true)

	// CIE (24 bytes)
	eh.AddUint32(ctxt.Arch, ciePayloadLen)
	eh.AddUint32(ctxt.Arch, 0) // CIE_id
	switch ctxt.Arch.Family {
	case sys.AMD64:
		eh.AddBytes([]byte{
			1,           // version
			'z', 'R', 0, // augmentation
			1,                          // code alignment factor
			0x78,                       // data alignment factor (-8)
			16,                         // return address reg (RIP)
			1,                          // augmentation data length
			dwEhPePcrel | dwEhPeSdata4, // FDE pointer encoding
			0x0c, 7, 8,                 // DW_CFA_def_cfa: RSP + 8
			0x80 | 16, 1, // DW_CFA_offset: RIP at CFA - 8
			0, 0, // DW_CFA_nop padding
		})
	case sys.ARM64:
		eh.AddBytes([]byte{
			1,           // version
			'z', 'R', 0, // augmentation
			1,                          // code alignment factor
			0x78,                       // data alignment factor (-8)
			30,                         // return address reg (X30/LR)
			1,                          // augmentation data length
			dwEhPePcrel | dwEhPeSdata4, // FDE pointer encoding
			0x0c, 31, 0,                // DW_CFA_def_cfa: SP + 0
			0, 0, 0, 0, // DW_CFA_nop padding
		})
	}

	// One FDE (32 bytes) per function.
	fdeOffs := make([]int64, len(fns))
	for i, fn := range fns {
		fdeOff := eh.Size()
		fdeOffs[i] = fdeOff
		eh.AddUint32(ctxt.Arch, fdePayloadLen)
		eh.AddUint32(ctxt.Arch, uint32(fdeOff+4)) // CIE_pointer back to offset 0
		// DW_EH_PE_pcrel is relative to the field itself; R_PCREL subtracts place+4.
		eh.AddPCRelPlus(ctxt.Arch, fn, 4)
		eh.AddUint32(ctxt.Arch, uint32(ldr.SymSize(fn)))
		eh.AddUint8(0) // augmentation data length
		switch ctxt.Arch.Family {
		case sys.AMD64:
			eh.AddBytes([]byte{
				0x40 + 1, // DW_CFA_advance_loc 1 (PUSHQ BP)
				0x0e, 16, // DW_CFA_def_cfa_offset 16
				0x80 | 6, 2, // DW_CFA_offset RBP at CFA - 16
				0x40 + 3, // DW_CFA_advance_loc 3 (MOVQ SP, BP)
				0x0d, 6,  // DW_CFA_def_cfa_register RBP
				0, 0, 0, 0, 0, 0, 0,
			})
		case sys.ARM64:
			eh.AddBytes([]byte{
				0x40 + 4, // DW_CFA_advance_loc 4 (STP.W (R29, R30), -16(RSP))
				0x0e, 16, // DW_CFA_def_cfa_offset 16
				0x80 | 29, 2, // DW_CFA_offset X29 at CFA - 16
				0x80 | 30, 1, // DW_CFA_offset X30 at CFA - 8
				0x40 + 4, // DW_CFA_advance_loc 4 (MOVD RSP, R29)
				0x0d, 29, // DW_CFA_def_cfa_register X29
				0, 0, 0, 0, 0,
			})
		}
	}

	if ctxt.IsExternal() {
		// External linker merges .eh_frame, appends terminator, and builds .eh_frame_hdr.
		return
	}

	// Zero terminator for final .eh_frame.
	eh.AddUint32(ctxt.Arch, 0)

	// Build .eh_frame_hdr so libgcc_s / libunwind finds the FDEs via PT_GNU_EH_FRAME.
	hdr := ldr.CreateSymForUpdate(".eh_frame_hdr", 0)
	hdr.SetType(sym.SELFROSECT)
	hdr.SetAlign(4)
	hdr.SetReachable(true)

	hdr.AddUint8(1)                            // version
	hdr.AddUint8(dwEhPePcrel | dwEhPeSdata4)   // eh_frame_ptr_enc
	hdr.AddUint8(dwEhPeUdata4)                 // fde_count_enc
	hdr.AddUint8(dwEhPeDatarel | dwEhPeSdata4) // table_enc

	hdr.AddPCRelPlus(ctxt.Arch, eh.Sym(), 4)
	hdr.AddUint32(ctxt.Arch, uint32(len(fns)))

	// Binary search table sorted by initial_loc. Entries are relative to the
	// start of .eh_frame_hdr: at byte offset off inside hdr, R_PCREL with addend
	// off+4 resolves to target + (off+4) - (hdr+off+4) = target - hdr.
	for i, fn := range fns {
		hdr.AddPCRelPlus(ctxt.Arch, fn, hdr.Size()+4)
		hdr.AddPCRelPlus(ctxt.Arch, eh.Sym(), fdeOffs[i]+hdr.Size()+4)
	}
}
