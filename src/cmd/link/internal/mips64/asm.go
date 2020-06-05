// Inferno utils/5l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/5l/asm.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package mips64

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
)

func gentext(ctxt *ld.Link, ldr *loader.Loader) {}

func elfreloc1(ctxt *ld.Link, ldr *loader.Loader, s loader.Sym, r loader.ExtRelocView, sectoff int64) bool {

	// mips64 ELF relocation (endian neutral)
	//		offset	uint64
	//		sym		uint32
	//		ssym	uint8
	//		type3	uint8
	//		type2	uint8
	//		type	uint8
	//		addend	int64

	ctxt.Out.Write64(uint64(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	ctxt.Out.Write32(uint32(elfsym))
	ctxt.Out.Write8(0)
	ctxt.Out.Write8(0)
	ctxt.Out.Write8(0)
	switch r.Type() {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		switch r.Siz() {
		case 4:
			ctxt.Out.Write8(uint8(elf.R_MIPS_32))
		case 8:
			ctxt.Out.Write8(uint8(elf.R_MIPS_64))
		default:
			return false
		}
	case objabi.R_ADDRMIPS:
		ctxt.Out.Write8(uint8(elf.R_MIPS_LO16))
	case objabi.R_ADDRMIPSU:
		ctxt.Out.Write8(uint8(elf.R_MIPS_HI16))
	case objabi.R_ADDRMIPSTLS:
		ctxt.Out.Write8(uint8(elf.R_MIPS_TLS_TPREL_LO16))
	case objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		ctxt.Out.Write8(uint8(elf.R_MIPS_26))
	}
	ctxt.Out.Write64(uint64(r.Xadd))

	return true
}

func elfsetupplt(ctxt *ld.Link, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	return
}

func machoreloc1(*sys.Arch, *ld.OutBuf, *loader.Loader, loader.Sym, loader.ExtRelocView, int64) bool {
	return false
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc2, rr *loader.ExtReloc, s loader.Sym, val int64) (o int64, needExtReloc bool, ok bool) {
	rs := r.Sym()
	rs = ldr.ResolveABIAlias(rs)
	if target.IsExternal() {
		switch r.Type() {
		default:
			return val, false, false

		case objabi.R_ADDRMIPS,
			objabi.R_ADDRMIPSU:
			// set up addend for eventual relocation via outer symbol.
			rs, off := ld.FoldSubSymbolOffset(ldr, rs)
			rr.Xadd = r.Add() + off
			rst := ldr.SymType(rs)
			if rst != sym.SHOSTOBJ && rst != sym.SDYNIMPORT && ldr.SymSect(rs) == nil {
				ldr.Errorf(s, "missing section for %s", ldr.SymName(rs))
			}
			rr.Xsym = rs
			return val, true, true

		case objabi.R_ADDRMIPSTLS,
			objabi.R_CALLMIPS,
			objabi.R_JMPMIPS:
			rr.Xsym = rs
			rr.Xadd = r.Add()
			return val, true, true
		}
	}

	const isOk = true
	const noExtReloc = false
	switch r.Type() {
	case objabi.R_ADDRMIPS,
		objabi.R_ADDRMIPSU:
		t := ldr.SymValue(rs) + r.Add()
		o1 := target.Arch.ByteOrder.Uint32(ldr.OutData(s)[r.Off():])
		if r.Type() == objabi.R_ADDRMIPS {
			return int64(o1&0xffff0000 | uint32(t)&0xffff), noExtReloc, isOk
		}
		return int64(o1&0xffff0000 | uint32((t+1<<15)>>16)&0xffff), noExtReloc, isOk
	case objabi.R_ADDRMIPSTLS:
		// thread pointer is at 0x7000 offset from the start of TLS data area
		t := ldr.SymValue(rs) + r.Add() - 0x7000
		if t < -32768 || t >= 32678 {
			ldr.Errorf(s, "TLS offset out of range %d", t)
		}
		o1 := target.Arch.ByteOrder.Uint32(ldr.OutData(s)[r.Off():])
		return int64(o1&0xffff0000 | uint32(t)&0xffff), noExtReloc, isOk
	case objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		// Low 26 bits = (S + A) >> 2
		t := ldr.SymValue(rs) + r.Add()
		o1 := target.Arch.ByteOrder.Uint32(ldr.OutData(s)[r.Off():])
		return int64(o1&0xfc000000 | uint32(t>>2)&^0xfc000000), noExtReloc, isOk
	}

	return val, false, false
}

func archrelocvariant(*ld.Target, *loader.Loader, loader.Reloc2, sym.RelocVariant, loader.Sym, int64) int64 {
	return -1
}
