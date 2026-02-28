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
//	Portions Copyright © 2016 The Go Authors. All rights reserved.
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

package mips

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
)

func gentext(ctxt *ld.Link, ldr *loader.Loader) {
	return
}

func elfreloc1(ctxt *ld.Link, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, ri int, sectoff int64) bool {
	out.Write32(uint32(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		if r.Size != 4 {
			return false
		}
		out.Write32(uint32(elf.R_MIPS_32) | uint32(elfsym)<<8)
	case objabi.R_ADDRMIPS:
		out.Write32(uint32(elf.R_MIPS_LO16) | uint32(elfsym)<<8)
	case objabi.R_ADDRMIPSU:
		out.Write32(uint32(elf.R_MIPS_HI16) | uint32(elfsym)<<8)
	case objabi.R_ADDRMIPSTLS:
		out.Write32(uint32(elf.R_MIPS_TLS_TPREL_LO16) | uint32(elfsym)<<8)
	case objabi.R_CALLMIPS, objabi.R_JMPMIPS:
		out.Write32(uint32(elf.R_MIPS_26) | uint32(elfsym)<<8)
	}

	return true
}

func elfsetupplt(ctxt *ld.Link, ldr *loader.Loader, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	return
}

func machoreloc1(*sys.Arch, *ld.OutBuf, *loader.Loader, loader.Sym, loader.ExtReloc, int64) bool {
	return false
}

func applyrel(arch *sys.Arch, ldr *loader.Loader, rt objabi.RelocType, off int32, s loader.Sym, val int64, t int64) int64 {
	o := uint32(val)
	switch rt {
	case objabi.R_ADDRMIPS, objabi.R_ADDRMIPSTLS:
		return int64(o&0xffff0000 | uint32(t)&0xffff)
	case objabi.R_ADDRMIPSU:
		return int64(o&0xffff0000 | uint32((t+(1<<15))>>16)&0xffff)
	case objabi.R_CALLMIPS, objabi.R_JMPMIPS:
		return int64(o&0xfc000000 | uint32(t>>2)&^0xfc000000)
	default:
		return val
	}
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (o int64, nExtReloc int, ok bool) {
	rs := r.Sym()
	if target.IsExternal() {
		switch r.Type() {
		default:
			return val, 0, false

		case objabi.R_ADDRMIPS, objabi.R_ADDRMIPSU:
			// set up addend for eventual relocation via outer symbol.
			_, off := ld.FoldSubSymbolOffset(ldr, rs)
			xadd := r.Add() + off
			return applyrel(target.Arch, ldr, r.Type(), r.Off(), s, val, xadd), 1, true

		case objabi.R_ADDRMIPSTLS, objabi.R_CALLMIPS, objabi.R_JMPMIPS:
			return applyrel(target.Arch, ldr, r.Type(), r.Off(), s, val, r.Add()), 1, true
		}
	}

	const isOk = true
	const noExtReloc = 0
	switch rt := r.Type(); rt {
	case objabi.R_ADDRMIPS, objabi.R_ADDRMIPSU:
		t := ldr.SymValue(rs) + r.Add()
		return applyrel(target.Arch, ldr, rt, r.Off(), s, val, t), noExtReloc, isOk
	case objabi.R_CALLMIPS, objabi.R_JMPMIPS:
		t := ldr.SymValue(rs) + r.Add()

		if t&3 != 0 {
			ldr.Errorf(s, "direct call is not aligned: %s %x", ldr.SymName(rs), t)
		}

		// check if target address is in the same 256 MB region as the next instruction
		if (ldr.SymValue(s)+int64(r.Off())+4)&0xf0000000 != (t & 0xf0000000) {
			ldr.Errorf(s, "direct call too far: %s %x", ldr.SymName(rs), t)
		}

		return applyrel(target.Arch, ldr, rt, r.Off(), s, val, t), noExtReloc, isOk
	case objabi.R_ADDRMIPSTLS:
		// thread pointer is at 0x7000 offset from the start of TLS data area
		t := ldr.SymValue(rs) + r.Add() - 0x7000
		if t < -32768 || t >= 32678 {
			ldr.Errorf(s, "TLS offset out of range %d", t)
		}
		return applyrel(target.Arch, ldr, rt, r.Off(), s, val, t), noExtReloc, isOk
	}

	return val, 0, false
}

func archrelocvariant(*ld.Target, *loader.Loader, loader.Reloc, sym.RelocVariant, loader.Sym, int64, []byte) int64 {
	return -1
}

func extreloc(target *ld.Target, ldr *loader.Loader, r loader.Reloc, s loader.Sym) (loader.ExtReloc, bool) {
	switch r.Type() {
	case objabi.R_ADDRMIPS, objabi.R_ADDRMIPSU:
		return ld.ExtrelocViaOuterSym(ldr, r, s), true

	case objabi.R_ADDRMIPSTLS, objabi.R_CALLMIPS, objabi.R_JMPMIPS:
		return ld.ExtrelocSimple(ldr, r), true
	}
	return loader.ExtReloc{}, false
}
