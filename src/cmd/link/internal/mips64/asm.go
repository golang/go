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

func elfreloc1(ctxt *ld.Link, out *ld.OutBuf, ldr *loader.Loader, s loader.Sym, r loader.ExtReloc, ri int, sectoff int64) bool {

	// mips64 ELF relocation (endian neutral)
	//		offset	uint64
	//		sym		uint32
	//		ssym	uint8
	//		type3	uint8
	//		type2	uint8
	//		type	uint8
	//		addend	int64

	out.Write64(uint64(sectoff))

	elfsym := ld.ElfSymForReloc(ctxt, r.Xsym)
	out.Write32(uint32(elfsym))
	out.Write8(0)
	out.Write8(0)
	out.Write8(0)
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR, objabi.R_DWARFSECREF:
		switch r.Size {
		case 4:
			out.Write8(uint8(elf.R_MIPS_32))
		case 8:
			out.Write8(uint8(elf.R_MIPS_64))
		default:
			return false
		}
	case objabi.R_ADDRMIPS:
		out.Write8(uint8(elf.R_MIPS_LO16))
	case objabi.R_ADDRMIPSU:
		out.Write8(uint8(elf.R_MIPS_HI16))
	case objabi.R_ADDRMIPSTLS:
		out.Write8(uint8(elf.R_MIPS_TLS_TPREL_LO16))
	case objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		out.Write8(uint8(elf.R_MIPS_26))
	}
	out.Write64(uint64(r.Xadd))

	return true
}

func elfsetupplt(ctxt *ld.Link, plt, gotplt *loader.SymbolBuilder, dynamic loader.Sym) {
	return
}

func machoreloc1(*sys.Arch, *ld.OutBuf, *loader.Loader, loader.Sym, loader.ExtReloc, int64) bool {
	return false
}

func archreloc(target *ld.Target, ldr *loader.Loader, syms *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (o int64, nExtReloc int, ok bool) {
	if target.IsExternal() {
		switch r.Type() {
		default:
			return val, 0, false

		case objabi.R_ADDRMIPS,
			objabi.R_ADDRMIPSU,
			objabi.R_ADDRMIPSTLS,
			objabi.R_CALLMIPS,
			objabi.R_JMPMIPS:
			return val, 1, true
		}
	}

	const isOk = true
	const noExtReloc = 0
	rs := r.Sym()
	rs = ldr.ResolveABIAlias(rs)
	switch r.Type() {
	case objabi.R_ADDRMIPS,
		objabi.R_ADDRMIPSU:
		t := ldr.SymValue(rs) + r.Add()
		if r.Type() == objabi.R_ADDRMIPS {
			return int64(val&0xffff0000 | t&0xffff), noExtReloc, isOk
		}
		return int64(val&0xffff0000 | ((t+1<<15)>>16)&0xffff), noExtReloc, isOk
	case objabi.R_ADDRMIPSTLS:
		// thread pointer is at 0x7000 offset from the start of TLS data area
		t := ldr.SymValue(rs) + r.Add() - 0x7000
		if t < -32768 || t >= 32678 {
			ldr.Errorf(s, "TLS offset out of range %d", t)
		}
		return int64(val&0xffff0000 | t&0xffff), noExtReloc, isOk
	case objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		// Low 26 bits = (S + A) >> 2
		t := ldr.SymValue(rs) + r.Add()
		return int64(val&0xfc000000 | (t>>2)&^0xfc000000), noExtReloc, isOk
	}

	return val, 0, false
}

func archrelocvariant(*ld.Target, *loader.Loader, loader.Reloc, sym.RelocVariant, loader.Sym, int64) int64 {
	return -1
}

func extreloc(target *ld.Target, ldr *loader.Loader, r loader.Reloc, s loader.Sym) (loader.ExtReloc, bool) {
	switch r.Type() {
	case objabi.R_ADDRMIPS,
		objabi.R_ADDRMIPSU:
		return ld.ExtrelocViaOuterSym(ldr, r, s), true

	case objabi.R_ADDRMIPSTLS,
		objabi.R_CALLMIPS,
		objabi.R_JMPMIPS:
		return ld.ExtrelocSimple(ldr, r), true
	}
	return loader.ExtReloc{}, false
}
