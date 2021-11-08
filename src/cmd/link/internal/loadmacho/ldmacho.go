// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loadmacho implements a Mach-O file reader.
package loadmacho

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"encoding/binary"
	"fmt"
)

/*
Derived from Plan 9 from User Space's src/libmach/elf.h, elf.c
https://github.com/9fans/plan9port/tree/master/src/libmach/

	Copyright © 2004 Russ Cox.
	Portions Copyright © 2008-2010 Google Inc.
	Portions Copyright © 2010 The Go Authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// TODO(crawshaw): de-duplicate these symbols with cmd/link/internal/ld
const (
	MACHO_X86_64_RELOC_UNSIGNED = 0
	MACHO_X86_64_RELOC_SIGNED   = 1
	MACHO_ARM64_RELOC_ADDEND    = 10
)

type ldMachoObj struct {
	f          *bio.Reader
	base       int64 // off in f where Mach-O begins
	length     int64 // length of Mach-O
	is64       bool
	name       string
	e          binary.ByteOrder
	cputype    uint
	subcputype uint
	filetype   uint32
	flags      uint32
	cmd        []ldMachoCmd
	ncmd       uint
}

type ldMachoCmd struct {
	type_ int
	off   uint32
	size  uint32
	seg   ldMachoSeg
	sym   ldMachoSymtab
	dsym  ldMachoDysymtab
}

type ldMachoSeg struct {
	name     string
	vmaddr   uint64
	vmsize   uint64
	fileoff  uint32
	filesz   uint32
	maxprot  uint32
	initprot uint32
	nsect    uint32
	flags    uint32
	sect     []ldMachoSect
}

type ldMachoSect struct {
	name    string
	segname string
	addr    uint64
	size    uint64
	off     uint32
	align   uint32
	reloff  uint32
	nreloc  uint32
	flags   uint32
	res1    uint32
	res2    uint32
	sym     loader.Sym
	rel     []ldMachoRel
}

type ldMachoRel struct {
	addr      uint32
	symnum    uint32
	pcrel     uint8
	length    uint8
	extrn     uint8
	type_     uint8
	scattered uint8
	value     uint32
}

type ldMachoSymtab struct {
	symoff  uint32
	nsym    uint32
	stroff  uint32
	strsize uint32
	str     []byte
	sym     []ldMachoSym
}

type ldMachoSym struct {
	name    string
	type_   uint8
	sectnum uint8
	desc    uint16
	kind    int8
	value   uint64
	sym     loader.Sym
}

type ldMachoDysymtab struct {
	ilocalsym      uint32
	nlocalsym      uint32
	iextdefsym     uint32
	nextdefsym     uint32
	iundefsym      uint32
	nundefsym      uint32
	tocoff         uint32
	ntoc           uint32
	modtaboff      uint32
	nmodtab        uint32
	extrefsymoff   uint32
	nextrefsyms    uint32
	indirectsymoff uint32
	nindirectsyms  uint32
	extreloff      uint32
	nextrel        uint32
	locreloff      uint32
	nlocrel        uint32
	indir          []uint32
}

// ldMachoSym.type_
const (
	N_EXT  = 0x01
	N_TYPE = 0x1e
	N_STAB = 0xe0
)

// ldMachoSym.desc
const (
	N_WEAK_REF = 0x40
	N_WEAK_DEF = 0x80
)

const (
	LdMachoCpuVax         = 1
	LdMachoCpu68000       = 6
	LdMachoCpu386         = 7
	LdMachoCpuAmd64       = 1<<24 | 7
	LdMachoCpuMips        = 8
	LdMachoCpu98000       = 10
	LdMachoCpuHppa        = 11
	LdMachoCpuArm         = 12
	LdMachoCpuArm64       = 1<<24 | 12
	LdMachoCpu88000       = 13
	LdMachoCpuSparc       = 14
	LdMachoCpu860         = 15
	LdMachoCpuAlpha       = 16
	LdMachoCpuPower       = 18
	LdMachoCmdSegment     = 1
	LdMachoCmdSymtab      = 2
	LdMachoCmdSymseg      = 3
	LdMachoCmdThread      = 4
	LdMachoCmdDysymtab    = 11
	LdMachoCmdSegment64   = 25
	LdMachoFileObject     = 1
	LdMachoFileExecutable = 2
	LdMachoFileFvmlib     = 3
	LdMachoFileCore       = 4
	LdMachoFilePreload    = 5
)

func unpackcmd(p []byte, m *ldMachoObj, c *ldMachoCmd, type_ uint, sz uint) int {
	e4 := m.e.Uint32
	e8 := m.e.Uint64

	c.type_ = int(type_)
	c.size = uint32(sz)
	switch type_ {
	default:
		return -1

	case LdMachoCmdSegment:
		if sz < 56 {
			return -1
		}
		c.seg.name = cstring(p[8:24])
		c.seg.vmaddr = uint64(e4(p[24:]))
		c.seg.vmsize = uint64(e4(p[28:]))
		c.seg.fileoff = e4(p[32:])
		c.seg.filesz = e4(p[36:])
		c.seg.maxprot = e4(p[40:])
		c.seg.initprot = e4(p[44:])
		c.seg.nsect = e4(p[48:])
		c.seg.flags = e4(p[52:])
		c.seg.sect = make([]ldMachoSect, c.seg.nsect)
		if uint32(sz) < 56+c.seg.nsect*68 {
			return -1
		}
		p = p[56:]
		var s *ldMachoSect
		for i := 0; uint32(i) < c.seg.nsect; i++ {
			s = &c.seg.sect[i]
			s.name = cstring(p[0:16])
			s.segname = cstring(p[16:32])
			s.addr = uint64(e4(p[32:]))
			s.size = uint64(e4(p[36:]))
			s.off = e4(p[40:])
			s.align = e4(p[44:])
			s.reloff = e4(p[48:])
			s.nreloc = e4(p[52:])
			s.flags = e4(p[56:])
			s.res1 = e4(p[60:])
			s.res2 = e4(p[64:])
			p = p[68:]
		}

	case LdMachoCmdSegment64:
		if sz < 72 {
			return -1
		}
		c.seg.name = cstring(p[8:24])
		c.seg.vmaddr = e8(p[24:])
		c.seg.vmsize = e8(p[32:])
		c.seg.fileoff = uint32(e8(p[40:]))
		c.seg.filesz = uint32(e8(p[48:]))
		c.seg.maxprot = e4(p[56:])
		c.seg.initprot = e4(p[60:])
		c.seg.nsect = e4(p[64:])
		c.seg.flags = e4(p[68:])
		c.seg.sect = make([]ldMachoSect, c.seg.nsect)
		if uint32(sz) < 72+c.seg.nsect*80 {
			return -1
		}
		p = p[72:]
		var s *ldMachoSect
		for i := 0; uint32(i) < c.seg.nsect; i++ {
			s = &c.seg.sect[i]
			s.name = cstring(p[0:16])
			s.segname = cstring(p[16:32])
			s.addr = e8(p[32:])
			s.size = e8(p[40:])
			s.off = e4(p[48:])
			s.align = e4(p[52:])
			s.reloff = e4(p[56:])
			s.nreloc = e4(p[60:])
			s.flags = e4(p[64:])
			s.res1 = e4(p[68:])
			s.res2 = e4(p[72:])

			// p+76 is reserved
			p = p[80:]
		}

	case LdMachoCmdSymtab:
		if sz < 24 {
			return -1
		}
		c.sym.symoff = e4(p[8:])
		c.sym.nsym = e4(p[12:])
		c.sym.stroff = e4(p[16:])
		c.sym.strsize = e4(p[20:])

	case LdMachoCmdDysymtab:
		if sz < 80 {
			return -1
		}
		c.dsym.ilocalsym = e4(p[8:])
		c.dsym.nlocalsym = e4(p[12:])
		c.dsym.iextdefsym = e4(p[16:])
		c.dsym.nextdefsym = e4(p[20:])
		c.dsym.iundefsym = e4(p[24:])
		c.dsym.nundefsym = e4(p[28:])
		c.dsym.tocoff = e4(p[32:])
		c.dsym.ntoc = e4(p[36:])
		c.dsym.modtaboff = e4(p[40:])
		c.dsym.nmodtab = e4(p[44:])
		c.dsym.extrefsymoff = e4(p[48:])
		c.dsym.nextrefsyms = e4(p[52:])
		c.dsym.indirectsymoff = e4(p[56:])
		c.dsym.nindirectsyms = e4(p[60:])
		c.dsym.extreloff = e4(p[64:])
		c.dsym.nextrel = e4(p[68:])
		c.dsym.locreloff = e4(p[72:])
		c.dsym.nlocrel = e4(p[76:])
	}

	return 0
}

func macholoadrel(m *ldMachoObj, sect *ldMachoSect) int {
	if sect.rel != nil || sect.nreloc == 0 {
		return 0
	}
	rel := make([]ldMachoRel, sect.nreloc)
	m.f.MustSeek(m.base+int64(sect.reloff), 0)
	buf, _, err := m.f.Slice(uint64(sect.nreloc * 8))
	if err != nil {
		return -1
	}
	for i := uint32(0); i < sect.nreloc; i++ {
		r := &rel[i]
		p := buf[i*8:]
		r.addr = m.e.Uint32(p)

		// TODO(rsc): Wrong interpretation for big-endian bitfields?
		if r.addr&0x80000000 != 0 {
			// scatterbrained relocation
			r.scattered = 1

			v := r.addr >> 24
			r.addr &= 0xFFFFFF
			r.type_ = uint8(v & 0xF)
			v >>= 4
			r.length = 1 << (v & 3)
			v >>= 2
			r.pcrel = uint8(v & 1)
			r.value = m.e.Uint32(p[4:])
		} else {
			v := m.e.Uint32(p[4:])
			r.symnum = v & 0xFFFFFF
			v >>= 24
			r.pcrel = uint8(v & 1)
			v >>= 1
			r.length = 1 << (v & 3)
			v >>= 2
			r.extrn = uint8(v & 1)
			v >>= 1
			r.type_ = uint8(v)
		}
	}

	sect.rel = rel
	return 0
}

func macholoaddsym(m *ldMachoObj, d *ldMachoDysymtab) int {
	n := int(d.nindirectsyms)
	m.f.MustSeek(m.base+int64(d.indirectsymoff), 0)
	p, _, err := m.f.Slice(uint64(n * 4))
	if err != nil {
		return -1
	}

	d.indir = make([]uint32, n)
	for i := 0; i < n; i++ {
		d.indir[i] = m.e.Uint32(p[4*i:])
	}
	return 0
}

func macholoadsym(m *ldMachoObj, symtab *ldMachoSymtab) int {
	if symtab.sym != nil {
		return 0
	}

	m.f.MustSeek(m.base+int64(symtab.stroff), 0)
	strbuf, _, err := m.f.Slice(uint64(symtab.strsize))
	if err != nil {
		return -1
	}

	symsize := 12
	if m.is64 {
		symsize = 16
	}
	n := int(symtab.nsym * uint32(symsize))
	m.f.MustSeek(m.base+int64(symtab.symoff), 0)
	symbuf, _, err := m.f.Slice(uint64(n))
	if err != nil {
		return -1
	}
	sym := make([]ldMachoSym, symtab.nsym)
	p := symbuf
	for i := uint32(0); i < symtab.nsym; i++ {
		s := &sym[i]
		v := m.e.Uint32(p)
		if v >= symtab.strsize {
			return -1
		}
		s.name = cstring(strbuf[v:])
		s.type_ = p[4]
		s.sectnum = p[5]
		s.desc = m.e.Uint16(p[6:])
		if m.is64 {
			s.value = m.e.Uint64(p[8:])
		} else {
			s.value = uint64(m.e.Uint32(p[8:]))
		}
		p = p[symsize:]
	}

	symtab.str = strbuf
	symtab.sym = sym
	return 0
}

// Load the Mach-O file pn from f.
// Symbols are written into syms, and a slice of the text symbols is returned.
func Load(l *loader.Loader, arch *sys.Arch, localSymVersion int, f *bio.Reader, pkg string, length int64, pn string) (textp []loader.Sym, err error) {
	errorf := func(str string, args ...interface{}) ([]loader.Sym, error) {
		return nil, fmt.Errorf("loadmacho: %v: %v", pn, fmt.Sprintf(str, args...))
	}

	base := f.Offset()

	hdr, _, err := f.Slice(7 * 4)
	if err != nil {
		return errorf("reading hdr: %v", err)
	}

	var e binary.ByteOrder
	if binary.BigEndian.Uint32(hdr[:])&^1 == 0xFEEDFACE {
		e = binary.BigEndian
	} else if binary.LittleEndian.Uint32(hdr[:])&^1 == 0xFEEDFACE {
		e = binary.LittleEndian
	} else {
		return errorf("bad magic - not mach-o file")
	}

	is64 := e.Uint32(hdr[:]) == 0xFEEDFACF
	ncmd := e.Uint32(hdr[4*4:])
	cmdsz := e.Uint32(hdr[5*4:])
	if ncmd > 0x10000 || cmdsz >= 0x01000000 {
		return errorf("implausible mach-o header ncmd=%d cmdsz=%d", ncmd, cmdsz)
	}

	if is64 {
		f.MustSeek(4, 1) // skip reserved word in header
	}

	m := &ldMachoObj{
		f:          f,
		e:          e,
		cputype:    uint(e.Uint32(hdr[1*4:])),
		subcputype: uint(e.Uint32(hdr[2*4:])),
		filetype:   e.Uint32(hdr[3*4:]),
		ncmd:       uint(ncmd),
		flags:      e.Uint32(hdr[6*4:]),
		is64:       is64,
		base:       base,
		length:     length,
		name:       pn,
	}

	switch arch.Family {
	default:
		return errorf("mach-o %s unimplemented", arch.Name)
	case sys.AMD64:
		if e != binary.LittleEndian || m.cputype != LdMachoCpuAmd64 {
			return errorf("mach-o object but not amd64")
		}
	case sys.ARM64:
		if e != binary.LittleEndian || m.cputype != LdMachoCpuArm64 {
			return errorf("mach-o object but not arm64")
		}
	}

	m.cmd = make([]ldMachoCmd, ncmd)
	cmdp, _, err := f.Slice(uint64(cmdsz))
	if err != nil {
		return errorf("reading cmds: %v", err)
	}

	// read and parse load commands
	var c *ldMachoCmd

	var symtab *ldMachoSymtab
	var dsymtab *ldMachoDysymtab

	off := uint32(len(hdr))
	for i := uint32(0); i < ncmd; i++ {
		ty := e.Uint32(cmdp)
		sz := e.Uint32(cmdp[4:])
		m.cmd[i].off = off
		unpackcmd(cmdp, m, &m.cmd[i], uint(ty), uint(sz))
		cmdp = cmdp[sz:]
		off += sz
		if ty == LdMachoCmdSymtab {
			if symtab != nil {
				return errorf("multiple symbol tables")
			}

			symtab = &m.cmd[i].sym
			macholoadsym(m, symtab)
		}

		if ty == LdMachoCmdDysymtab {
			dsymtab = &m.cmd[i].dsym
			macholoaddsym(m, dsymtab)
		}

		if (is64 && ty == LdMachoCmdSegment64) || (!is64 && ty == LdMachoCmdSegment) {
			if c != nil {
				return errorf("multiple load commands")
			}

			c = &m.cmd[i]
		}
	}

	// load text and data segments into memory.
	// they are not as small as the load commands, but we'll need
	// the memory anyway for the symbol images, so we might
	// as well use one large chunk.
	if c == nil {
		return errorf("no load command")
	}

	if symtab == nil {
		// our work is done here - no symbols means nothing can refer to this file
		return
	}

	if int64(c.seg.fileoff+c.seg.filesz) >= length {
		return errorf("load segment out of range")
	}

	f.MustSeek(m.base+int64(c.seg.fileoff), 0)
	dat, readOnly, err := f.Slice(uint64(c.seg.filesz))
	if err != nil {
		return errorf("cannot load object data: %v", err)
	}

	for i := uint32(0); i < c.seg.nsect; i++ {
		sect := &c.seg.sect[i]
		if sect.segname != "__TEXT" && sect.segname != "__DATA" {
			continue
		}
		if sect.name == "__eh_frame" {
			continue
		}
		name := fmt.Sprintf("%s(%s/%s)", pkg, sect.segname, sect.name)
		s := l.LookupOrCreateSym(name, localSymVersion)
		bld := l.MakeSymbolUpdater(s)
		if bld.Type() != 0 {
			return errorf("duplicate %s/%s", sect.segname, sect.name)
		}

		if sect.flags&0xff == 1 { // S_ZEROFILL
			bld.SetData(make([]byte, sect.size))
		} else {
			bld.SetReadOnly(readOnly)
			bld.SetData(dat[sect.addr-c.seg.vmaddr:][:sect.size])
		}
		bld.SetSize(int64(len(bld.Data())))

		if sect.segname == "__TEXT" {
			if sect.name == "__text" {
				bld.SetType(sym.STEXT)
			} else {
				bld.SetType(sym.SRODATA)
			}
		} else {
			if sect.name == "__bss" {
				bld.SetType(sym.SNOPTRBSS)
				bld.SetData(nil)
			} else {
				bld.SetType(sym.SNOPTRDATA)
			}
		}

		sect.sym = s
	}

	// enter sub-symbols into symbol table.
	// have to guess sizes from next symbol.
	for i := uint32(0); i < symtab.nsym; i++ {
		machsym := &symtab.sym[i]
		if machsym.type_&N_STAB != 0 {
			continue
		}

		// TODO: check sym->type against outer->type.
		name := machsym.name

		if name[0] == '_' && name[1] != '\x00' {
			name = name[1:]
		}
		v := 0
		if machsym.type_&N_EXT == 0 {
			v = localSymVersion
		}
		s := l.LookupOrCreateCgoExport(name, v)
		if machsym.type_&N_EXT == 0 {
			l.SetAttrDuplicateOK(s, true)
		}
		if machsym.desc&(N_WEAK_REF|N_WEAK_DEF) != 0 {
			l.SetAttrDuplicateOK(s, true)
		}
		machsym.sym = s
		if machsym.sectnum == 0 { // undefined
			continue
		}
		if uint32(machsym.sectnum) > c.seg.nsect {
			return errorf("reference to invalid section %d", machsym.sectnum)
		}

		sect := &c.seg.sect[machsym.sectnum-1]
		bld := l.MakeSymbolUpdater(s)
		outer := sect.sym
		if outer == 0 {
			continue // ignore reference to invalid section
		}

		if osym := l.OuterSym(s); osym != 0 {
			if l.AttrDuplicateOK(s) {
				continue
			}
			return errorf("duplicate symbol reference: %s in both %s and %s", l.SymName(s), l.SymName(osym), l.SymName(sect.sym))
		}

		bld.SetType(l.SymType(outer))
		if l.SymSize(outer) != 0 { // skip empty section (0-sized symbol)
			l.AddInteriorSym(outer, s)
		}

		bld.SetValue(int64(machsym.value - sect.addr))
		if !l.AttrCgoExportDynamic(s) {
			bld.SetDynimplib("") // satisfy dynimport
		}
		if l.SymType(outer) == sym.STEXT {
			if bld.External() && !bld.DuplicateOK() {
				return errorf("%v: duplicate symbol definition", s)
			}
			bld.SetExternal(true)
		}
	}

	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for i := 0; uint32(i) < c.seg.nsect; i++ {
		sect := &c.seg.sect[i]
		s := sect.sym
		if s == 0 {
			continue
		}
		bld := l.MakeSymbolUpdater(s)
		if bld.SubSym() != 0 {

			bld.SortSub()

			// assign sizes, now that we know symbols in sorted order.
			for s1 := bld.Sub(); s1 != 0; s1 = l.SubSym(s1) {
				s1Bld := l.MakeSymbolUpdater(s1)
				if sub := l.SubSym(s1); sub != 0 {
					s1Bld.SetSize(l.SymValue(sub) - l.SymValue(s1))
				} else {
					dlen := int64(len(l.Data(s)))
					s1Bld.SetSize(l.SymValue(s) + dlen - l.SymValue(s1))
				}
			}
		}

		if bld.Type() == sym.STEXT {
			if bld.OnList() {
				return errorf("symbol %s listed multiple times", bld.Name())
			}
			bld.SetOnList(true)
			textp = append(textp, s)
			for s1 := bld.Sub(); s1 != 0; s1 = l.SubSym(s1) {
				if l.AttrOnList(s1) {
					return errorf("symbol %s listed multiple times", l.RawSymName(s1))
				}
				l.SetAttrOnList(s1, true)
				textp = append(textp, s1)
			}
		}
	}

	// load relocations
	for i := 0; uint32(i) < c.seg.nsect; i++ {
		sect := &c.seg.sect[i]
		s := sect.sym
		if s == 0 {
			continue
		}
		macholoadrel(m, sect)
		if sect.rel == nil {
			continue
		}

		sb := l.MakeSymbolUpdater(sect.sym)
		var rAdd int64
		for j := uint32(0); j < sect.nreloc; j++ {
			var (
				rOff  int32
				rSize uint8
				rType objabi.RelocType
				rSym  loader.Sym
			)
			rel := &sect.rel[j]
			if rel.scattered != 0 {
				// mach-o only uses scattered relocation on 32-bit platforms,
				// which are no longer supported.
				return errorf("%v: unexpected scattered relocation", s)
			}

			if arch.Family == sys.ARM64 && rel.type_ == MACHO_ARM64_RELOC_ADDEND {
				// Two relocations. This addend will be applied to the next one.
				rAdd = int64(rel.symnum) << 40 >> 40 // convert unsigned 24-bit to signed 24-bit
				continue
			}

			rSize = rel.length
			rType = objabi.MachoRelocOffset + (objabi.RelocType(rel.type_) << 1) + objabi.RelocType(rel.pcrel)
			rOff = int32(rel.addr)

			// Handle X86_64_RELOC_SIGNED referencing a section (rel.extrn == 0).
			p := l.Data(s)
			if arch.Family == sys.AMD64 {
				if rel.extrn == 0 && rel.type_ == MACHO_X86_64_RELOC_SIGNED {
					// Calculate the addend as the offset into the section.
					//
					// The rip-relative offset stored in the object file is encoded
					// as follows:
					//
					//    movsd	0x00000360(%rip),%xmm0
					//
					// To get the absolute address of the value this rip-relative address is pointing
					// to, we must add the address of the next instruction to it. This is done by
					// taking the address of the relocation and adding 4 to it (since the rip-relative
					// offset can at most be 32 bits long).  To calculate the offset into the section the
					// relocation is referencing, we subtract the vaddr of the start of the referenced
					// section found in the original object file.
					//
					// [For future reference, see Darwin's /usr/include/mach-o/x86_64/reloc.h]
					secaddr := c.seg.sect[rel.symnum-1].addr
					rAdd = int64(uint64(int64(int32(e.Uint32(p[rOff:])))+int64(rOff)+4) - secaddr)
				} else {
					rAdd = int64(int32(e.Uint32(p[rOff:])))
				}
			}

			// An unsigned internal relocation has a value offset
			// by the section address.
			if arch.Family == sys.AMD64 && rel.extrn == 0 && rel.type_ == MACHO_X86_64_RELOC_UNSIGNED {
				secaddr := c.seg.sect[rel.symnum-1].addr
				rAdd -= int64(secaddr)
			}

			if rel.extrn == 0 {
				if rel.symnum < 1 || rel.symnum > c.seg.nsect {
					return errorf("invalid relocation: section reference out of range %d vs %d", rel.symnum, c.seg.nsect)
				}

				rSym = c.seg.sect[rel.symnum-1].sym
				if rSym == 0 {
					return errorf("invalid relocation: %s", c.seg.sect[rel.symnum-1].name)
				}
			} else {
				if rel.symnum >= symtab.nsym {
					return errorf("invalid relocation: symbol reference out of range")
				}

				rSym = symtab.sym[rel.symnum].sym
			}

			r, _ := sb.AddRel(rType)
			r.SetOff(rOff)
			r.SetSiz(rSize)
			r.SetSym(rSym)
			r.SetAdd(rAdd)

			rAdd = 0 // clear rAdd for next iteration
		}

		sb.SortRelocs()
	}

	return textp, nil
}

func cstring(x []byte) string {
	i := bytes.IndexByte(x, '\x00')
	if i >= 0 {
		x = x[:i]
	}
	return string(x)
}
