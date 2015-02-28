package ld

import (
	"bytes"
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"log"
	"sort"
	"strings"
)

/*
Derived from Plan 9 from User Space's src/libmach/elf.h, elf.c
http://code.swtch.com/plan9port/src/tip/src/libmach/

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
const (
	ElfClassNone = 0 + iota
	ElfClass32
	ElfClass64
	ElfDataNone = 0 + iota - 3
	ElfDataLsb
	ElfDataMsb
	ElfTypeNone = 0 + iota - 6
	ElfTypeRelocatable
	ElfTypeExecutable
	ElfTypeSharedObject
	ElfTypeCore
	ElfMachNone = 0 + iota - 11
	ElfMach32100
	ElfMachSparc
	ElfMach386
	ElfMach68000
	ElfMach88000
	ElfMach486
	ElfMach860
	ElfMachMips
	ElfMachS370
	ElfMachMipsLe
	ElfMachParisc = 15
	ElfMachVpp500 = 17 + iota - 23
	ElfMachSparc32Plus
	ElfMach960
	ElfMachPower
	ElfMachPower64
	ElfMachS390
	ElfMachV800 = 36 + iota - 29
	ElfMachFr20
	ElfMachRh32
	ElfMachRce
	ElfMachArm
	ElfMachAlpha
	ElfMachSH
	ElfMachSparc9
	ElfMachAmd64  = 62
	ElfAbiNone    = 0
	ElfAbiSystemV = 0 + iota - 39
	ElfAbiHPUX
	ElfAbiNetBSD
	ElfAbiLinux
	ElfAbiSolaris = 6 + iota - 43
	ElfAbiAix
	ElfAbiIrix
	ElfAbiFreeBSD
	ElfAbiTru64
	ElfAbiModesto
	ElfAbiOpenBSD
	ElfAbiARM      = 97
	ElfAbiEmbedded = 255
	ElfSectNone    = 0 + iota - 52
	ElfSectProgbits
	ElfSectSymtab
	ElfSectStrtab
	ElfSectRela
	ElfSectHash
	ElfSectDynamic
	ElfSectNote
	ElfSectNobits
	ElfSectRel
	ElfSectShlib
	ElfSectDynsym
	ElfSectFlagWrite = 0x1
	ElfSectFlagAlloc = 0x2
	ElfSectFlagExec  = 0x4
	ElfSymBindLocal  = 0 + iota - 67
	ElfSymBindGlobal
	ElfSymBindWeak
	ElfSymTypeNone = 0 + iota - 70
	ElfSymTypeObject
	ElfSymTypeFunc
	ElfSymTypeSection
	ElfSymTypeFile
	ElfSymShnNone   = 0
	ElfSymShnAbs    = 0xFFF1
	ElfSymShnCommon = 0xFFF2
	ElfProgNone     = 0 + iota - 78
	ElfProgLoad
	ElfProgDynamic
	ElfProgInterp
	ElfProgNote
	ElfProgShlib
	ElfProgPhdr
	ElfProgFlagExec     = 0x1
	ElfProgFlagWrite    = 0x2
	ElfProgFlagRead     = 0x4
	ElfNotePrStatus     = 1
	ElfNotePrFpreg      = 2
	ElfNotePrPsinfo     = 3
	ElfNotePrTaskstruct = 4
	ElfNotePrAuxv       = 6
	ElfNotePrXfpreg     = 0x46e62b7f
)

type ElfHdrBytes struct {
	Ident     [16]uint8
	Type      [2]uint8
	Machine   [2]uint8
	Version   [4]uint8
	Entry     [4]uint8
	Phoff     [4]uint8
	Shoff     [4]uint8
	Flags     [4]uint8
	Ehsize    [2]uint8
	Phentsize [2]uint8
	Phnum     [2]uint8
	Shentsize [2]uint8
	Shnum     [2]uint8
	Shstrndx  [2]uint8
}

type ElfSectBytes struct {
	Name    [4]uint8
	Type    [4]uint8
	Flags   [4]uint8
	Addr    [4]uint8
	Off     [4]uint8
	Size    [4]uint8
	Link    [4]uint8
	Info    [4]uint8
	Align   [4]uint8
	Entsize [4]uint8
}

type ElfProgBytes struct {
}

type ElfSymBytes struct {
	Name  [4]uint8
	Value [4]uint8
	Size  [4]uint8
	Info  uint8
	Other uint8
	Shndx [2]uint8
}

type ElfHdrBytes64 struct {
	Ident     [16]uint8
	Type      [2]uint8
	Machine   [2]uint8
	Version   [4]uint8
	Entry     [8]uint8
	Phoff     [8]uint8
	Shoff     [8]uint8
	Flags     [4]uint8
	Ehsize    [2]uint8
	Phentsize [2]uint8
	Phnum     [2]uint8
	Shentsize [2]uint8
	Shnum     [2]uint8
	Shstrndx  [2]uint8
}

type ElfSectBytes64 struct {
	Name    [4]uint8
	Type    [4]uint8
	Flags   [8]uint8
	Addr    [8]uint8
	Off     [8]uint8
	Size    [8]uint8
	Link    [4]uint8
	Info    [4]uint8
	Align   [8]uint8
	Entsize [8]uint8
}

type ElfProgBytes64 struct {
}

type ElfSymBytes64 struct {
	Name  [4]uint8
	Info  uint8
	Other uint8
	Shndx [2]uint8
	Value [8]uint8
	Size  [8]uint8
}

type ElfSect struct {
	name    string
	nameoff uint32
	type_   uint32
	flags   uint64
	addr    uint64
	off     uint64
	size    uint64
	link    uint32
	info    uint32
	align   uint64
	entsize uint64
	base    []byte
	sym     *LSym
}

type ElfObj struct {
	f         *Biobuf
	base      int64
	length    int64
	is64      int
	name      string
	e         binary.ByteOrder
	sect      []ElfSect
	nsect     uint
	shstrtab  string
	nsymtab   int
	symtab    *ElfSect
	symstr    *ElfSect
	type_     uint32
	machine   uint32
	version   uint32
	entry     uint64
	phoff     uint64
	shoff     uint64
	flags     uint32
	ehsize    uint32
	phentsize uint32
	phnum     uint32
	shentsize uint32
	shnum     uint32
	shstrndx  uint32
}

type ElfSym struct {
	name  string
	value uint64
	size  uint64
	bind  uint8
	type_ uint8
	other uint8
	shndx uint16
	sym   *LSym
}

var ElfMagic = [4]uint8{0x7F, 'E', 'L', 'F'}

func valuecmp(a *LSym, b *LSym) int {
	if a.Value < b.Value {
		return -1
	}
	if a.Value > b.Value {
		return +1
	}
	return 0
}

func ldelf(f *Biobuf, pkg string, length int64, pn string) {
	var err error
	var base int32
	var add uint64
	var info uint64
	var name string
	var i int
	var j int
	var rela int
	var is64 int
	var n int
	var flag int
	var hdrbuf [64]uint8
	var p []byte
	var hdr *ElfHdrBytes
	var elfobj *ElfObj
	var sect *ElfSect
	var rsect *ElfSect
	var sym ElfSym
	var e binary.ByteOrder
	var r []Reloc
	var rp *Reloc
	var s *LSym
	var symbols []*LSym

	symbols = nil

	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f ldelf %s\n", obj.Cputime(), pn)
	}

	Ctxt.Version++
	base = int32(Boffset(f))

	if Bread(f, hdrbuf[:]) != len(hdrbuf) {
		goto bad
	}
	hdr = new(ElfHdrBytes)
	binary.Read(bytes.NewReader(hdrbuf[:]), binary.BigEndian, hdr) // only byte arrays; byte order doesn't matter
	if string(hdr.Ident[:4]) != "\x7FELF" {
		goto bad
	}
	switch hdr.Ident[5] {
	case ElfDataLsb:
		e = binary.LittleEndian

	case ElfDataMsb:
		e = binary.BigEndian

	default:
		goto bad
	}

	// read header
	elfobj = new(ElfObj)

	elfobj.e = e
	elfobj.f = f
	elfobj.base = int64(base)
	elfobj.length = length
	elfobj.name = pn

	is64 = 0
	if hdr.Ident[4] == ElfClass64 {
		var hdr *ElfHdrBytes64

		is64 = 1
		hdr = new(ElfHdrBytes64)
		binary.Read(bytes.NewReader(hdrbuf[:]), binary.BigEndian, hdr) // only byte arrays; byte order doesn't matter
		elfobj.type_ = uint32(e.Uint16(hdr.Type[:]))
		elfobj.machine = uint32(e.Uint16(hdr.Machine[:]))
		elfobj.version = e.Uint32(hdr.Version[:])
		elfobj.phoff = e.Uint64(hdr.Phoff[:])
		elfobj.shoff = e.Uint64(hdr.Shoff[:])
		elfobj.flags = e.Uint32(hdr.Flags[:])
		elfobj.ehsize = uint32(e.Uint16(hdr.Ehsize[:]))
		elfobj.phentsize = uint32(e.Uint16(hdr.Phentsize[:]))
		elfobj.phnum = uint32(e.Uint16(hdr.Phnum[:]))
		elfobj.shentsize = uint32(e.Uint16(hdr.Shentsize[:]))
		elfobj.shnum = uint32(e.Uint16(hdr.Shnum[:]))
		elfobj.shstrndx = uint32(e.Uint16(hdr.Shstrndx[:]))
	} else {
		elfobj.type_ = uint32(e.Uint16(hdr.Type[:]))
		elfobj.machine = uint32(e.Uint16(hdr.Machine[:]))
		elfobj.version = e.Uint32(hdr.Version[:])
		elfobj.entry = uint64(e.Uint32(hdr.Entry[:]))
		elfobj.phoff = uint64(e.Uint32(hdr.Phoff[:]))
		elfobj.shoff = uint64(e.Uint32(hdr.Shoff[:]))
		elfobj.flags = e.Uint32(hdr.Flags[:])
		elfobj.ehsize = uint32(e.Uint16(hdr.Ehsize[:]))
		elfobj.phentsize = uint32(e.Uint16(hdr.Phentsize[:]))
		elfobj.phnum = uint32(e.Uint16(hdr.Phnum[:]))
		elfobj.shentsize = uint32(e.Uint16(hdr.Shentsize[:]))
		elfobj.shnum = uint32(e.Uint16(hdr.Shnum[:]))
		elfobj.shstrndx = uint32(e.Uint16(hdr.Shstrndx[:]))
	}

	elfobj.is64 = is64

	if uint32(hdr.Ident[6]) != elfobj.version {
		goto bad
	}

	if e.Uint16(hdr.Type[:]) != ElfTypeRelocatable {
		Diag("%s: elf but not elf relocatable object", pn)
		return
	}

	switch Thearch.Thechar {
	default:
		Diag("%s: elf %s unimplemented", pn, Thestring)
		return

	case '5':
		if e != binary.LittleEndian || elfobj.machine != ElfMachArm || hdr.Ident[4] != ElfClass32 {
			Diag("%s: elf object but not arm", pn)
			return
		}

	case '6':
		if e != binary.LittleEndian || elfobj.machine != ElfMachAmd64 || hdr.Ident[4] != ElfClass64 {
			Diag("%s: elf object but not amd64", pn)
			return
		}

	case '8':
		if e != binary.LittleEndian || elfobj.machine != ElfMach386 || hdr.Ident[4] != ElfClass32 {
			Diag("%s: elf object but not 386", pn)
			return
		}

	case '9':
		if elfobj.machine != ElfMachPower64 || hdr.Ident[4] != ElfClass64 {
			Diag("%s: elf object but not ppc64", pn)
			return
		}
	}

	// load section list into memory.
	elfobj.sect = make([]ElfSect, elfobj.shnum)

	elfobj.nsect = uint(elfobj.shnum)
	for i = 0; uint(i) < elfobj.nsect; i++ {
		if Bseek(f, int64(uint64(base)+elfobj.shoff+uint64(int64(i)*int64(elfobj.shentsize))), 0) < 0 {
			goto bad
		}
		sect = &elfobj.sect[i]
		if is64 != 0 {
			var b ElfSectBytes64

			if err = binary.Read(f, e, &b); err != nil {
				goto bad
			}

			sect.nameoff = uint32(e.Uint32(b.Name[:]))
			sect.type_ = e.Uint32(b.Type[:])
			sect.flags = e.Uint64(b.Flags[:])
			sect.addr = e.Uint64(b.Addr[:])
			sect.off = e.Uint64(b.Off[:])
			sect.size = e.Uint64(b.Size[:])
			sect.link = e.Uint32(b.Link[:])
			sect.info = e.Uint32(b.Info[:])
			sect.align = e.Uint64(b.Align[:])
			sect.entsize = e.Uint64(b.Entsize[:])
		} else {
			var b ElfSectBytes

			if err = binary.Read(f, e, &b); err != nil {
				goto bad
			}

			sect.nameoff = uint32(e.Uint32(b.Name[:]))
			sect.type_ = e.Uint32(b.Type[:])
			sect.flags = uint64(e.Uint32(b.Flags[:]))
			sect.addr = uint64(e.Uint32(b.Addr[:]))
			sect.off = uint64(e.Uint32(b.Off[:]))
			sect.size = uint64(e.Uint32(b.Size[:]))
			sect.link = e.Uint32(b.Link[:])
			sect.info = e.Uint32(b.Info[:])
			sect.align = uint64(e.Uint32(b.Align[:]))
			sect.entsize = uint64(e.Uint32(b.Entsize[:]))
		}
	}

	// read section string table and translate names
	if elfobj.shstrndx >= uint32(elfobj.nsect) {
		err = fmt.Errorf("shstrndx out of range %d >= %d", elfobj.shstrndx, elfobj.nsect)
		goto bad
	}

	sect = &elfobj.sect[elfobj.shstrndx]
	if err = elfmap(elfobj, sect); err != nil {
		goto bad
	}
	for i = 0; uint(i) < elfobj.nsect; i++ {
		if elfobj.sect[i].nameoff != 0 {
			elfobj.sect[i].name = cstring(sect.base[elfobj.sect[i].nameoff:])
		}
	}

	// load string table for symbols into memory.
	elfobj.symtab = section(elfobj, ".symtab")

	if elfobj.symtab == nil {
		// our work is done here - no symbols means nothing can refer to this file
		return
	}

	if elfobj.symtab.link <= 0 || elfobj.symtab.link >= uint32(elfobj.nsect) {
		Diag("%s: elf object has symbol table with invalid string table link", pn)
		return
	}

	elfobj.symstr = &elfobj.sect[elfobj.symtab.link]
	if is64 != 0 {
		elfobj.nsymtab = int(elfobj.symtab.size / ELF64SYMSIZE)
	} else {
		elfobj.nsymtab = int(elfobj.symtab.size / ELF32SYMSIZE)
	}

	if err = elfmap(elfobj, elfobj.symtab); err != nil {
		goto bad
	}
	if err = elfmap(elfobj, elfobj.symstr); err != nil {
		goto bad
	}

	// load text and data segments into memory.
	// they are not as small as the section lists, but we'll need
	// the memory anyway for the symbol images, so we might
	// as well use one large chunk.

	// create symbols for elfmapped sections
	for i = 0; uint(i) < elfobj.nsect; i++ {
		sect = &elfobj.sect[i]
		if (sect.type_ != ElfSectProgbits && sect.type_ != ElfSectNobits) || sect.flags&ElfSectFlagAlloc == 0 {
			continue
		}
		if sect.type_ != ElfSectNobits {
			if err = elfmap(elfobj, sect); err != nil {
				goto bad
			}
		}

		name = fmt.Sprintf("%s(%s)", pkg, sect.name)
		s = Linklookup(Ctxt, name, Ctxt.Version)

		switch int(sect.flags) & (ElfSectFlagAlloc | ElfSectFlagWrite | ElfSectFlagExec) {
		default:
			err = fmt.Errorf("unexpected flags for ELF section %s", sect.name)
			goto bad

		case ElfSectFlagAlloc:
			s.Type = SRODATA

		case ElfSectFlagAlloc + ElfSectFlagWrite:
			if sect.type_ == ElfSectNobits {
				s.Type = SNOPTRBSS
			} else {
				s.Type = SNOPTRDATA
			}

		case ElfSectFlagAlloc + ElfSectFlagExec:
			s.Type = STEXT
		}

		if sect.name == ".got" || sect.name == ".toc" {
			s.Type = SELFGOT
		}
		if sect.type_ == ElfSectProgbits {
			s.P = sect.base
			s.P = s.P[:sect.size]
		}

		s.Size = int64(sect.size)
		s.Align = int32(sect.align)
		sect.sym = s
	}

	// enter sub-symbols into symbol table.
	// symbol 0 is the null symbol.
	symbols = make([]*LSym, elfobj.nsymtab)

	if symbols == nil {
		Diag("out of memory")
		Errorexit()
	}

	for i = 1; i < elfobj.nsymtab; i++ {
		if err = readelfsym(elfobj, i, &sym, 1); err != nil {
			goto bad
		}
		symbols[i] = sym.sym
		if sym.type_ != ElfSymTypeFunc && sym.type_ != ElfSymTypeObject && sym.type_ != ElfSymTypeNone {
			continue
		}
		if sym.shndx == ElfSymShnCommon {
			s = sym.sym
			if uint64(s.Size) < sym.size {
				s.Size = int64(sym.size)
			}
			if s.Type == 0 || s.Type == SXREF {
				s.Type = SNOPTRBSS
			}
			continue
		}

		if uint(sym.shndx) >= elfobj.nsect || sym.shndx == 0 {
			continue
		}

		// even when we pass needSym == 1 to readelfsym, it might still return nil to skip some unwanted symbols
		if sym.sym == nil {
			continue
		}
		sect = &elfobj.sect[sym.shndx:][0]
		if sect.sym == nil {
			if strings.HasPrefix(sym.name, ".Linfo_string") {
				continue
			}
			Diag("%s: sym#%d: ignoring %s in section %d (type %d)", pn, i, sym.name, sym.shndx, sym.type_)
			continue
		}

		s = sym.sym
		if s.Outer != nil {
			if s.Dupok != 0 {
				continue
			}
			Diag("%s: duplicate symbol reference: %s in both %s and %s", pn, s.Name, s.Outer.Name, sect.sym.Name)
			Errorexit()
		}

		s.Sub = sect.sym.Sub
		sect.sym.Sub = s
		s.Type = sect.sym.Type | s.Type&^SMASK | SSUB
		if s.Cgoexport&CgoExportDynamic == 0 {
			s.Dynimplib = "" // satisfy dynimport
		}
		s.Value = int64(sym.value)
		s.Size = int64(sym.size)
		s.Outer = sect.sym
		if sect.sym.Type == STEXT {
			if s.External != 0 && s.Dupok == 0 {
				Diag("%s: duplicate definition of %s", pn, s.Name)
			}
			s.External = 1
		}

		if elfobj.machine == ElfMachPower64 {
			flag = int(sym.other) >> 5
			if 2 <= flag && flag <= 6 {
				s.Localentry = 1 << uint(flag-2)
			} else if flag == 7 {
				Diag("%s: invalid sym.other 0x%x for %s", pn, sym.other, s.Name)
			}
		}
	}

	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for i = 0; uint(i) < elfobj.nsect; i++ {
		s = elfobj.sect[i].sym
		if s == nil {
			continue
		}
		if s.Sub != nil {
			s.Sub = listsort(s.Sub, valuecmp, listsubp)
		}
		if s.Type == STEXT {
			if s.Onlist != 0 {
				log.Fatalf("symbol %s listed multiple times", s.Name)
			}
			s.Onlist = 1
			if Ctxt.Etextp != nil {
				Ctxt.Etextp.Next = s
			} else {
				Ctxt.Textp = s
			}
			Ctxt.Etextp = s
			for s = s.Sub; s != nil; s = s.Sub {
				if s.Onlist != 0 {
					log.Fatalf("symbol %s listed multiple times", s.Name)
				}
				s.Onlist = 1
				Ctxt.Etextp.Next = s
				Ctxt.Etextp = s
			}
		}
	}

	// load relocations
	for i = 0; uint(i) < elfobj.nsect; i++ {
		rsect = &elfobj.sect[i]
		if rsect.type_ != ElfSectRela && rsect.type_ != ElfSectRel {
			continue
		}
		if rsect.info >= uint32(elfobj.nsect) || elfobj.sect[rsect.info].base == nil {
			continue
		}
		sect = &elfobj.sect[rsect.info]
		if err = elfmap(elfobj, rsect); err != nil {
			goto bad
		}
		rela = 0
		if rsect.type_ == ElfSectRela {
			rela = 1
		}
		n = int(rsect.size / uint64(4+4*is64) / uint64(2+rela))
		r = make([]Reloc, n)
		p = rsect.base
		for j = 0; j < n; j++ {
			add = 0
			rp = &r[j]
			if is64 != 0 {
				// 64-bit rel/rela
				rp.Off = int32(e.Uint64(p))

				p = p[8:]
				info = e.Uint64(p)
				p = p[8:]
				if rela != 0 {
					add = e.Uint64(p)
					p = p[8:]
				}
			} else {
				// 32-bit rel/rela
				rp.Off = int32(e.Uint32(p))

				p = p[4:]
				info = uint64(e.Uint32(p))
				info = info>>8<<32 | info&0xff // convert to 64-bit info
				p = p[4:]
				if rela != 0 {
					add = uint64(e.Uint32(p))
					p = p[4:]
				}
			}

			if info&0xffffffff == 0 { // skip R_*_NONE relocation
				j--
				n--
				continue
			}

			if info>>32 == 0 { // absolute relocation, don't bother reading the null symbol
				rp.Sym = nil
			} else {
				if err = readelfsym(elfobj, int(info>>32), &sym, 0); err != nil {
					goto bad
				}
				sym.sym = symbols[info>>32]
				if sym.sym == nil {
					err = fmt.Errorf("%s#%d: reloc of invalid sym #%d %s shndx=%d type=%d", sect.sym.Name, j, int(info>>32), sym.name, sym.shndx, sym.type_)
					goto bad
				}

				rp.Sym = sym.sym
			}

			rp.Type = int32(reltype(pn, int(uint32(info)), &rp.Siz))
			if rela != 0 {
				rp.Add = int64(add)
			} else {
				// load addend from image
				if rp.Siz == 4 {
					rp.Add = int64(e.Uint32(sect.base[rp.Off:]))
				} else if rp.Siz == 8 {
					rp.Add = int64(e.Uint64(sect.base[rp.Off:]))
				} else {
					Diag("invalid rela size %d", rp.Siz)
				}
			}

			if rp.Siz == 2 {
				rp.Add = int64(int16(rp.Add))
			}
			if rp.Siz == 4 {
				rp.Add = int64(int32(rp.Add))
			}
		}

		//print("rel %s %d %d %s %#llx\n", sect->sym->name, rp->type, rp->siz, rp->sym->name, rp->add);
		sort.Sort(rbyoff(r[:n]))
		// just in case

		s = sect.sym
		s.R = r
		s.R = s.R[:n]
	}

	return

bad:
	Diag("%s: malformed elf file: %v", pn, err)
}

func section(elfobj *ElfObj, name string) *ElfSect {
	var i int

	for i = 0; uint(i) < elfobj.nsect; i++ {
		if elfobj.sect[i].name != "" && name != "" && elfobj.sect[i].name == name {
			return &elfobj.sect[i]
		}
	}
	return nil
}

func elfmap(elfobj *ElfObj, sect *ElfSect) (err error) {
	if sect.base != nil {
		return nil
	}

	if sect.off+sect.size > uint64(elfobj.length) {
		err = fmt.Errorf("elf section past end of file")
		return err
	}

	sect.base = make([]byte, sect.size)
	err = fmt.Errorf("short read")
	if Bseek(elfobj.f, int64(uint64(elfobj.base)+sect.off), 0) < 0 || Bread(elfobj.f, sect.base) != len(sect.base) {
		return err
	}

	return nil
}

func readelfsym(elfobj *ElfObj, i int, sym *ElfSym, needSym int) (err error) {
	var s *LSym

	if i >= elfobj.nsymtab || i < 0 {
		err = fmt.Errorf("invalid elf symbol index")
		return err
	}

	if i == 0 {
		Diag("readym: read null symbol!")
	}

	if elfobj.is64 != 0 {
		b := new(ElfSymBytes64)
		binary.Read(bytes.NewReader(elfobj.symtab.base[i*ELF64SYMSIZE:(i+1)*ELF64SYMSIZE]), elfobj.e, b)
		sym.name = cstring(elfobj.symstr.base[elfobj.e.Uint32(b.Name[:]):])
		sym.value = elfobj.e.Uint64(b.Value[:])
		sym.size = elfobj.e.Uint64(b.Size[:])
		sym.shndx = elfobj.e.Uint16(b.Shndx[:])
		sym.bind = b.Info >> 4
		sym.type_ = b.Info & 0xf
		sym.other = b.Other
	} else {
		b := new(ElfSymBytes)
		binary.Read(bytes.NewReader(elfobj.symtab.base[i*ELF32SYMSIZE:(i+1)*ELF32SYMSIZE]), elfobj.e, b)
		sym.name = cstring(elfobj.symstr.base[elfobj.e.Uint32(b.Name[:]):])
		sym.value = uint64(elfobj.e.Uint32(b.Value[:]))
		sym.size = uint64(elfobj.e.Uint32(b.Size[:]))
		sym.shndx = elfobj.e.Uint16(b.Shndx[:])
		sym.bind = b.Info >> 4
		sym.type_ = b.Info & 0xf
		sym.other = b.Other
	}

	s = nil
	if sym.name == "_GLOBAL_OFFSET_TABLE_" {
		sym.name = ".got"
	}
	if sym.name == ".TOC." {
		// Magic symbol on ppc64.  Will be set to this object
		// file's .got+0x8000.
		sym.bind = ElfSymBindLocal
	}

	switch sym.type_ {
	case ElfSymTypeSection:
		s = elfobj.sect[sym.shndx].sym

	case ElfSymTypeObject,
		ElfSymTypeFunc,
		ElfSymTypeNone:
		switch sym.bind {
		case ElfSymBindGlobal:
			if needSym != 0 {
				s = Linklookup(Ctxt, sym.name, 0)

				// for global scoped hidden symbols we should insert it into
				// symbol hash table, but mark them as hidden.
				// __i686.get_pc_thunk.bx is allowed to be duplicated, to
				// workaround that we set dupok.
				// TODO(minux): correctly handle __i686.get_pc_thunk.bx without
				// set dupok generally. See http://codereview.appspot.com/5823055/
				// comment #5 for details.
				if s != nil && sym.other == 2 {
					s.Type |= SHIDDEN
					s.Dupok = 1
				}
			}

		case ElfSymBindLocal:
			if Thearch.Thechar == '5' && (strings.HasPrefix(sym.name, "$a") || strings.HasPrefix(sym.name, "$d")) {
				// binutils for arm generate these elfmapping
				// symbols, ignore these
				break
			}

			if sym.name == ".TOC." {
				// We need to be able to look this up,
				// so put it in the hash table.
				if needSym != 0 {
					s = Linklookup(Ctxt, sym.name, Ctxt.Version)
					s.Type |= SHIDDEN
				}

				break
			}

			if needSym != 0 {
				// local names and hidden visiblity global names are unique
				// and should only reference by its index, not name, so we
				// don't bother to add them into hash table
				s = linknewsym(Ctxt, sym.name, Ctxt.Version)

				s.Type |= SHIDDEN
			}

		case ElfSymBindWeak:
			if needSym != 0 {
				s = linknewsym(Ctxt, sym.name, 0)
				if sym.other == 2 {
					s.Type |= SHIDDEN
				}
			}

		default:
			err = fmt.Errorf("%s: invalid symbol binding %d", sym.name, sym.bind)
			return err
		}
	}

	if s != nil && s.Type == 0 && sym.type_ != ElfSymTypeSection {
		s.Type = SXREF
	}
	sym.sym = s

	return nil
}

type rbyoff []Reloc

func (x rbyoff) Len() int {
	return len(x)
}

func (x rbyoff) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}

func (x rbyoff) Less(i, j int) bool {
	var a *Reloc
	var b *Reloc

	a = &x[i]
	b = &x[j]
	if a.Off < b.Off {
		return true
	}
	if a.Off > b.Off {
		return false
	}
	return false
}

func reltype(pn string, elftype int, siz *uint8) int {
	switch uint32(Thearch.Thechar) | uint32(elftype)<<24 {
	default:
		Diag("%s: unknown relocation type %d; compiled without -fpic?", pn, elftype)
		fallthrough

	case '9' | R_PPC64_TOC16<<24,
		'9' | R_PPC64_TOC16_LO<<24,
		'9' | R_PPC64_TOC16_HI<<24,
		'9' | R_PPC64_TOC16_HA<<24,
		'9' | R_PPC64_TOC16_DS<<24,
		'9' | R_PPC64_TOC16_LO_DS<<24,
		'9' | R_PPC64_REL16_LO<<24,
		'9' | R_PPC64_REL16_HI<<24,
		'9' | R_PPC64_REL16_HA<<24:
		*siz = 2

	case '5' | R_ARM_ABS32<<24,
		'5' | R_ARM_GOT32<<24,
		'5' | R_ARM_PLT32<<24,
		'5' | R_ARM_GOTOFF<<24,
		'5' | R_ARM_GOTPC<<24,
		'5' | R_ARM_THM_PC22<<24,
		'5' | R_ARM_REL32<<24,
		'5' | R_ARM_CALL<<24,
		'5' | R_ARM_V4BX<<24,
		'5' | R_ARM_GOT_PREL<<24,
		'5' | R_ARM_PC24<<24,
		'5' | R_ARM_JUMP24<<24,
		'6' | R_X86_64_PC32<<24,
		'6' | R_X86_64_PLT32<<24,
		'6' | R_X86_64_GOTPCREL<<24,
		'8' | R_386_32<<24,
		'8' | R_386_PC32<<24,
		'8' | R_386_GOT32<<24,
		'8' | R_386_PLT32<<24,
		'8' | R_386_GOTOFF<<24,
		'8' | R_386_GOTPC<<24,
		'9' | R_PPC64_REL24<<24:
		*siz = 4

	case '6' | R_X86_64_64<<24,
		'9' | R_PPC64_ADDR64<<24:
		*siz = 8
	}

	return 256 + elftype
}
