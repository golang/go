package ld

import (
	"bytes"
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"io"
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
	ElfClassNone = 0
	ElfClass32   = 1
	ElfClass64   = 2
)

const (
	ElfDataNone = 0
	ElfDataLsb  = 1
	ElfDataMsb  = 2
)

const (
	ElfTypeNone         = 0
	ElfTypeRelocatable  = 1
	ElfTypeExecutable   = 2
	ElfTypeSharedObject = 3
	ElfTypeCore         = 4
)

const (
	ElfMachNone        = 0
	ElfMach32100       = 1
	ElfMachSparc       = 2
	ElfMach386         = 3
	ElfMach68000       = 4
	ElfMach88000       = 5
	ElfMach486         = 6
	ElfMach860         = 7
	ElfMachMips        = 8
	ElfMachS370        = 9
	ElfMachMipsLe      = 10
	ElfMachParisc      = 15
	ElfMachVpp500      = 17
	ElfMachSparc32Plus = 18
	ElfMach960         = 19
	ElfMachPower       = 20
	ElfMachPower64     = 21
	ElfMachS390        = 22
	ElfMachV800        = 36
	ElfMachFr20        = 37
	ElfMachRh32        = 38
	ElfMachRce         = 39
	ElfMachArm         = 40
	ElfMachAlpha       = 41
	ElfMachSH          = 42
	ElfMachSparc9      = 43
	ElfMachAmd64       = 62
	ElfMachArm64       = 183
)

const (
	ElfAbiNone     = 0
	ElfAbiSystemV  = 0
	ElfAbiHPUX     = 1
	ElfAbiNetBSD   = 2
	ElfAbiLinux    = 3
	ElfAbiSolaris  = 6
	ElfAbiAix      = 7
	ElfAbiIrix     = 8
	ElfAbiFreeBSD  = 9
	ElfAbiTru64    = 10
	ElfAbiModesto  = 11
	ElfAbiOpenBSD  = 12
	ElfAbiARM      = 97
	ElfAbiEmbedded = 255
)

const (
	ElfSectNone      = 0
	ElfSectProgbits  = 1
	ElfSectSymtab    = 2
	ElfSectStrtab    = 3
	ElfSectRela      = 4
	ElfSectHash      = 5
	ElfSectDynamic   = 6
	ElfSectNote      = 7
	ElfSectNobits    = 8
	ElfSectRel       = 9
	ElfSectShlib     = 10
	ElfSectDynsym    = 11
	ElfSectFlagWrite = 0x1
	ElfSectFlagAlloc = 0x2
	ElfSectFlagExec  = 0x4
)

const (
	ElfSymBindLocal  = 0
	ElfSymBindGlobal = 1
	ElfSymBindWeak   = 2
)

const (
	ElfSymTypeNone    = 0
	ElfSymTypeObject  = 1
	ElfSymTypeFunc    = 2
	ElfSymTypeSection = 3
	ElfSymTypeFile    = 4
)

const (
	ElfSymShnNone   = 0
	ElfSymShnAbs    = 0xFFF1
	ElfSymShnCommon = 0xFFF2
)

const (
	ElfProgNone      = 0
	ElfProgLoad      = 1
	ElfProgDynamic   = 2
	ElfProgInterp    = 3
	ElfProgNote      = 4
	ElfProgShlib     = 5
	ElfProgPhdr      = 6
	ElfProgFlagExec  = 0x1
	ElfProgFlagWrite = 0x2
	ElfProgFlagRead  = 0x4
)

const (
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
	f         *obj.Biobuf
	base      int64 // offset in f where ELF begins
	length    int64 // length of ELF
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

const (
	Tag_file                 = 1
	Tag_CPU_name             = 4
	Tag_CPU_raw_name         = 5
	Tag_compatibility        = 32
	Tag_nodefaults           = 64
	Tag_also_compatible_with = 65
	Tag_ABI_VFP_args         = 28
)

type elfAttribute struct {
	tag  uint64
	sval string
	ival uint64
}

type elfAttributeList struct {
	data []byte
	err  error
}

func (a *elfAttributeList) string() string {
	if a.err != nil {
		return ""
	}
	nul := bytes.IndexByte(a.data, 0)
	if nul < 0 {
		a.err = io.EOF
		return ""
	}
	s := string(a.data[:nul])
	a.data = a.data[nul+1:]
	return s
}

func (a *elfAttributeList) uleb128() uint64 {
	if a.err != nil {
		return 0
	}
	v, size := binary.Uvarint(a.data)
	a.data = a.data[size:]
	return v
}

// Read an elfAttribute from the list following the rules used on ARM systems.
func (a *elfAttributeList) armAttr() elfAttribute {
	attr := elfAttribute{tag: a.uleb128()}
	switch {
	case attr.tag == Tag_compatibility:
		attr.ival = a.uleb128()
		attr.sval = a.string()

	case attr.tag == 64: // Tag_nodefaults has no argument

	case attr.tag == 65: // Tag_also_compatible_with
		// Not really, but we don't actually care about this tag.
		attr.sval = a.string()

	// Tag with string argument
	case attr.tag == Tag_CPU_name || attr.tag == Tag_CPU_raw_name || (attr.tag >= 32 && attr.tag&1 != 0):
		attr.sval = a.string()

	default: // Tag with integer argument
		attr.ival = a.uleb128()
	}
	return attr
}

func (a *elfAttributeList) done() bool {
	if a.err != nil || len(a.data) == 0 {
		return true
	}
	return false
}

// Look for the attribute that indicates the object uses the hard-float ABI (a
// file-level attribute with tag Tag_VFP_arch and value 1). Unfortunately the
// format used means that we have to parse all of the file-level attributes to
// find the one we are looking for. This format is slightly documented in "ELF
// for the ARM Architecture" but mostly this is derived from reading the source
// to gold and readelf.
func parseArmAttributes(e binary.ByteOrder, data []byte) {
	// We assume the soft-float ABI unless we see a tag indicating otherwise.
	if ehdr.flags == 0x5000002 {
		ehdr.flags = 0x5000202
	}
	if data[0] != 'A' {
		fmt.Fprintf(&Bso, ".ARM.attributes has unexpected format %c\n", data[0])
		return
	}
	data = data[1:]
	for len(data) != 0 {
		sectionlength := e.Uint32(data)
		sectiondata := data[4:sectionlength]
		data = data[sectionlength:]

		nulIndex := bytes.IndexByte(sectiondata, 0)
		if nulIndex < 0 {
			fmt.Fprintf(&Bso, "corrupt .ARM.attributes (section name not NUL-terminated)\n")
			return
		}
		name := string(sectiondata[:nulIndex])
		sectiondata = sectiondata[nulIndex+1:]

		if name != "aeabi" {
			continue
		}
		for len(sectiondata) != 0 {
			subsectiontag, sz := binary.Uvarint(sectiondata)
			subsectionsize := e.Uint32(sectiondata[sz:])
			subsectiondata := sectiondata[sz+4 : subsectionsize]
			sectiondata = sectiondata[subsectionsize:]

			if subsectiontag == Tag_file {
				attrList := elfAttributeList{data: subsectiondata}
				for !attrList.done() {
					attr := attrList.armAttr()
					if attr.tag == Tag_ABI_VFP_args && attr.ival == 1 {
						ehdr.flags = 0x5000402 // has entry point, Version5 EABI, hard-float ABI
					}
				}
				if attrList.err != nil {
					fmt.Fprintf(&Bso, "could not parse .ARM.attributes\n")
				}
			}
		}
	}
}

func ldelf(f *obj.Biobuf, pkg string, length int64, pn string) {
	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f ldelf %s\n", obj.Cputime(), pn)
	}

	Ctxt.Version++
	base := int32(obj.Boffset(f))

	var add uint64
	var e binary.ByteOrder
	var elfobj *ElfObj
	var err error
	var flag int
	var hdr *ElfHdrBytes
	var hdrbuf [64]uint8
	var info uint64
	var is64 int
	var j int
	var n int
	var name string
	var p []byte
	var r []Reloc
	var rela int
	var rp *Reloc
	var rsect *ElfSect
	var s *LSym
	var sect *ElfSect
	var sym ElfSym
	var symbols []*LSym
	if obj.Bread(f, hdrbuf[:]) != len(hdrbuf) {
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
		is64 = 1
		hdr := new(ElfHdrBytes64)
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

	case '0':
		if elfobj.machine != ElfMachMips || hdr.Ident[4] != ElfClass64 {
			Diag("%s: elf object but not mips64", pn)
			return
		}

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

	case '7':
		if e != binary.LittleEndian || elfobj.machine != ElfMachArm64 || hdr.Ident[4] != ElfClass64 {
			Diag("%s: elf object but not arm64", pn)
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
	for i := 0; uint(i) < elfobj.nsect; i++ {
		if obj.Bseek(f, int64(uint64(base)+elfobj.shoff+uint64(int64(i)*int64(elfobj.shentsize))), 0) < 0 {
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
	for i := 0; uint(i) < elfobj.nsect; i++ {
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
	for i := 0; uint(i) < elfobj.nsect; i++ {
		sect = &elfobj.sect[i]
		if sect.type_ == SHT_ARM_ATTRIBUTES && sect.name == ".ARM.attributes" {
			if err = elfmap(elfobj, sect); err != nil {
				goto bad
			}
			parseArmAttributes(e, sect.base[:sect.size])
		}
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
			s.Type = obj.SRODATA

		case ElfSectFlagAlloc + ElfSectFlagWrite:
			if sect.type_ == ElfSectNobits {
				s.Type = obj.SNOPTRBSS
			} else {
				s.Type = obj.SNOPTRDATA
			}

		case ElfSectFlagAlloc + ElfSectFlagExec:
			s.Type = obj.STEXT
		}

		if sect.name == ".got" || sect.name == ".toc" {
			s.Type = obj.SELFGOT
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

	for i := 1; i < elfobj.nsymtab; i++ {
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
			if s.Type == 0 || s.Type == obj.SXREF {
				s.Type = obj.SNOPTRBSS
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
			if strings.HasPrefix(sym.name, ".Linfo_string") { // clang does this
				continue
			}

			if sym.name == "" && sym.type_ == 0 && sect.name == ".debug_str" {
				// This reportedly happens with clang 3.7 on ARM.
				// See issue 13139.
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
			Exitf("%s: duplicate symbol reference: %s in both %s and %s", pn, s.Name, s.Outer.Name, sect.sym.Name)
		}

		s.Sub = sect.sym.Sub
		sect.sym.Sub = s
		s.Type = sect.sym.Type | s.Type&^obj.SMASK | obj.SSUB
		if s.Cgoexport&CgoExportDynamic == 0 {
			s.Dynimplib = "" // satisfy dynimport
		}
		s.Value = int64(sym.value)
		s.Size = int64(sym.size)
		s.Outer = sect.sym
		if sect.sym.Type == obj.STEXT {
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
	for i := 0; uint(i) < elfobj.nsect; i++ {
		s = elfobj.sect[i].sym
		if s == nil {
			continue
		}
		if s.Sub != nil {
			s.Sub = listsort(s.Sub, valuecmp, listsubp)
		}
		if s.Type == obj.STEXT {
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
	for i := 0; uint(i) < elfobj.nsect; i++ {
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
	for i := 0; uint(i) < elfobj.nsect; i++ {
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
	if obj.Bseek(elfobj.f, int64(uint64(elfobj.base)+sect.off), 0) < 0 || obj.Bread(elfobj.f, sect.base) != len(sect.base) {
		return err
	}

	return nil
}

func readelfsym(elfobj *ElfObj, i int, sym *ElfSym, needSym int) (err error) {
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

	var s *LSym
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

	case ElfSymTypeObject, ElfSymTypeFunc, ElfSymTypeNone:
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
					s.Type |= obj.SHIDDEN
					s.Dupok = 1
				}
			}

		case ElfSymBindLocal:
			if Thearch.Thechar == '5' && (strings.HasPrefix(sym.name, "$a") || strings.HasPrefix(sym.name, "$d")) {
				// binutils for arm generate these mapping
				// symbols, ignore these
				break
			}

			if sym.name == ".TOC." {
				// We need to be able to look this up,
				// so put it in the hash table.
				if needSym != 0 {
					s = Linklookup(Ctxt, sym.name, Ctxt.Version)
					s.Type |= obj.SHIDDEN
				}

				break
			}

			if needSym != 0 {
				// local names and hidden visiblity global names are unique
				// and should only reference by its index, not name, so we
				// don't bother to add them into hash table
				s = linknewsym(Ctxt, sym.name, Ctxt.Version)

				s.Type |= obj.SHIDDEN
			}

		case ElfSymBindWeak:
			if needSym != 0 {
				s = Linklookup(Ctxt, sym.name, 0)
				if sym.other == 2 {
					s.Type |= obj.SHIDDEN
				}
			}

		default:
			err = fmt.Errorf("%s: invalid symbol binding %d", sym.name, sym.bind)
			return err
		}
	}

	if s != nil && s.Type == 0 && sym.type_ != ElfSymTypeSection {
		s.Type = obj.SXREF
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
	a := &x[i]
	b := &x[j]
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
		'6' | R_X86_64_GOTPCRELX<<24,
		'6' | R_X86_64_REX_GOTPCRELX<<24,
		'8' | R_386_32<<24,
		'8' | R_386_PC32<<24,
		'8' | R_386_GOT32<<24,
		'8' | R_386_PLT32<<24,
		'8' | R_386_GOTOFF<<24,
		'8' | R_386_GOTPC<<24,
		'8' | R_386_GOT32X<<24,
		'9' | R_PPC64_REL24<<24,
		'9' | R_PPC_REL32<<24:
		*siz = 4

	case '6' | R_X86_64_64<<24,
		'9' | R_PPC64_ADDR64<<24:
		*siz = 8
	}

	return 256 + elftype
}
