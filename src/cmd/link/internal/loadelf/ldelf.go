// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loadelf implements an ELF file reader.
package loadelf

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"debug/elf"
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
	ElfSymTypeCommon  = 5
	ElfSymTypeTLS     = 6
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

// TODO(crawshaw): de-duplicate with cmd/link/internal/ld/elf.go.
const (
	ELF64SYMSIZE = 24
	ELF32SYMSIZE = 16

	SHT_ARM_ATTRIBUTES = 0x70000003
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
	sym     *sym.Symbol
}

type ElfObj struct {
	f         *bio.Reader
	base      int64 // offset in f where ELF begins
	length    int64 // length of ELF
	is64      int
	name      string
	e         binary.ByteOrder
	sect      []ElfSect
	nsect     uint
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
	sym   *sym.Symbol
}

var ElfMagic = [4]uint8{0x7F, 'E', 'L', 'F'}

const (
	TagFile               = 1
	TagCPUName            = 4
	TagCPURawName         = 5
	TagCompatibility      = 32
	TagNoDefaults         = 64
	TagAlsoCompatibleWith = 65
	TagABIVFPArgs         = 28
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
	case attr.tag == TagCompatibility:
		attr.ival = a.uleb128()
		attr.sval = a.string()

	case attr.tag == 64: // Tag_nodefaults has no argument

	case attr.tag == 65: // Tag_also_compatible_with
		// Not really, but we don't actually care about this tag.
		attr.sval = a.string()

	// Tag with string argument
	case attr.tag == TagCPUName || attr.tag == TagCPURawName || (attr.tag >= 32 && attr.tag&1 != 0):
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
func parseArmAttributes(e binary.ByteOrder, data []byte) (found bool, ehdrFlags uint32, err error) {
	found = false
	if data[0] != 'A' {
		return false, 0, fmt.Errorf(".ARM.attributes has unexpected format %c\n", data[0])
	}
	data = data[1:]
	for len(data) != 0 {
		sectionlength := e.Uint32(data)
		sectiondata := data[4:sectionlength]
		data = data[sectionlength:]

		nulIndex := bytes.IndexByte(sectiondata, 0)
		if nulIndex < 0 {
			return false, 0, fmt.Errorf("corrupt .ARM.attributes (section name not NUL-terminated)\n")
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

			if subsectiontag != TagFile {
				continue
			}
			attrList := elfAttributeList{data: subsectiondata}
			for !attrList.done() {
				attr := attrList.armAttr()
				if attr.tag == TagABIVFPArgs && attr.ival == 1 {
					found = true
					ehdrFlags = 0x5000402 // has entry point, Version5 EABI, hard-float ABI
				}
			}
			if attrList.err != nil {
				return false, 0, fmt.Errorf("could not parse .ARM.attributes\n")
			}
		}
	}
	return found, ehdrFlags, nil
}

// Load loads the ELF file pn from f.
// Symbols are written into syms, and a slice of the text symbols is returned.
//
// On ARM systems, Load will attempt to determine what ELF header flags to
// emit by scanning the attributes in the ELF file being loaded. The
// parameter initEhdrFlags contains the current header flags for the output
// object, and the returned ehdrFlags contains what this Load function computes.
// TODO: find a better place for this logic.
func Load(arch *sys.Arch, syms *sym.Symbols, f *bio.Reader, pkg string, length int64, pn string, initEhdrFlags uint32) (textp []*sym.Symbol, ehdrFlags uint32, err error) {
	errorf := func(str string, args ...interface{}) ([]*sym.Symbol, uint32, error) {
		return nil, 0, fmt.Errorf("loadelf: %s: %v", pn, fmt.Sprintf(str, args...))
	}

	localSymVersion := syms.IncVersion()
	base := f.Offset()

	var hdrbuf [64]uint8
	if _, err := io.ReadFull(f, hdrbuf[:]); err != nil {
		return errorf("malformed elf file: %v", err)
	}
	hdr := new(ElfHdrBytes)
	binary.Read(bytes.NewReader(hdrbuf[:]), binary.BigEndian, hdr) // only byte arrays; byte order doesn't matter
	if string(hdr.Ident[:4]) != "\x7FELF" {
		return errorf("malformed elf file, bad header")
	}
	var e binary.ByteOrder
	switch hdr.Ident[5] {
	case ElfDataLsb:
		e = binary.LittleEndian

	case ElfDataMsb:
		e = binary.BigEndian

	default:
		return errorf("malformed elf file, unknown header")
	}

	// read header
	elfobj := new(ElfObj)

	elfobj.e = e
	elfobj.f = f
	elfobj.base = base
	elfobj.length = length
	elfobj.name = pn

	is64 := 0
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

	if v := uint32(hdr.Ident[6]); v != elfobj.version {
		return errorf("malformed elf version: got %d, want %d", v, elfobj.version)
	}

	if e.Uint16(hdr.Type[:]) != ElfTypeRelocatable {
		return errorf("elf but not elf relocatable object")
	}

	switch arch.Family {
	default:
		return errorf("elf %s unimplemented", arch.Name)

	case sys.MIPS:
		if elfobj.machine != ElfMachMips || hdr.Ident[4] != ElfClass32 {
			return errorf("elf object but not mips")
		}

	case sys.MIPS64:
		if elfobj.machine != ElfMachMips || hdr.Ident[4] != ElfClass64 {
			return errorf("elf object but not mips64")
		}

	case sys.ARM:
		if e != binary.LittleEndian || elfobj.machine != ElfMachArm || hdr.Ident[4] != ElfClass32 {
			return errorf("elf object but not arm")
		}

	case sys.AMD64:
		if e != binary.LittleEndian || elfobj.machine != ElfMachAmd64 || hdr.Ident[4] != ElfClass64 {
			return errorf("elf object but not amd64")
		}

	case sys.ARM64:
		if e != binary.LittleEndian || elfobj.machine != ElfMachArm64 || hdr.Ident[4] != ElfClass64 {
			return errorf("elf object but not arm64")
		}

	case sys.I386:
		if e != binary.LittleEndian || elfobj.machine != ElfMach386 || hdr.Ident[4] != ElfClass32 {
			return errorf("elf object but not 386")
		}

	case sys.PPC64:
		if elfobj.machine != ElfMachPower64 || hdr.Ident[4] != ElfClass64 {
			return errorf("elf object but not ppc64")
		}

	case sys.S390X:
		if elfobj.machine != ElfMachS390 || hdr.Ident[4] != ElfClass64 {
			return errorf("elf object but not s390x")
		}
	}

	// load section list into memory.
	elfobj.sect = make([]ElfSect, elfobj.shnum)

	elfobj.nsect = uint(elfobj.shnum)
	for i := 0; uint(i) < elfobj.nsect; i++ {
		f.MustSeek(int64(uint64(base)+elfobj.shoff+uint64(int64(i)*int64(elfobj.shentsize))), 0)
		sect := &elfobj.sect[i]
		if is64 != 0 {
			var b ElfSectBytes64

			if err := binary.Read(f, e, &b); err != nil {
				return errorf("malformed elf file: %v", err)
			}

			sect.nameoff = e.Uint32(b.Name[:])
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

			if err := binary.Read(f, e, &b); err != nil {
				return errorf("malformed elf file: %v", err)
			}

			sect.nameoff = e.Uint32(b.Name[:])
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
		return errorf("malformed elf file: shstrndx out of range %d >= %d", elfobj.shstrndx, elfobj.nsect)
	}

	sect := &elfobj.sect[elfobj.shstrndx]
	if err := elfmap(elfobj, sect); err != nil {
		return errorf("malformed elf file: %v", err)
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
		return errorf("elf object has symbol table with invalid string table link")
	}

	elfobj.symstr = &elfobj.sect[elfobj.symtab.link]
	if is64 != 0 {
		elfobj.nsymtab = int(elfobj.symtab.size / ELF64SYMSIZE)
	} else {
		elfobj.nsymtab = int(elfobj.symtab.size / ELF32SYMSIZE)
	}

	if err := elfmap(elfobj, elfobj.symtab); err != nil {
		return errorf("malformed elf file: %v", err)
	}
	if err := elfmap(elfobj, elfobj.symstr); err != nil {
		return errorf("malformed elf file: %v", err)
	}

	// load text and data segments into memory.
	// they are not as small as the section lists, but we'll need
	// the memory anyway for the symbol images, so we might
	// as well use one large chunk.

	// create symbols for elfmapped sections
	sectsymNames := make(map[string]bool)
	counter := 0
	for i := 0; uint(i) < elfobj.nsect; i++ {
		sect = &elfobj.sect[i]
		if sect.type_ == SHT_ARM_ATTRIBUTES && sect.name == ".ARM.attributes" {
			if err := elfmap(elfobj, sect); err != nil {
				return errorf("%s: malformed elf file: %v", pn, err)
			}
			// We assume the soft-float ABI unless we see a tag indicating otherwise.
			if initEhdrFlags == 0x5000002 {
				ehdrFlags = 0x5000202
			} else {
				ehdrFlags = initEhdrFlags
			}
			found, newEhdrFlags, err := parseArmAttributes(e, sect.base[:sect.size])
			if err != nil {
				// TODO(dfc) should this return an error?
				log.Printf("%s: %v", pn, err)
			}
			if found {
				ehdrFlags = newEhdrFlags
			}
		}
		if (sect.type_ != ElfSectProgbits && sect.type_ != ElfSectNobits) || sect.flags&ElfSectFlagAlloc == 0 {
			continue
		}
		if sect.type_ != ElfSectNobits {
			if err := elfmap(elfobj, sect); err != nil {
				return errorf("%s: malformed elf file: %v", pn, err)
			}
		}

		name := fmt.Sprintf("%s(%s)", pkg, sect.name)
		for sectsymNames[name] {
			counter++
			name = fmt.Sprintf("%s(%s%d)", pkg, sect.name, counter)
		}
		sectsymNames[name] = true

		s := syms.Lookup(name, localSymVersion)

		switch int(sect.flags) & (ElfSectFlagAlloc | ElfSectFlagWrite | ElfSectFlagExec) {
		default:
			return errorf("%s: unexpected flags for ELF section %s", pn, sect.name)

		case ElfSectFlagAlloc:
			s.Type = sym.SRODATA

		case ElfSectFlagAlloc + ElfSectFlagWrite:
			if sect.type_ == ElfSectNobits {
				s.Type = sym.SNOPTRBSS
			} else {
				s.Type = sym.SNOPTRDATA
			}

		case ElfSectFlagAlloc + ElfSectFlagExec:
			s.Type = sym.STEXT
		}

		if sect.name == ".got" || sect.name == ".toc" {
			s.Type = sym.SELFGOT
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
	symbols := make([]*sym.Symbol, elfobj.nsymtab)

	for i := 1; i < elfobj.nsymtab; i++ {
		var elfsym ElfSym
		if err := readelfsym(arch, syms, elfobj, i, &elfsym, 1, localSymVersion); err != nil {
			return errorf("%s: malformed elf file: %v", pn, err)
		}
		symbols[i] = elfsym.sym
		if elfsym.type_ != ElfSymTypeFunc && elfsym.type_ != ElfSymTypeObject && elfsym.type_ != ElfSymTypeNone && elfsym.type_ != ElfSymTypeCommon {
			continue
		}
		if elfsym.shndx == ElfSymShnCommon || elfsym.type_ == ElfSymTypeCommon {
			s := elfsym.sym
			if uint64(s.Size) < elfsym.size {
				s.Size = int64(elfsym.size)
			}
			if s.Type == 0 || s.Type == sym.SXREF {
				s.Type = sym.SNOPTRBSS
			}
			continue
		}

		if uint(elfsym.shndx) >= elfobj.nsect || elfsym.shndx == 0 {
			continue
		}

		// even when we pass needSym == 1 to readelfsym, it might still return nil to skip some unwanted symbols
		if elfsym.sym == nil {
			continue
		}
		sect = &elfobj.sect[elfsym.shndx]
		if sect.sym == nil {
			if strings.HasPrefix(elfsym.name, ".Linfo_string") { // clang does this
				continue
			}

			if elfsym.name == "" && elfsym.type_ == 0 && sect.name == ".debug_str" {
				// This reportedly happens with clang 3.7 on ARM.
				// See issue 13139.
				continue
			}

			if strings.HasPrefix(elfsym.name, "$d") && elfsym.type_ == 0 && sect.name == ".debug_frame" {
				// "$d" is a marker, not a real symbol.
				// This happens with gcc on ARM64.
				// See https://sourceware.org/bugzilla/show_bug.cgi?id=21809
				continue
			}

			if strings.HasPrefix(elfsym.name, ".LASF") { // gcc on s390x does this
				continue
			}
			return errorf("%v: sym#%d: ignoring symbol in section %d (type %d)", elfsym.sym, i, elfsym.shndx, elfsym.type_)
		}

		s := elfsym.sym
		if s.Outer != nil {
			if s.Attr.DuplicateOK() {
				continue
			}
			return errorf("duplicate symbol reference: %s in both %s and %s", s.Name, s.Outer.Name, sect.sym.Name)
		}

		s.Sub = sect.sym.Sub
		sect.sym.Sub = s
		s.Type = sect.sym.Type
		s.Attr |= sym.AttrSubSymbol
		if !s.Attr.CgoExportDynamic() {
			s.SetDynimplib("") // satisfy dynimport
		}
		s.Value = int64(elfsym.value)
		s.Size = int64(elfsym.size)
		s.Outer = sect.sym
		if sect.sym.Type == sym.STEXT {
			if s.Attr.External() && !s.Attr.DuplicateOK() {
				return errorf("%v: duplicate symbol definition", s)
			}
			s.Attr |= sym.AttrExternal
		}

		if elfobj.machine == ElfMachPower64 {
			flag := int(elfsym.other) >> 5
			if 2 <= flag && flag <= 6 {
				s.SetLocalentry(1 << uint(flag-2))
			} else if flag == 7 {
				return errorf("%v: invalid sym.other 0x%x", s, elfsym.other)
			}
		}
	}

	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for i := uint(0); i < elfobj.nsect; i++ {
		s := elfobj.sect[i].sym
		if s == nil {
			continue
		}
		if s.Sub != nil {
			s.Sub = sym.SortSub(s.Sub)
		}
		if s.Type == sym.STEXT {
			if s.Attr.OnList() {
				return errorf("symbol %s listed multiple times", s.Name)
			}
			s.Attr |= sym.AttrOnList
			textp = append(textp, s)
			for s = s.Sub; s != nil; s = s.Sub {
				if s.Attr.OnList() {
					return errorf("symbol %s listed multiple times", s.Name)
				}
				s.Attr |= sym.AttrOnList
				textp = append(textp, s)
			}
		}
	}

	// load relocations
	for i := uint(0); i < elfobj.nsect; i++ {
		rsect := &elfobj.sect[i]
		if rsect.type_ != ElfSectRela && rsect.type_ != ElfSectRel {
			continue
		}
		if rsect.info >= uint32(elfobj.nsect) || elfobj.sect[rsect.info].base == nil {
			continue
		}
		sect = &elfobj.sect[rsect.info]
		if err := elfmap(elfobj, rsect); err != nil {
			return errorf("malformed elf file: %v", err)
		}
		rela := 0
		if rsect.type_ == ElfSectRela {
			rela = 1
		}
		n := int(rsect.size / uint64(4+4*is64) / uint64(2+rela))
		r := make([]sym.Reloc, n)
		p := rsect.base
		for j := 0; j < n; j++ {
			var add uint64
			rp := &r[j]
			var info uint64
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
				var elfsym ElfSym
				if err := readelfsym(arch, syms, elfobj, int(info>>32), &elfsym, 0, 0); err != nil {
					return errorf("malformed elf file: %v", err)
				}
				elfsym.sym = symbols[info>>32]
				if elfsym.sym == nil {
					return errorf("malformed elf file: %s#%d: reloc of invalid sym #%d %s shndx=%d type=%d", sect.sym.Name, j, int(info>>32), elfsym.name, elfsym.shndx, elfsym.type_)
				}

				rp.Sym = elfsym.sym
			}

			rp.Type = objabi.ElfRelocOffset + objabi.RelocType(info)
			rp.Siz, err = relSize(arch, pn, uint32(info))
			if err != nil {
				return nil, 0, err
			}
			if rela != 0 {
				rp.Add = int64(add)
			} else {
				// load addend from image
				if rp.Siz == 4 {
					rp.Add = int64(e.Uint32(sect.base[rp.Off:]))
				} else if rp.Siz == 8 {
					rp.Add = int64(e.Uint64(sect.base[rp.Off:]))
				} else {
					return errorf("invalid rela size %d", rp.Siz)
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
		sort.Sort(sym.RelocByOff(r[:n]))
		// just in case

		s := sect.sym
		s.R = r
		s.R = s.R[:n]
	}

	return textp, ehdrFlags, nil
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
	elfobj.f.MustSeek(int64(uint64(elfobj.base)+sect.off), 0)
	if _, err := io.ReadFull(elfobj.f, sect.base); err != nil {
		return fmt.Errorf("short read: %v", err)
	}

	return nil
}

func readelfsym(arch *sys.Arch, syms *sym.Symbols, elfobj *ElfObj, i int, elfsym *ElfSym, needSym int, localSymVersion int) (err error) {
	if i >= elfobj.nsymtab || i < 0 {
		err = fmt.Errorf("invalid elf symbol index")
		return err
	}

	if i == 0 {
		return fmt.Errorf("readym: read null symbol!")
	}

	if elfobj.is64 != 0 {
		b := new(ElfSymBytes64)
		binary.Read(bytes.NewReader(elfobj.symtab.base[i*ELF64SYMSIZE:(i+1)*ELF64SYMSIZE]), elfobj.e, b)
		elfsym.name = cstring(elfobj.symstr.base[elfobj.e.Uint32(b.Name[:]):])
		elfsym.value = elfobj.e.Uint64(b.Value[:])
		elfsym.size = elfobj.e.Uint64(b.Size[:])
		elfsym.shndx = elfobj.e.Uint16(b.Shndx[:])
		elfsym.bind = b.Info >> 4
		elfsym.type_ = b.Info & 0xf
		elfsym.other = b.Other
	} else {
		b := new(ElfSymBytes)
		binary.Read(bytes.NewReader(elfobj.symtab.base[i*ELF32SYMSIZE:(i+1)*ELF32SYMSIZE]), elfobj.e, b)
		elfsym.name = cstring(elfobj.symstr.base[elfobj.e.Uint32(b.Name[:]):])
		elfsym.value = uint64(elfobj.e.Uint32(b.Value[:]))
		elfsym.size = uint64(elfobj.e.Uint32(b.Size[:]))
		elfsym.shndx = elfobj.e.Uint16(b.Shndx[:])
		elfsym.bind = b.Info >> 4
		elfsym.type_ = b.Info & 0xf
		elfsym.other = b.Other
	}

	var s *sym.Symbol
	if elfsym.name == "_GLOBAL_OFFSET_TABLE_" {
		elfsym.name = ".got"
	}
	if elfsym.name == ".TOC." {
		// Magic symbol on ppc64.  Will be set to this object
		// file's .got+0x8000.
		elfsym.bind = ElfSymBindLocal
	}

	switch elfsym.type_ {
	case ElfSymTypeSection:
		s = elfobj.sect[elfsym.shndx].sym

	case ElfSymTypeObject, ElfSymTypeFunc, ElfSymTypeNone, ElfSymTypeCommon:
		switch elfsym.bind {
		case ElfSymBindGlobal:
			if needSym != 0 {
				s = syms.Lookup(elfsym.name, 0)

				// for global scoped hidden symbols we should insert it into
				// symbol hash table, but mark them as hidden.
				// __i686.get_pc_thunk.bx is allowed to be duplicated, to
				// workaround that we set dupok.
				// TODO(minux): correctly handle __i686.get_pc_thunk.bx without
				// set dupok generally. See https://golang.org/cl/5823055
				// comment #5 for details.
				if s != nil && elfsym.other == 2 {
					s.Attr |= sym.AttrDuplicateOK | sym.AttrVisibilityHidden
				}
			}

		case ElfSymBindLocal:
			if (arch.Family == sys.ARM || arch.Family == sys.ARM64) && (strings.HasPrefix(elfsym.name, "$a") || strings.HasPrefix(elfsym.name, "$d") || strings.HasPrefix(elfsym.name, "$x")) {
				// binutils for arm and arm64 generate these mapping
				// symbols, ignore these
				break
			}

			if elfsym.name == ".TOC." {
				// We need to be able to look this up,
				// so put it in the hash table.
				if needSym != 0 {
					s = syms.Lookup(elfsym.name, localSymVersion)
					s.Attr |= sym.AttrVisibilityHidden
				}

				break
			}

			if needSym != 0 {
				// local names and hidden global names are unique
				// and should only be referenced by their index, not name, so we
				// don't bother to add them into the hash table
				s = syms.Newsym(elfsym.name, localSymVersion)

				s.Attr |= sym.AttrVisibilityHidden
			}

		case ElfSymBindWeak:
			if needSym != 0 {
				s = syms.Lookup(elfsym.name, 0)
				if elfsym.other == 2 {
					s.Attr |= sym.AttrVisibilityHidden
				}

				// Allow weak symbols to be duplicated when already defined.
				if s.Outer != nil {
					s.Attr |= sym.AttrDuplicateOK
				}
			}

		default:
			err = fmt.Errorf("%s: invalid symbol binding %d", elfsym.name, elfsym.bind)
			return err
		}
	}

	// TODO(mwhudson): the test of VisibilityHidden here probably doesn't make
	// sense and should be removed when someone has thought about it properly.
	if s != nil && s.Type == 0 && !s.Attr.VisibilityHidden() && elfsym.type_ != ElfSymTypeSection {
		s.Type = sym.SXREF
	}
	elfsym.sym = s

	return nil
}

func relSize(arch *sys.Arch, pn string, elftype uint32) (uint8, error) {
	// TODO(mdempsky): Replace this with a struct-valued switch statement
	// once golang.org/issue/15164 is fixed or found to not impair cmd/link
	// performance.

	const (
		AMD64 = uint32(sys.AMD64)
		ARM   = uint32(sys.ARM)
		ARM64 = uint32(sys.ARM64)
		I386  = uint32(sys.I386)
		PPC64 = uint32(sys.PPC64)
		S390X = uint32(sys.S390X)
	)

	switch uint32(arch.Family) | elftype<<16 {
	default:
		return 0, fmt.Errorf("%s: unknown relocation type %d; compiled without -fpic?", pn, elftype)

	case S390X | uint32(elf.R_390_8)<<16:
		return 1, nil

	case PPC64 | uint32(elf.R_PPC64_TOC16)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_LO)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_HI)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_HA)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_DS)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_LO_DS)<<16,
		PPC64 | uint32(elf.R_PPC64_REL16_LO)<<16,
		PPC64 | uint32(elf.R_PPC64_REL16_HI)<<16,
		PPC64 | uint32(elf.R_PPC64_REL16_HA)<<16,
		S390X | uint32(elf.R_390_16)<<16,
		S390X | uint32(elf.R_390_GOT16)<<16,
		S390X | uint32(elf.R_390_PC16)<<16,
		S390X | uint32(elf.R_390_PC16DBL)<<16,
		S390X | uint32(elf.R_390_PLT16DBL)<<16:
		return 2, nil

	case ARM | uint32(elf.R_ARM_ABS32)<<16,
		ARM | uint32(elf.R_ARM_GOT32)<<16,
		ARM | uint32(elf.R_ARM_PLT32)<<16,
		ARM | uint32(elf.R_ARM_GOTOFF)<<16,
		ARM | uint32(elf.R_ARM_GOTPC)<<16,
		ARM | uint32(elf.R_ARM_THM_PC22)<<16,
		ARM | uint32(elf.R_ARM_REL32)<<16,
		ARM | uint32(elf.R_ARM_CALL)<<16,
		ARM | uint32(elf.R_ARM_V4BX)<<16,
		ARM | uint32(elf.R_ARM_GOT_PREL)<<16,
		ARM | uint32(elf.R_ARM_PC24)<<16,
		ARM | uint32(elf.R_ARM_JUMP24)<<16,
		ARM64 | uint32(elf.R_AARCH64_CALL26)<<16,
		ARM64 | uint32(elf.R_AARCH64_ADR_GOT_PAGE)<<16,
		ARM64 | uint32(elf.R_AARCH64_LD64_GOT_LO12_NC)<<16,
		ARM64 | uint32(elf.R_AARCH64_ADR_PREL_PG_HI21)<<16,
		ARM64 | uint32(elf.R_AARCH64_ADD_ABS_LO12_NC)<<16,
		ARM64 | uint32(elf.R_AARCH64_LDST8_ABS_LO12_NC)<<16,
		ARM64 | uint32(elf.R_AARCH64_LDST32_ABS_LO12_NC)<<16,
		ARM64 | uint32(elf.R_AARCH64_LDST64_ABS_LO12_NC)<<16,
		ARM64 | uint32(elf.R_AARCH64_LDST128_ABS_LO12_NC)<<16,
		ARM64 | uint32(elf.R_AARCH64_PREL32)<<16,
		ARM64 | uint32(elf.R_AARCH64_JUMP26)<<16,
		AMD64 | uint32(elf.R_X86_64_PC32)<<16,
		AMD64 | uint32(elf.R_X86_64_PLT32)<<16,
		AMD64 | uint32(elf.R_X86_64_GOTPCREL)<<16,
		AMD64 | uint32(elf.R_X86_64_GOTPCRELX)<<16,
		AMD64 | uint32(elf.R_X86_64_REX_GOTPCRELX)<<16,
		I386 | uint32(elf.R_386_32)<<16,
		I386 | uint32(elf.R_386_PC32)<<16,
		I386 | uint32(elf.R_386_GOT32)<<16,
		I386 | uint32(elf.R_386_PLT32)<<16,
		I386 | uint32(elf.R_386_GOTOFF)<<16,
		I386 | uint32(elf.R_386_GOTPC)<<16,
		I386 | uint32(elf.R_386_GOT32X)<<16,
		PPC64 | uint32(elf.R_PPC64_REL24)<<16,
		PPC64 | uint32(elf.R_PPC_REL32)<<16,
		S390X | uint32(elf.R_390_32)<<16,
		S390X | uint32(elf.R_390_PC32)<<16,
		S390X | uint32(elf.R_390_GOT32)<<16,
		S390X | uint32(elf.R_390_PLT32)<<16,
		S390X | uint32(elf.R_390_PC32DBL)<<16,
		S390X | uint32(elf.R_390_PLT32DBL)<<16,
		S390X | uint32(elf.R_390_GOTPCDBL)<<16,
		S390X | uint32(elf.R_390_GOTENT)<<16:
		return 4, nil

	case AMD64 | uint32(elf.R_X86_64_64)<<16,
		AMD64 | uint32(elf.R_X86_64_PC64)<<16,
		ARM64 | uint32(elf.R_AARCH64_ABS64)<<16,
		ARM64 | uint32(elf.R_AARCH64_PREL64)<<16,
		PPC64 | uint32(elf.R_PPC64_ADDR64)<<16,
		S390X | uint32(elf.R_390_GLOB_DAT)<<16,
		S390X | uint32(elf.R_390_RELATIVE)<<16,
		S390X | uint32(elf.R_390_GOTOFF)<<16,
		S390X | uint32(elf.R_390_GOTPC)<<16,
		S390X | uint32(elf.R_390_64)<<16,
		S390X | uint32(elf.R_390_PC64)<<16,
		S390X | uint32(elf.R_390_GOT64)<<16,
		S390X | uint32(elf.R_390_PLT64)<<16:
		return 8, nil
	}
}

func cstring(x []byte) string {
	i := bytes.IndexByte(x, '\x00')
	if i >= 0 {
		x = x[:i]
	}
	return string(x)
}
