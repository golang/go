// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loadelf implements an ELF file reader.
package loadelf

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"strings"
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

const (
	SHT_ARM_ATTRIBUTES = 0x70000003
)

type ElfSect struct {
	name        string
	nameoff     uint32
	type_       elf.SectionType
	flags       elf.SectionFlag
	addr        uint64
	off         uint64
	size        uint64
	link        uint32
	info        uint32
	align       uint64
	entsize     uint64
	base        []byte
	readOnlyMem bool // Is this section in readonly memory?
	sym         loader.Sym
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
	bind  elf.SymBind
	type_ elf.SymType
	other uint8
	shndx elf.SectionIndex
	sym   loader.Sym
}

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

	case attr.tag == TagNoDefaults: // Tag_nodefaults has no argument

	case attr.tag == TagAlsoCompatibleWith:
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
// Symbols are installed into the loader, and a slice of the text symbols is returned.
//
// On ARM systems, Load will attempt to determine what ELF header flags to
// emit by scanning the attributes in the ELF file being loaded. The
// parameter initEhdrFlags contains the current header flags for the output
// object, and the returned ehdrFlags contains what this Load function computes.
// TODO: find a better place for this logic.
func Load(l *loader.Loader, arch *sys.Arch, localSymVersion int, f *bio.Reader, pkg string, length int64, pn string, initEhdrFlags uint32) (textp []loader.Sym, ehdrFlags uint32, err error) {
	errorf := func(str string, args ...interface{}) ([]loader.Sym, uint32, error) {
		return nil, 0, fmt.Errorf("loadelf: %s: %v", pn, fmt.Sprintf(str, args...))
	}

	ehdrFlags = initEhdrFlags

	base := f.Offset()

	var hdrbuf [64]byte
	if _, err := io.ReadFull(f, hdrbuf[:]); err != nil {
		return errorf("malformed elf file: %v", err)
	}

	var e binary.ByteOrder
	switch elf.Data(hdrbuf[elf.EI_DATA]) {
	case elf.ELFDATA2LSB:
		e = binary.LittleEndian

	case elf.ELFDATA2MSB:
		e = binary.BigEndian

	default:
		return errorf("malformed elf file, unknown header")
	}

	hdr := new(elf.Header32)
	binary.Read(bytes.NewReader(hdrbuf[:]), e, hdr)

	if string(hdr.Ident[:elf.EI_CLASS]) != elf.ELFMAG {
		return errorf("malformed elf file, bad header")
	}

	// read header
	elfobj := new(ElfObj)

	elfobj.e = e
	elfobj.f = f
	elfobj.base = base
	elfobj.length = length
	elfobj.name = pn

	is64 := 0
	class := elf.Class(hdrbuf[elf.EI_CLASS])
	if class == elf.ELFCLASS64 {
		is64 = 1
		hdr := new(elf.Header64)
		binary.Read(bytes.NewReader(hdrbuf[:]), e, hdr)
		elfobj.type_ = uint32(hdr.Type)
		elfobj.machine = uint32(hdr.Machine)
		elfobj.version = hdr.Version
		elfobj.entry = hdr.Entry
		elfobj.phoff = hdr.Phoff
		elfobj.shoff = hdr.Shoff
		elfobj.flags = hdr.Flags
		elfobj.ehsize = uint32(hdr.Ehsize)
		elfobj.phentsize = uint32(hdr.Phentsize)
		elfobj.phnum = uint32(hdr.Phnum)
		elfobj.shentsize = uint32(hdr.Shentsize)
		elfobj.shnum = uint32(hdr.Shnum)
		elfobj.shstrndx = uint32(hdr.Shstrndx)
	} else {
		elfobj.type_ = uint32(hdr.Type)
		elfobj.machine = uint32(hdr.Machine)
		elfobj.version = hdr.Version
		elfobj.entry = uint64(hdr.Entry)
		elfobj.phoff = uint64(hdr.Phoff)
		elfobj.shoff = uint64(hdr.Shoff)
		elfobj.flags = hdr.Flags
		elfobj.ehsize = uint32(hdr.Ehsize)
		elfobj.phentsize = uint32(hdr.Phentsize)
		elfobj.phnum = uint32(hdr.Phnum)
		elfobj.shentsize = uint32(hdr.Shentsize)
		elfobj.shnum = uint32(hdr.Shnum)
		elfobj.shstrndx = uint32(hdr.Shstrndx)
	}

	elfobj.is64 = is64

	if v := uint32(hdrbuf[elf.EI_VERSION]); v != elfobj.version {
		return errorf("malformed elf version: got %d, want %d", v, elfobj.version)
	}

	if elf.Type(elfobj.type_) != elf.ET_REL {
		return errorf("elf but not elf relocatable object")
	}

	mach := elf.Machine(elfobj.machine)
	switch arch.Family {
	default:
		return errorf("elf %s unimplemented", arch.Name)

	case sys.MIPS:
		if mach != elf.EM_MIPS || class != elf.ELFCLASS32 {
			return errorf("elf object but not mips")
		}

	case sys.MIPS64:
		if mach != elf.EM_MIPS || class != elf.ELFCLASS64 {
			return errorf("elf object but not mips64")
		}
	case sys.Loong64:
		if mach != elf.EM_LOONGARCH || class != elf.ELFCLASS64 {
			return errorf("elf object but not loong64")
		}

	case sys.ARM:
		if e != binary.LittleEndian || mach != elf.EM_ARM || class != elf.ELFCLASS32 {
			return errorf("elf object but not arm")
		}

	case sys.AMD64:
		if e != binary.LittleEndian || mach != elf.EM_X86_64 || class != elf.ELFCLASS64 {
			return errorf("elf object but not amd64")
		}

	case sys.ARM64:
		if e != binary.LittleEndian || mach != elf.EM_AARCH64 || class != elf.ELFCLASS64 {
			return errorf("elf object but not arm64")
		}

	case sys.I386:
		if e != binary.LittleEndian || mach != elf.EM_386 || class != elf.ELFCLASS32 {
			return errorf("elf object but not 386")
		}

	case sys.PPC64:
		if mach != elf.EM_PPC64 || class != elf.ELFCLASS64 {
			return errorf("elf object but not ppc64")
		}

	case sys.RISCV64:
		if mach != elf.EM_RISCV || class != elf.ELFCLASS64 {
			return errorf("elf object but not riscv64")
		}

	case sys.S390X:
		if mach != elf.EM_S390 || class != elf.ELFCLASS64 {
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
			var b elf.Section64
			if err := binary.Read(f, e, &b); err != nil {
				return errorf("malformed elf file: %v", err)
			}

			sect.nameoff = b.Name
			sect.type_ = elf.SectionType(b.Type)
			sect.flags = elf.SectionFlag(b.Flags)
			sect.addr = b.Addr
			sect.off = b.Off
			sect.size = b.Size
			sect.link = b.Link
			sect.info = b.Info
			sect.align = b.Addralign
			sect.entsize = b.Entsize
		} else {
			var b elf.Section32

			if err := binary.Read(f, e, &b); err != nil {
				return errorf("malformed elf file: %v", err)
			}
			sect.nameoff = b.Name
			sect.type_ = elf.SectionType(b.Type)
			sect.flags = elf.SectionFlag(b.Flags)
			sect.addr = uint64(b.Addr)
			sect.off = uint64(b.Off)
			sect.size = uint64(b.Size)
			sect.link = b.Link
			sect.info = b.Info
			sect.align = uint64(b.Addralign)
			sect.entsize = uint64(b.Entsize)
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
		elfobj.nsymtab = int(elfobj.symtab.size / elf.Sym64Size)
	} else {
		elfobj.nsymtab = int(elfobj.symtab.size / elf.Sym32Size)
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
		if (sect.type_ != elf.SHT_PROGBITS && sect.type_ != elf.SHT_NOBITS) || sect.flags&elf.SHF_ALLOC == 0 {
			continue
		}
		if sect.type_ != elf.SHT_NOBITS {
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

		sb := l.MakeSymbolUpdater(l.LookupOrCreateCgoExport(name, localSymVersion))

		switch sect.flags & (elf.SHF_ALLOC | elf.SHF_WRITE | elf.SHF_EXECINSTR) {
		default:
			return errorf("%s: unexpected flags for ELF section %s", pn, sect.name)

		case elf.SHF_ALLOC:
			sb.SetType(sym.SRODATA)

		case elf.SHF_ALLOC + elf.SHF_WRITE:
			if sect.type_ == elf.SHT_NOBITS {
				sb.SetType(sym.SNOPTRBSS)
			} else {
				sb.SetType(sym.SNOPTRDATA)
			}

		case elf.SHF_ALLOC + elf.SHF_EXECINSTR:
			sb.SetType(sym.STEXT)
		}

		if sect.name == ".got" || sect.name == ".toc" {
			sb.SetType(sym.SELFGOT)
		}
		if sect.type_ == elf.SHT_PROGBITS {
			sb.SetData(sect.base[:sect.size])
			sb.SetExternal(true)
		}

		sb.SetSize(int64(sect.size))
		sb.SetAlign(int32(sect.align))
		sb.SetReadOnly(sect.readOnlyMem)

		sect.sym = sb.Sym()
	}

	// enter sub-symbols into symbol table.
	// symbol 0 is the null symbol.
	symbols := make([]loader.Sym, elfobj.nsymtab)

	for i := 1; i < elfobj.nsymtab; i++ {
		var elfsym ElfSym
		if err := readelfsym(l, arch, elfobj, i, &elfsym, 1, localSymVersion); err != nil {
			return errorf("%s: malformed elf file: %v", pn, err)
		}
		symbols[i] = elfsym.sym
		if elfsym.type_ != elf.STT_FUNC && elfsym.type_ != elf.STT_OBJECT && elfsym.type_ != elf.STT_NOTYPE && elfsym.type_ != elf.STT_COMMON {
			continue
		}
		if elfsym.shndx == elf.SHN_COMMON || elfsym.type_ == elf.STT_COMMON {
			sb := l.MakeSymbolUpdater(elfsym.sym)
			if uint64(sb.Size()) < elfsym.size {
				sb.SetSize(int64(elfsym.size))
			}
			if sb.Type() == 0 || sb.Type() == sym.SXREF {
				sb.SetType(sym.SNOPTRBSS)
			}
			continue
		}

		if uint(elfsym.shndx) >= elfobj.nsect || elfsym.shndx == 0 {
			continue
		}

		// even when we pass needSym == 1 to readelfsym, it might still return nil to skip some unwanted symbols
		if elfsym.sym == 0 {
			continue
		}
		sect = &elfobj.sect[elfsym.shndx]
		if sect.sym == 0 {
			if elfsym.type_ == 0 {
				if strings.HasPrefix(sect.name, ".debug_") && elfsym.name == "" {
					// clang on arm and riscv64.
					// This reportedly happens with clang 3.7 on ARM.
					// See issue 13139.
					continue
				}
				if strings.HasPrefix(elfsym.name, ".Ldebug_") || elfsym.name == ".L0 " {
					// gcc on riscv64.
					continue
				}
				if elfsym.name == ".Lline_table_start0" {
					// clang on riscv64.
					continue
				}

				if strings.HasPrefix(elfsym.name, "$d") && sect.name == ".debug_frame" {
					// "$d" is a marker, not a real symbol.
					// This happens with gcc on ARM64.
					// See https://sourceware.org/bugzilla/show_bug.cgi?id=21809
					continue
				}

				if arch.Family == sys.RISCV64 &&
					(strings.HasPrefix(elfsym.name, "$d") || strings.HasPrefix(elfsym.name, "$x")) {
					// Ignore RISC-V mapping symbols, which
					// are similar to ARM64's case.
					// See issue 73591.
					continue
				}
			}

			if strings.HasPrefix(elfsym.name, ".Linfo_string") {
				// clang does this
				continue
			}

			if strings.HasPrefix(elfsym.name, ".LASF") || strings.HasPrefix(elfsym.name, ".LLRL") || strings.HasPrefix(elfsym.name, ".LLST") || strings.HasPrefix(elfsym.name, ".LVUS") {
				// gcc on s390x and riscv64 does this.
				continue
			}

			return errorf("%v: sym#%d (%q): ignoring symbol in section %d (%q) (type %d)", elfsym.sym, i, elfsym.name, elfsym.shndx, sect.name, elfsym.type_)
		}

		s := elfsym.sym
		if l.OuterSym(s) != 0 {
			if l.AttrDuplicateOK(s) {
				continue
			}
			return errorf("duplicate symbol reference: %s in both %s and %s",
				l.SymName(s), l.SymName(l.OuterSym(s)), l.SymName(sect.sym))
		}

		sectsb := l.MakeSymbolUpdater(sect.sym)
		sb := l.MakeSymbolUpdater(s)

		sb.SetType(sectsb.Type())
		sectsb.AddInteriorSym(s)
		if !l.AttrCgoExportDynamic(s) {
			sb.SetDynimplib("") // satisfy dynimport
		}
		sb.SetValue(int64(elfsym.value))
		sb.SetSize(int64(elfsym.size))
		if sectsb.Type().IsText() {
			if l.AttrExternal(s) && !l.AttrDuplicateOK(s) {
				return errorf("%s: duplicate symbol definition", sb.Name())
			}
			l.SetAttrExternal(s, true)
		}

		if elf.Machine(elfobj.machine) == elf.EM_PPC64 {
			flag := int(elfsym.other) >> 5
			switch flag {
			case 0:
				// No local entry. R2 is preserved.
			case 1:
				// This is kind of a hack, but pass the hint about this symbol's
				// usage of R2 (R2 is a caller-save register not a TOC pointer, and
				// this function does not have a distinct local entry) by setting
				// its SymLocalentry to 1.
				l.SetSymLocalentry(s, 1)
			case 7:
				return errorf("%s: invalid sym.other 0x%x", sb.Name(), elfsym.other)
			default:
				// Convert the word sized offset into bytes.
				l.SetSymLocalentry(s, 4<<uint(flag-2))
			}
		}
	}

	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for i := uint(0); i < elfobj.nsect; i++ {
		s := elfobj.sect[i].sym
		if s == 0 {
			continue
		}
		sb := l.MakeSymbolUpdater(s)
		if l.SubSym(s) != 0 {
			sb.SortSub()
		}
		if sb.Type().IsText() {
			if l.AttrOnList(s) {
				return errorf("symbol %s listed multiple times",
					l.SymName(s))
			}
			l.SetAttrOnList(s, true)
			textp = append(textp, s)
			for ss := l.SubSym(s); ss != 0; ss = l.SubSym(ss) {
				if l.AttrOnList(ss) {
					return errorf("symbol %s listed multiple times",
						l.SymName(ss))
				}
				l.SetAttrOnList(ss, true)
				textp = append(textp, ss)
			}
		}
	}

	// load relocations
	for i := uint(0); i < elfobj.nsect; i++ {
		rsect := &elfobj.sect[i]
		if rsect.type_ != elf.SHT_RELA && rsect.type_ != elf.SHT_REL {
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
		if rsect.type_ == elf.SHT_RELA {
			rela = 1
		}
		n := int(rsect.size / uint64(4+4*is64) / uint64(2+rela))
		p := rsect.base
		sb := l.MakeSymbolUpdater(sect.sym)
		for j := 0; j < n; j++ {
			var add uint64
			var symIdx int
			var relocType uint64
			var rOff int32
			var rAdd int64
			var rSym loader.Sym

			if is64 != 0 {
				// 64-bit rel/rela
				rOff = int32(e.Uint64(p))

				p = p[8:]
				switch arch.Family {
				case sys.MIPS64:
					// https://www.linux-mips.org/pub/linux/mips/doc/ABI/elf64-2.4.pdf
					// The doc shows it's different with general Linux ELF
					symIdx = int(e.Uint32(p))
					relocType = uint64(p[7])
				default:
					info := e.Uint64(p)
					relocType = info & 0xffffffff
					symIdx = int(info >> 32)
				}
				p = p[8:]
				if rela != 0 {
					add = e.Uint64(p)
					p = p[8:]
				}
			} else {
				// 32-bit rel/rela
				rOff = int32(e.Uint32(p))

				p = p[4:]
				info := e.Uint32(p)
				relocType = uint64(info & 0xff)
				symIdx = int(info >> 8)
				p = p[4:]
				if rela != 0 {
					add = uint64(e.Uint32(p))
					p = p[4:]
				}
			}

			if relocType == 0 { // skip R_*_NONE relocation
				j--
				n--
				continue
			}

			if symIdx == 0 { // absolute relocation, don't bother reading the null symbol
				rSym = 0
			} else {
				var elfsym ElfSym
				if err := readelfsym(l, arch, elfobj, int(symIdx), &elfsym, 0, 0); err != nil {
					return errorf("malformed elf file: %v", err)
				}
				elfsym.sym = symbols[symIdx]
				if elfsym.sym == 0 {
					return errorf("malformed elf file: %s#%d: reloc of invalid sym #%d %s shndx=%d type=%d", l.SymName(sect.sym), j, int(symIdx), elfsym.name, elfsym.shndx, elfsym.type_)
				}

				rSym = elfsym.sym
			}

			rType := objabi.ElfRelocOffset + objabi.RelocType(relocType)
			rSize, addendSize, err := relSize(arch, pn, uint32(relocType))
			if err != nil {
				return nil, 0, err
			}
			if rela != 0 {
				rAdd = int64(add)
			} else {
				// load addend from image
				if rSize == 4 {
					rAdd = int64(e.Uint32(sect.base[rOff:]))
				} else if rSize == 8 {
					rAdd = int64(e.Uint64(sect.base[rOff:]))
				} else {
					return errorf("invalid rela size %d", rSize)
				}
			}

			if addendSize == 2 {
				rAdd = int64(int16(rAdd))
			}
			if addendSize == 4 {
				rAdd = int64(int32(rAdd))
			}

			r, _ := sb.AddRel(rType)
			r.SetOff(rOff)
			r.SetSiz(rSize)
			r.SetSym(rSym)
			r.SetAdd(rAdd)
		}

		sb.SortRelocs() // just in case
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

	elfobj.f.MustSeek(int64(uint64(elfobj.base)+sect.off), 0)
	sect.base, sect.readOnlyMem, err = elfobj.f.Slice(uint64(sect.size))
	if err != nil {
		return fmt.Errorf("short read: %v", err)
	}

	return nil
}

func readelfsym(l *loader.Loader, arch *sys.Arch, elfobj *ElfObj, i int, elfsym *ElfSym, needSym int, localSymVersion int) (err error) {
	if i >= elfobj.nsymtab || i < 0 {
		err = fmt.Errorf("invalid elf symbol index")
		return err
	}

	if i == 0 {
		return fmt.Errorf("readym: read null symbol!")
	}

	if elfobj.is64 != 0 {
		b := new(elf.Sym64)
		binary.Read(bytes.NewReader(elfobj.symtab.base[i*elf.Sym64Size:(i+1)*elf.Sym64Size]), elfobj.e, b)
		elfsym.name = cstring(elfobj.symstr.base[b.Name:])
		elfsym.value = b.Value
		elfsym.size = b.Size
		elfsym.shndx = elf.SectionIndex(b.Shndx)
		elfsym.bind = elf.ST_BIND(b.Info)
		elfsym.type_ = elf.ST_TYPE(b.Info)
		elfsym.other = b.Other
	} else {
		b := new(elf.Sym32)
		binary.Read(bytes.NewReader(elfobj.symtab.base[i*elf.Sym32Size:(i+1)*elf.Sym32Size]), elfobj.e, b)
		elfsym.name = cstring(elfobj.symstr.base[b.Name:])
		elfsym.value = uint64(b.Value)
		elfsym.size = uint64(b.Size)
		elfsym.shndx = elf.SectionIndex(b.Shndx)
		elfsym.bind = elf.ST_BIND(b.Info)
		elfsym.type_ = elf.ST_TYPE(b.Info)
		elfsym.other = b.Other
	}

	var s loader.Sym

	if elfsym.name == "_GLOBAL_OFFSET_TABLE_" {
		elfsym.name = ".got"
	}
	if elfsym.name == ".TOC." {
		// Magic symbol on ppc64.  Will be set to this object
		// file's .got+0x8000.
		elfsym.bind = elf.STB_LOCAL
	}

	switch elfsym.type_ {
	case elf.STT_SECTION:
		s = elfobj.sect[elfsym.shndx].sym

	case elf.STT_OBJECT, elf.STT_FUNC, elf.STT_NOTYPE, elf.STT_COMMON:
		switch elfsym.bind {
		case elf.STB_GLOBAL:
			if needSym != 0 {
				s = l.LookupOrCreateCgoExport(elfsym.name, 0)

				// for global scoped hidden symbols we should insert it into
				// symbol hash table, but mark them as hidden.
				// __i686.get_pc_thunk.bx is allowed to be duplicated, to
				// workaround that we set dupok.
				// TODO(minux): correctly handle __i686.get_pc_thunk.bx without
				// set dupok generally. See https://golang.org/cl/5823055
				// comment #5 for details.
				if s != 0 && elfsym.other == 2 {
					if !l.IsExternal(s) {
						l.MakeSymbolUpdater(s)
					}
					l.SetAttrDuplicateOK(s, true)
					l.SetAttrVisibilityHidden(s, true)
				}
			}

		case elf.STB_LOCAL:
			if (arch.Family == sys.ARM || arch.Family == sys.ARM64) && (strings.HasPrefix(elfsym.name, "$a") || strings.HasPrefix(elfsym.name, "$d") || strings.HasPrefix(elfsym.name, "$x")) {
				// binutils for arm and arm64 generate these mapping
				// symbols, ignore these
				break
			}

			if elfsym.name == ".TOC." {
				// We need to be able to look this up,
				// so put it in the hash table.
				if needSym != 0 {
					s = l.LookupOrCreateCgoExport(elfsym.name, localSymVersion)
					l.SetAttrVisibilityHidden(s, true)
				}
				break
			}

			if needSym != 0 {
				// local names and hidden global names are unique
				// and should only be referenced by their index, not name, so we
				// don't bother to add them into the hash table
				// FIXME: pass empty string here for name? This would
				// reduce mem use, but also (possibly) make it harder
				// to debug problems.
				s = l.CreateStaticSym(elfsym.name)
				l.SetAttrVisibilityHidden(s, true)
			}

		case elf.STB_WEAK:
			if needSym != 0 {
				s = l.LookupOrCreateCgoExport(elfsym.name, 0)
				if elfsym.other == 2 {
					l.SetAttrVisibilityHidden(s, true)
				}

				// Allow weak symbols to be duplicated when already defined.
				if l.OuterSym(s) != 0 {
					l.SetAttrDuplicateOK(s, true)
				}
			}

		default:
			err = fmt.Errorf("%s: invalid symbol binding %d", elfsym.name, elfsym.bind)
			return err
		}
	}

	if s != 0 && l.SymType(s) == 0 && elfsym.type_ != elf.STT_SECTION {
		sb := l.MakeSymbolUpdater(s)
		sb.SetType(sym.SXREF)
	}
	elfsym.sym = s

	return nil
}

// Return the size of the relocated field, and the size of the addend as the first
// and second values. Note, the addend may be larger than the relocation field in
// some cases when a relocated value is split across multiple relocations.
func relSize(arch *sys.Arch, pn string, elftype uint32) (uint8, uint8, error) {
	// TODO(mdempsky): Replace this with a struct-valued switch statement
	// once golang.org/issue/15164 is fixed or found to not impair cmd/link
	// performance.

	const (
		AMD64   = uint32(sys.AMD64)
		ARM     = uint32(sys.ARM)
		ARM64   = uint32(sys.ARM64)
		I386    = uint32(sys.I386)
		LOONG64 = uint32(sys.Loong64)
		MIPS    = uint32(sys.MIPS)
		MIPS64  = uint32(sys.MIPS64)
		PPC64   = uint32(sys.PPC64)
		RISCV64 = uint32(sys.RISCV64)
		S390X   = uint32(sys.S390X)
	)

	switch uint32(arch.Family) | elftype<<16 {
	default:
		return 0, 0, fmt.Errorf("%s: unknown relocation type %d; compiled without -fpic?", pn, elftype)

	case MIPS | uint32(elf.R_MIPS_HI16)<<16,
		MIPS | uint32(elf.R_MIPS_LO16)<<16,
		MIPS | uint32(elf.R_MIPS_GOT16)<<16,
		MIPS | uint32(elf.R_MIPS_GOT_HI16)<<16,
		MIPS | uint32(elf.R_MIPS_GOT_LO16)<<16,
		MIPS | uint32(elf.R_MIPS_GPREL16)<<16,
		MIPS | uint32(elf.R_MIPS_GOT_PAGE)<<16,
		MIPS | uint32(elf.R_MIPS_JALR)<<16,
		MIPS | uint32(elf.R_MIPS_GOT_OFST)<<16,
		MIPS64 | uint32(elf.R_MIPS_HI16)<<16,
		MIPS64 | uint32(elf.R_MIPS_LO16)<<16,
		MIPS64 | uint32(elf.R_MIPS_GOT16)<<16,
		MIPS64 | uint32(elf.R_MIPS_GOT_HI16)<<16,
		MIPS64 | uint32(elf.R_MIPS_GOT_LO16)<<16,
		MIPS64 | uint32(elf.R_MIPS_GPREL16)<<16,
		MIPS64 | uint32(elf.R_MIPS_GOT_PAGE)<<16,
		MIPS64 | uint32(elf.R_MIPS_JALR)<<16,
		MIPS64 | uint32(elf.R_MIPS_GOT_OFST)<<16,
		MIPS64 | uint32(elf.R_MIPS_CALL16)<<16,
		MIPS64 | uint32(elf.R_MIPS_GPREL32)<<16,
		MIPS64 | uint32(elf.R_MIPS_64)<<16,
		MIPS64 | uint32(elf.R_MIPS_GOT_DISP)<<16,
		MIPS64 | uint32(elf.R_MIPS_PC32)<<16:
		return 4, 4, nil

	case LOONG64 | uint32(elf.R_LARCH_ADD8)<<16,
		LOONG64 | uint32(elf.R_LARCH_SUB8)<<16:
		return 1, 1, nil

	case LOONG64 | uint32(elf.R_LARCH_ADD16)<<16,
		LOONG64 | uint32(elf.R_LARCH_SUB16)<<16:
		return 2, 2, nil

	case LOONG64 | uint32(elf.R_LARCH_MARK_LA)<<16,
		LOONG64 | uint32(elf.R_LARCH_MARK_PCREL)<<16,
		LOONG64 | uint32(elf.R_LARCH_ADD24)<<16,
		LOONG64 | uint32(elf.R_LARCH_ADD32)<<16,
		LOONG64 | uint32(elf.R_LARCH_SUB24)<<16,
		LOONG64 | uint32(elf.R_LARCH_SUB32)<<16,
		LOONG64 | uint32(elf.R_LARCH_B26)<<16,
		LOONG64 | uint32(elf.R_LARCH_32_PCREL)<<16:
		return 4, 4, nil

	case LOONG64 | uint32(elf.R_LARCH_64)<<16,
		LOONG64 | uint32(elf.R_LARCH_ADD64)<<16,
		LOONG64 | uint32(elf.R_LARCH_SUB64)<<16,
		LOONG64 | uint32(elf.R_LARCH_64_PCREL)<<16:
		return 8, 8, nil

	case S390X | uint32(elf.R_390_8)<<16:
		return 1, 1, nil

	case PPC64 | uint32(elf.R_PPC64_TOC16)<<16,
		S390X | uint32(elf.R_390_16)<<16,
		S390X | uint32(elf.R_390_GOT16)<<16,
		S390X | uint32(elf.R_390_PC16)<<16,
		S390X | uint32(elf.R_390_PC16DBL)<<16,
		S390X | uint32(elf.R_390_PLT16DBL)<<16:
		return 2, 2, nil

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
		ARM64 | uint32(elf.R_AARCH64_LDST16_ABS_LO12_NC)<<16,
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
		PPC64 | uint32(elf.R_PPC64_REL24_NOTOC)<<16,
		PPC64 | uint32(elf.R_PPC64_REL24_P9NOTOC)<<16,
		PPC64 | uint32(elf.R_PPC_REL32)<<16,
		S390X | uint32(elf.R_390_32)<<16,
		S390X | uint32(elf.R_390_PC32)<<16,
		S390X | uint32(elf.R_390_GOT32)<<16,
		S390X | uint32(elf.R_390_PLT32)<<16,
		S390X | uint32(elf.R_390_PC32DBL)<<16,
		S390X | uint32(elf.R_390_PLT32DBL)<<16,
		S390X | uint32(elf.R_390_GOTPCDBL)<<16,
		S390X | uint32(elf.R_390_GOTENT)<<16:
		return 4, 4, nil

	case AMD64 | uint32(elf.R_X86_64_64)<<16,
		AMD64 | uint32(elf.R_X86_64_PC64)<<16,
		ARM64 | uint32(elf.R_AARCH64_ABS64)<<16,
		ARM64 | uint32(elf.R_AARCH64_PREL64)<<16,
		PPC64 | uint32(elf.R_PPC64_ADDR64)<<16,
		PPC64 | uint32(elf.R_PPC64_PCREL34)<<16,
		PPC64 | uint32(elf.R_PPC64_GOT_PCREL34)<<16,
		PPC64 | uint32(elf.R_PPC64_PLT_PCREL34_NOTOC)<<16,
		S390X | uint32(elf.R_390_GLOB_DAT)<<16,
		S390X | uint32(elf.R_390_RELATIVE)<<16,
		S390X | uint32(elf.R_390_GOTOFF)<<16,
		S390X | uint32(elf.R_390_GOTPC)<<16,
		S390X | uint32(elf.R_390_64)<<16,
		S390X | uint32(elf.R_390_PC64)<<16,
		S390X | uint32(elf.R_390_GOT64)<<16,
		S390X | uint32(elf.R_390_PLT64)<<16:
		return 8, 8, nil

	case RISCV64 | uint32(elf.R_RISCV_SET6)<<16,
		RISCV64 | uint32(elf.R_RISCV_SUB6)<<16,
		RISCV64 | uint32(elf.R_RISCV_SET8)<<16,
		RISCV64 | uint32(elf.R_RISCV_SUB8)<<16:
		return 1, 1, nil

	case RISCV64 | uint32(elf.R_RISCV_RVC_BRANCH)<<16,
		RISCV64 | uint32(elf.R_RISCV_RVC_JUMP)<<16,
		RISCV64 | uint32(elf.R_RISCV_SET16)<<16,
		RISCV64 | uint32(elf.R_RISCV_SUB16)<<16:
		return 2, 2, nil

	case RISCV64 | uint32(elf.R_RISCV_32)<<16,
		RISCV64 | uint32(elf.R_RISCV_BRANCH)<<16,
		RISCV64 | uint32(elf.R_RISCV_HI20)<<16,
		RISCV64 | uint32(elf.R_RISCV_LO12_I)<<16,
		RISCV64 | uint32(elf.R_RISCV_LO12_S)<<16,
		RISCV64 | uint32(elf.R_RISCV_GOT_HI20)<<16,
		RISCV64 | uint32(elf.R_RISCV_PCREL_HI20)<<16,
		RISCV64 | uint32(elf.R_RISCV_PCREL_LO12_I)<<16,
		RISCV64 | uint32(elf.R_RISCV_PCREL_LO12_S)<<16,
		RISCV64 | uint32(elf.R_RISCV_ADD32)<<16,
		RISCV64 | uint32(elf.R_RISCV_SET32)<<16,
		RISCV64 | uint32(elf.R_RISCV_SUB32)<<16,
		RISCV64 | uint32(elf.R_RISCV_32_PCREL)<<16,
		RISCV64 | uint32(elf.R_RISCV_RELAX)<<16:
		return 4, 4, nil

	case RISCV64 | uint32(elf.R_RISCV_64)<<16,
		RISCV64 | uint32(elf.R_RISCV_CALL)<<16,
		RISCV64 | uint32(elf.R_RISCV_CALL_PLT)<<16:
		return 8, 8, nil

	case PPC64 | uint32(elf.R_PPC64_TOC16_LO)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_HI)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_HA)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_DS)<<16,
		PPC64 | uint32(elf.R_PPC64_TOC16_LO_DS)<<16,
		PPC64 | uint32(elf.R_PPC64_REL16_LO)<<16,
		PPC64 | uint32(elf.R_PPC64_REL16_HI)<<16,
		PPC64 | uint32(elf.R_PPC64_REL16_HA)<<16,
		PPC64 | uint32(elf.R_PPC64_PLT16_HA)<<16,
		PPC64 | uint32(elf.R_PPC64_PLT16_LO_DS)<<16:
		return 2, 4, nil

	// PPC64 inline PLT sequence hint relocations (-fno-plt)
	// These are informational annotations to assist linker optimizations.
	case PPC64 | uint32(elf.R_PPC64_PLTSEQ)<<16,
		PPC64 | uint32(elf.R_PPC64_PLTCALL)<<16,
		PPC64 | uint32(elf.R_PPC64_PLTCALL_NOTOC)<<16,
		PPC64 | uint32(elf.R_PPC64_PLTSEQ_NOTOC)<<16:
		return 0, 0, nil

	}
}

func cstring(x []byte) string {
	i := bytes.IndexByte(x, '\x00')
	if i >= 0 {
		x = x[:i]
	}
	return string(x)
}
