// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"encoding/binary"
)

// Temporary dumping around for sym.Symbol version of helper
// functions in elf.go, still being used for some archs/oses.
// FIXME: get rid of this file when dodata() is completely
// converted and the sym.Symbol functions are not needed.

func elfsetstring(s *sym.Symbol, str string, off int) {
	if nelfstr >= len(elfstr) {
		Errorf(s, "too many elf strings")
		errorexit()
	}

	elfstr[nelfstr].s = str
	elfstr[nelfstr].off = off
	nelfstr++
}

func Asmbelf2(ctxt *Link, symo int64) {
	eh := getElfEhdr()
	switch ctxt.Arch.Family {
	default:
		Exitf("unknown architecture in asmbelf: %v", ctxt.Arch.Family)
	case sys.MIPS, sys.MIPS64:
		eh.machine = EM_MIPS
	case sys.ARM:
		eh.machine = EM_ARM
	case sys.AMD64:
		eh.machine = EM_X86_64
	case sys.ARM64:
		eh.machine = EM_AARCH64
	case sys.I386:
		eh.machine = EM_386
	case sys.PPC64:
		eh.machine = EM_PPC64
	case sys.RISCV64:
		eh.machine = EM_RISCV
	case sys.S390X:
		eh.machine = EM_S390
	}

	elfreserve := int64(ELFRESERVE)

	numtext := int64(0)
	for _, sect := range Segtext.Sections {
		if sect.Name == ".text" {
			numtext++
		}
	}

	// If there are multiple text sections, extra space is needed
	// in the elfreserve for the additional .text and .rela.text
	// section headers.  It can handle 4 extra now. Headers are
	// 64 bytes.

	if numtext > 4 {
		elfreserve += elfreserve + numtext*64*2
	}

	startva := *FlagTextAddr - int64(HEADR)
	resoff := elfreserve

	var pph *ElfPhdr
	var pnote *ElfPhdr
	if *flagRace && ctxt.IsNetbsd() {
		sh := elfshname(".note.netbsd.pax")
		resoff -= int64(elfnetbsdpax(sh, uint64(startva), uint64(resoff)))
		pnote = newElfPhdr()
		pnote.type_ = PT_NOTE
		pnote.flags = PF_R
		phsh(pnote, sh)
	}
	if ctxt.LinkMode == LinkExternal {
		/* skip program headers */
		eh.phoff = 0

		eh.phentsize = 0

		if ctxt.BuildMode == BuildModeShared {
			sh := elfshname(".note.go.pkg-list")
			sh.type_ = SHT_NOTE
			sh = elfshname(".note.go.abihash")
			sh.type_ = SHT_NOTE
			sh.flags = SHF_ALLOC
			sh = elfshname(".note.go.deps")
			sh.type_ = SHT_NOTE
		}

		if *flagBuildid != "" {
			sh := elfshname(".note.go.buildid")
			sh.type_ = SHT_NOTE
			sh.flags = SHF_ALLOC
		}

		goto elfobj
	}

	/* program header info */
	pph = newElfPhdr()

	pph.type_ = PT_PHDR
	pph.flags = PF_R
	pph.off = uint64(eh.ehsize)
	pph.vaddr = uint64(*FlagTextAddr) - uint64(HEADR) + pph.off
	pph.paddr = uint64(*FlagTextAddr) - uint64(HEADR) + pph.off
	pph.align = uint64(*FlagRound)

	/*
	 * PHDR must be in a loaded segment. Adjust the text
	 * segment boundaries downwards to include it.
	 */
	{
		o := int64(Segtext.Vaddr - pph.vaddr)
		Segtext.Vaddr -= uint64(o)
		Segtext.Length += uint64(o)
		o = int64(Segtext.Fileoff - pph.off)
		Segtext.Fileoff -= uint64(o)
		Segtext.Filelen += uint64(o)
	}

	if !*FlagD { /* -d suppresses dynamic loader format */
		/* interpreter */
		sh := elfshname(".interp")

		sh.type_ = SHT_PROGBITS
		sh.flags = SHF_ALLOC
		sh.addralign = 1

		if interpreter == "" && objabi.GO_LDSO != "" {
			interpreter = objabi.GO_LDSO
		}

		if interpreter == "" {
			switch ctxt.HeadType {
			case objabi.Hlinux:
				if objabi.GOOS == "android" {
					interpreter = thearch.Androiddynld
					if interpreter == "" {
						Exitf("ELF interpreter not set")
					}
				} else {
					interpreter = thearch.Linuxdynld
				}

			case objabi.Hfreebsd:
				interpreter = thearch.Freebsddynld

			case objabi.Hnetbsd:
				interpreter = thearch.Netbsddynld

			case objabi.Hopenbsd:
				interpreter = thearch.Openbsddynld

			case objabi.Hdragonfly:
				interpreter = thearch.Dragonflydynld

			case objabi.Hsolaris:
				interpreter = thearch.Solarisdynld
			}
		}

		resoff -= int64(elfinterp(sh, uint64(startva), uint64(resoff), interpreter))

		ph := newElfPhdr()
		ph.type_ = PT_INTERP
		ph.flags = PF_R
		phsh(ph, sh)
	}

	pnote = nil
	if ctxt.HeadType == objabi.Hnetbsd || ctxt.HeadType == objabi.Hopenbsd {
		var sh *ElfShdr
		switch ctxt.HeadType {
		case objabi.Hnetbsd:
			sh = elfshname(".note.netbsd.ident")
			resoff -= int64(elfnetbsdsig(sh, uint64(startva), uint64(resoff)))

		case objabi.Hopenbsd:
			sh = elfshname(".note.openbsd.ident")
			resoff -= int64(elfopenbsdsig(sh, uint64(startva), uint64(resoff)))
		}

		pnote = newElfPhdr()
		pnote.type_ = PT_NOTE
		pnote.flags = PF_R
		phsh(pnote, sh)
	}

	if len(buildinfo) > 0 {
		sh := elfshname(".note.gnu.build-id")
		resoff -= int64(elfbuildinfo(sh, uint64(startva), uint64(resoff)))

		if pnote == nil {
			pnote = newElfPhdr()
			pnote.type_ = PT_NOTE
			pnote.flags = PF_R
		}

		phsh(pnote, sh)
	}

	if *flagBuildid != "" {
		sh := elfshname(".note.go.buildid")
		resoff -= int64(elfgobuildid(sh, uint64(startva), uint64(resoff)))

		pnote := newElfPhdr()
		pnote.type_ = PT_NOTE
		pnote.flags = PF_R
		phsh(pnote, sh)
	}

	// Additions to the reserved area must be above this line.

	elfphload(&Segtext)
	if len(Segrodata.Sections) > 0 {
		elfphload(&Segrodata)
	}
	if len(Segrelrodata.Sections) > 0 {
		elfphload(&Segrelrodata)
		elfphrelro(&Segrelrodata)
	}
	elfphload(&Segdata)

	/* Dynamic linking sections */
	if !*FlagD {
		sh := elfshname(".dynsym")
		sh.type_ = SHT_DYNSYM
		sh.flags = SHF_ALLOC
		if elf64 {
			sh.entsize = ELF64SYMSIZE
		} else {
			sh.entsize = ELF32SYMSIZE
		}
		sh.addralign = uint64(ctxt.Arch.RegSize)
		sh.link = uint32(elfshname(".dynstr").shnum)

		// sh.info is the index of first non-local symbol (number of local symbols)
		s := ctxt.Syms.Lookup(".dynsym", 0)
		i := uint32(0)
		for sub := s; sub != nil; sub = symSub(ctxt, sub) {
			i++
			if !sub.Attr.Local() {
				break
			}
		}
		sh.info = i
		shsym(sh, s)

		sh = elfshname(".dynstr")
		sh.type_ = SHT_STRTAB
		sh.flags = SHF_ALLOC
		sh.addralign = 1
		shsym(sh, ctxt.Syms.Lookup(".dynstr", 0))

		if elfverneed != 0 {
			sh := elfshname(".gnu.version")
			sh.type_ = SHT_GNU_VERSYM
			sh.flags = SHF_ALLOC
			sh.addralign = 2
			sh.link = uint32(elfshname(".dynsym").shnum)
			sh.entsize = 2
			shsym(sh, ctxt.Syms.Lookup(".gnu.version", 0))

			sh = elfshname(".gnu.version_r")
			sh.type_ = SHT_GNU_VERNEED
			sh.flags = SHF_ALLOC
			sh.addralign = uint64(ctxt.Arch.RegSize)
			sh.info = uint32(elfverneed)
			sh.link = uint32(elfshname(".dynstr").shnum)
			shsym(sh, ctxt.Syms.Lookup(".gnu.version_r", 0))
		}

		if elfRelType == ".rela" {
			sh := elfshname(".rela.plt")
			sh.type_ = SHT_RELA
			sh.flags = SHF_ALLOC
			sh.entsize = ELF64RELASIZE
			sh.addralign = uint64(ctxt.Arch.RegSize)
			sh.link = uint32(elfshname(".dynsym").shnum)
			sh.info = uint32(elfshname(".plt").shnum)
			shsym(sh, ctxt.Syms.Lookup(".rela.plt", 0))

			sh = elfshname(".rela")
			sh.type_ = SHT_RELA
			sh.flags = SHF_ALLOC
			sh.entsize = ELF64RELASIZE
			sh.addralign = 8
			sh.link = uint32(elfshname(".dynsym").shnum)
			shsym(sh, ctxt.Syms.Lookup(".rela", 0))
		} else {
			sh := elfshname(".rel.plt")
			sh.type_ = SHT_REL
			sh.flags = SHF_ALLOC
			sh.entsize = ELF32RELSIZE
			sh.addralign = 4
			sh.link = uint32(elfshname(".dynsym").shnum)
			shsym(sh, ctxt.Syms.Lookup(".rel.plt", 0))

			sh = elfshname(".rel")
			sh.type_ = SHT_REL
			sh.flags = SHF_ALLOC
			sh.entsize = ELF32RELSIZE
			sh.addralign = 4
			sh.link = uint32(elfshname(".dynsym").shnum)
			shsym(sh, ctxt.Syms.Lookup(".rel", 0))
		}

		if eh.machine == EM_PPC64 {
			sh := elfshname(".glink")
			sh.type_ = SHT_PROGBITS
			sh.flags = SHF_ALLOC + SHF_EXECINSTR
			sh.addralign = 4
			shsym(sh, ctxt.Syms.Lookup(".glink", 0))
		}

		sh = elfshname(".plt")
		sh.type_ = SHT_PROGBITS
		sh.flags = SHF_ALLOC + SHF_EXECINSTR
		if eh.machine == EM_X86_64 {
			sh.entsize = 16
		} else if eh.machine == EM_S390 {
			sh.entsize = 32
		} else if eh.machine == EM_PPC64 {
			// On ppc64, this is just a table of addresses
			// filled by the dynamic linker
			sh.type_ = SHT_NOBITS

			sh.flags = SHF_ALLOC + SHF_WRITE
			sh.entsize = 8
		} else {
			sh.entsize = 4
		}
		sh.addralign = sh.entsize
		shsym(sh, ctxt.Syms.Lookup(".plt", 0))

		// On ppc64, .got comes from the input files, so don't
		// create it here, and .got.plt is not used.
		if eh.machine != EM_PPC64 {
			sh := elfshname(".got")
			sh.type_ = SHT_PROGBITS
			sh.flags = SHF_ALLOC + SHF_WRITE
			sh.entsize = uint64(ctxt.Arch.RegSize)
			sh.addralign = uint64(ctxt.Arch.RegSize)
			shsym(sh, ctxt.Syms.Lookup(".got", 0))

			sh = elfshname(".got.plt")
			sh.type_ = SHT_PROGBITS
			sh.flags = SHF_ALLOC + SHF_WRITE
			sh.entsize = uint64(ctxt.Arch.RegSize)
			sh.addralign = uint64(ctxt.Arch.RegSize)
			shsym(sh, ctxt.Syms.Lookup(".got.plt", 0))
		}

		sh = elfshname(".hash")
		sh.type_ = SHT_HASH
		sh.flags = SHF_ALLOC
		sh.entsize = 4
		sh.addralign = uint64(ctxt.Arch.RegSize)
		sh.link = uint32(elfshname(".dynsym").shnum)
		shsym(sh, ctxt.Syms.Lookup(".hash", 0))

		/* sh and PT_DYNAMIC for .dynamic section */
		sh = elfshname(".dynamic")

		sh.type_ = SHT_DYNAMIC
		sh.flags = SHF_ALLOC + SHF_WRITE
		sh.entsize = 2 * uint64(ctxt.Arch.RegSize)
		sh.addralign = uint64(ctxt.Arch.RegSize)
		sh.link = uint32(elfshname(".dynstr").shnum)
		shsym(sh, ctxt.Syms.Lookup(".dynamic", 0))
		ph := newElfPhdr()
		ph.type_ = PT_DYNAMIC
		ph.flags = PF_R + PF_W
		phsh(ph, sh)

		/*
		 * Thread-local storage segment (really just size).
		 */
		tlssize := uint64(0)
		for _, sect := range Segdata.Sections {
			if sect.Name == ".tbss" {
				tlssize = sect.Length
			}
		}
		if tlssize != 0 {
			ph := newElfPhdr()
			ph.type_ = PT_TLS
			ph.flags = PF_R
			ph.memsz = tlssize
			ph.align = uint64(ctxt.Arch.RegSize)
		}
	}

	if ctxt.HeadType == objabi.Hlinux {
		ph := newElfPhdr()
		ph.type_ = PT_GNU_STACK
		ph.flags = PF_W + PF_R
		ph.align = uint64(ctxt.Arch.RegSize)

		ph = newElfPhdr()
		ph.type_ = PT_PAX_FLAGS
		ph.flags = 0x2a00 // mprotect, randexec, emutramp disabled
		ph.align = uint64(ctxt.Arch.RegSize)
	} else if ctxt.HeadType == objabi.Hsolaris {
		ph := newElfPhdr()
		ph.type_ = PT_SUNWSTACK
		ph.flags = PF_W + PF_R
	}

elfobj:
	sh := elfshname(".shstrtab")
	sh.type_ = SHT_STRTAB
	sh.addralign = 1
	shsym(sh, ctxt.Syms.Lookup(".shstrtab", 0))
	eh.shstrndx = uint16(sh.shnum)

	// put these sections early in the list
	if !*FlagS {
		elfshname(".symtab")
		elfshname(".strtab")
	}

	for _, sect := range Segtext.Sections {
		elfshbits(ctxt.LinkMode, sect)
	}
	for _, sect := range Segrodata.Sections {
		elfshbits(ctxt.LinkMode, sect)
	}
	for _, sect := range Segrelrodata.Sections {
		elfshbits(ctxt.LinkMode, sect)
	}
	for _, sect := range Segdata.Sections {
		elfshbits(ctxt.LinkMode, sect)
	}
	for _, sect := range Segdwarf.Sections {
		elfshbits(ctxt.LinkMode, sect)
	}

	if ctxt.LinkMode == LinkExternal {
		for _, sect := range Segtext.Sections {
			elfshreloc(ctxt.Arch, sect)
		}
		for _, sect := range Segrodata.Sections {
			elfshreloc(ctxt.Arch, sect)
		}
		for _, sect := range Segrelrodata.Sections {
			elfshreloc(ctxt.Arch, sect)
		}
		for _, sect := range Segdata.Sections {
			elfshreloc(ctxt.Arch, sect)
		}
		for _, si := range dwarfp {
			s := si.secSym()
			elfshreloc(ctxt.Arch, s.Sect)
		}
		// add a .note.GNU-stack section to mark the stack as non-executable
		sh := elfshname(".note.GNU-stack")

		sh.type_ = SHT_PROGBITS
		sh.addralign = 1
		sh.flags = 0
	}

	if !*FlagS {
		sh := elfshname(".symtab")
		sh.type_ = SHT_SYMTAB
		sh.off = uint64(symo)
		sh.size = uint64(Symsize)
		sh.addralign = uint64(ctxt.Arch.RegSize)
		sh.entsize = 8 + 2*uint64(ctxt.Arch.RegSize)
		sh.link = uint32(elfshname(".strtab").shnum)
		sh.info = uint32(elfglobalsymndx)

		sh = elfshname(".strtab")
		sh.type_ = SHT_STRTAB
		sh.off = uint64(symo) + uint64(Symsize)
		sh.size = uint64(len(Elfstrdat))
		sh.addralign = 1
	}

	/* Main header */
	eh.ident[EI_MAG0] = '\177'

	eh.ident[EI_MAG1] = 'E'
	eh.ident[EI_MAG2] = 'L'
	eh.ident[EI_MAG3] = 'F'
	if ctxt.HeadType == objabi.Hfreebsd {
		eh.ident[EI_OSABI] = ELFOSABI_FREEBSD
	} else if ctxt.HeadType == objabi.Hnetbsd {
		eh.ident[EI_OSABI] = ELFOSABI_NETBSD
	} else if ctxt.HeadType == objabi.Hopenbsd {
		eh.ident[EI_OSABI] = ELFOSABI_OPENBSD
	} else if ctxt.HeadType == objabi.Hdragonfly {
		eh.ident[EI_OSABI] = ELFOSABI_NONE
	}
	if elf64 {
		eh.ident[EI_CLASS] = ELFCLASS64
	} else {
		eh.ident[EI_CLASS] = ELFCLASS32
	}
	if ctxt.Arch.ByteOrder == binary.BigEndian {
		eh.ident[EI_DATA] = ELFDATA2MSB
	} else {
		eh.ident[EI_DATA] = ELFDATA2LSB
	}
	eh.ident[EI_VERSION] = EV_CURRENT

	if ctxt.LinkMode == LinkExternal {
		eh.type_ = ET_REL
	} else if ctxt.BuildMode == BuildModePIE {
		eh.type_ = ET_DYN
	} else {
		eh.type_ = ET_EXEC
	}

	if ctxt.LinkMode != LinkExternal {
		eh.entry = uint64(Entryvalue(ctxt))
	}

	eh.version = EV_CURRENT

	if pph != nil {
		pph.filesz = uint64(eh.phnum) * uint64(eh.phentsize)
		pph.memsz = pph.filesz
	}

	ctxt.Out.SeekSet(0)
	a := int64(0)
	a += int64(elfwritehdr(ctxt.Out))
	a += int64(elfwritephdrs(ctxt.Out))
	a += int64(elfwriteshdrs(ctxt.Out))
	if !*FlagD {
		a += int64(elfwriteinterp(ctxt.Out))
	}
	if ctxt.LinkMode != LinkExternal {
		if ctxt.HeadType == objabi.Hnetbsd {
			a += int64(elfwritenetbsdsig(ctxt.Out))
		}
		if ctxt.HeadType == objabi.Hopenbsd {
			a += int64(elfwriteopenbsdsig(ctxt.Out))
		}
		if len(buildinfo) > 0 {
			a += int64(elfwritebuildinfo(ctxt.Out))
		}
		if *flagBuildid != "" {
			a += int64(elfwritegobuildid(ctxt.Out))
		}
	}
	if *flagRace && ctxt.IsNetbsd() {
		a += int64(elfwritenetbsdpax(ctxt.Out))
	}

	if a > elfreserve {
		Errorf(nil, "ELFRESERVE too small: %d > %d with %d text sections", a, elfreserve, numtext)
	}
}

// Do not write DT_NULL.  elfdynhash will finish it.
func shsym(sh *ElfShdr, s *sym.Symbol) {
	addr := Symaddr(s)
	if sh.flags&SHF_ALLOC != 0 {
		sh.addr = uint64(addr)
	}
	sh.off = uint64(datoff2(s, addr))
	sh.size = uint64(s.Size)
}

func Elfemitreloc2(ctxt *Link) {
	for ctxt.Out.Offset()&7 != 0 {
		ctxt.Out.Write8(0)
	}

	for _, sect := range Segtext.Sections {
		if sect.Name == ".text" {
			elfrelocsect2(ctxt, sect, ctxt.Textp)
		} else {
			elfrelocsect2(ctxt, sect, ctxt.datap)
		}
	}

	for _, sect := range Segrodata.Sections {
		elfrelocsect2(ctxt, sect, ctxt.datap)
	}
	for _, sect := range Segrelrodata.Sections {
		elfrelocsect2(ctxt, sect, ctxt.datap)
	}
	for _, sect := range Segdata.Sections {
		elfrelocsect2(ctxt, sect, ctxt.datap)
	}
	for i := 0; i < len(Segdwarf.Sections); i++ {
		sect := Segdwarf.Sections[i]
		si := dwarfp[i]
		if si.secSym() != sect.Sym ||
			si.secSym().Sect != sect {
			panic("inconsistency between dwarfp and Segdwarf")
		}
		elfrelocsect2(ctxt, sect, si.syms)
	}
}

func elfrelocsect2(ctxt *Link, sect *sym.Section, syms []*sym.Symbol) {
	// If main section is SHT_NOBITS, nothing to relocate.
	// Also nothing to relocate in .shstrtab.
	if sect.Vaddr >= sect.Seg.Vaddr+sect.Seg.Filelen {
		return
	}
	if sect.Name == ".shstrtab" {
		return
	}

	sect.Reloff = uint64(ctxt.Out.Offset())
	for i, s := range syms {
		if !s.Attr.Reachable() {
			continue
		}
		if uint64(s.Value) >= sect.Vaddr {
			syms = syms[i:]
			break
		}
	}

	eaddr := int32(sect.Vaddr + sect.Length)
	for _, s := range syms {
		if !s.Attr.Reachable() {
			continue
		}
		if s.Value >= int64(eaddr) {
			break
		}
		for ri := range s.R {
			r := &s.R[ri]
			if r.Done {
				continue
			}
			if r.Xsym == nil {
				Errorf(s, "missing xsym in relocation %#v %#v", r.Sym.Name, s)
				continue
			}
			esr := ElfSymForReloc(ctxt, r.Xsym)
			if esr == 0 {
				Errorf(s, "reloc %d (%s) to non-elf symbol %s (outer=%s) %d (%s)", r.Type, sym.RelocName(ctxt.Arch, r.Type), r.Sym.Name, r.Xsym.Name, r.Sym.Type, r.Sym.Type)
			}
			if !r.Xsym.Attr.Reachable() {
				Errorf(s, "unreachable reloc %d (%s) target %v", r.Type, sym.RelocName(ctxt.Arch, r.Type), r.Xsym.Name)
			}
			if !thearch.Elfreloc1(ctxt, r, int64(uint64(s.Value+int64(r.Off))-sect.Vaddr)) {
				Errorf(s, "unsupported obj reloc %d (%s)/%d to %s", r.Type, sym.RelocName(ctxt.Arch, r.Type), r.Siz, r.Sym.Name)
			}
		}
	}

	sect.Rellen = uint64(ctxt.Out.Offset()) - sect.Reloff
}
