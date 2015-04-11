// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"sort"
	"strings"
)

type MachoHdr struct {
	cpu    uint32
	subcpu uint32
}

type MachoSect struct {
	name    string
	segname string
	addr    uint64
	size    uint64
	off     uint32
	align   uint32
	reloc   uint32
	nreloc  uint32
	flag    uint32
	res1    uint32
	res2    uint32
}

type MachoSeg struct {
	name       string
	vsize      uint64
	vaddr      uint64
	fileoffset uint64
	filesize   uint64
	prot1      uint32
	prot2      uint32
	nsect      uint32
	msect      uint32
	sect       []MachoSect
	flag       uint32
}

type MachoLoad struct {
	type_ uint32
	data  []uint32
}

/*
 * Total amount of space to reserve at the start of the file
 * for Header, PHeaders, and SHeaders.
 * May waste some.
 */
const (
	INITIAL_MACHO_HEADR = 4 * 1024
)

const (
	MACHO_CPU_AMD64               = 1<<24 | 7
	MACHO_CPU_386                 = 7
	MACHO_SUBCPU_X86              = 3
	MACHO_CPU_ARM                 = 12
	MACHO_SUBCPU_ARM              = 0
	MACHO_SUBCPU_ARMV7            = 9
	MACHO_CPU_ARM64               = 1<<24 | 12
	MACHO_SUBCPU_ARM64_ALL        = 0
	MACHO32SYMSIZE                = 12
	MACHO64SYMSIZE                = 16
	MACHO_X86_64_RELOC_UNSIGNED   = 0
	MACHO_X86_64_RELOC_SIGNED     = 1
	MACHO_X86_64_RELOC_BRANCH     = 2
	MACHO_X86_64_RELOC_GOT_LOAD   = 3
	MACHO_X86_64_RELOC_GOT        = 4
	MACHO_X86_64_RELOC_SUBTRACTOR = 5
	MACHO_X86_64_RELOC_SIGNED_1   = 6
	MACHO_X86_64_RELOC_SIGNED_2   = 7
	MACHO_X86_64_RELOC_SIGNED_4   = 8
	MACHO_ARM_RELOC_VANILLA       = 0
	MACHO_ARM_RELOC_BR24          = 5
	MACHO_ARM64_RELOC_UNSIGNED    = 0
	MACHO_ARM64_RELOC_BRANCH26    = 2
	MACHO_ARM64_RELOC_PAGE21      = 3
	MACHO_ARM64_RELOC_PAGEOFF12   = 4
	MACHO_ARM64_RELOC_ADDEND      = 10
	MACHO_GENERIC_RELOC_VANILLA   = 0
	MACHO_FAKE_GOTPCREL           = 100
)

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mach-O file writing
// http://developer.apple.com/mac/library/DOCUMENTATION/DeveloperTools/Conceptual/MachORuntime/Reference/reference.html

var macho64 bool

var machohdr MachoHdr

var load []MachoLoad

var seg [16]MachoSeg

var nseg int

var ndebug int

var nsect int

const (
	SymKindLocal = 0 + iota
	SymKindExtdef
	SymKindUndef
	NumSymKind
)

var nkind [NumSymKind]int

var sortsym []*LSym

var nsortsym int

// Amount of space left for adding load commands
// that refer to dynamic libraries.  Because these have
// to go in the Mach-O header, we can't just pick a
// "big enough" header size.  The initial header is
// one page, the non-dynamic library stuff takes
// up about 1300 bytes; we overestimate that as 2k.
var load_budget int = INITIAL_MACHO_HEADR - 2*1024

func Machoinit() {
	switch Thearch.Thechar {
	// 64-bit architectures
	case '6', '7', '9':
		macho64 = true

		// 32-bit architectures
	default:
		break
	}
}

func getMachoHdr() *MachoHdr {
	return &machohdr
}

func newMachoLoad(type_ uint32, ndata uint32) *MachoLoad {
	if macho64 && (ndata&1 != 0) {
		ndata++
	}

	load = append(load, MachoLoad{})
	l := &load[len(load)-1]
	l.type_ = type_
	l.data = make([]uint32, ndata)
	return l
}

func newMachoSeg(name string, msect int) *MachoSeg {
	if nseg >= len(seg) {
		Diag("too many segs")
		Errorexit()
	}

	s := &seg[nseg]
	nseg++
	s.name = name
	s.msect = uint32(msect)
	s.sect = make([]MachoSect, msect)
	return s
}

func newMachoSect(seg *MachoSeg, name string, segname string) *MachoSect {
	if seg.nsect >= seg.msect {
		Diag("too many sects in segment %s", seg.name)
		Errorexit()
	}

	s := &seg.sect[seg.nsect]
	seg.nsect++
	s.name = name
	s.segname = segname
	nsect++
	return s
}

// Generic linking code.

var dylib []string

var linkoff int64

func machowrite() int {
	o1 := Cpos()

	loadsize := 4 * 4 * ndebug
	for i := 0; i < len(load); i++ {
		loadsize += 4 * (len(load[i].data) + 2)
	}
	if macho64 {
		loadsize += 18 * 4 * nseg
		loadsize += 20 * 4 * nsect
	} else {
		loadsize += 14 * 4 * nseg
		loadsize += 17 * 4 * nsect
	}

	if macho64 {
		Thearch.Lput(0xfeedfacf)
	} else {
		Thearch.Lput(0xfeedface)
	}
	Thearch.Lput(machohdr.cpu)
	Thearch.Lput(machohdr.subcpu)
	if Linkmode == LinkExternal {
		Thearch.Lput(1) /* file type - mach object */
	} else {
		Thearch.Lput(2) /* file type - mach executable */
	}
	Thearch.Lput(uint32(len(load)) + uint32(nseg) + uint32(ndebug))
	Thearch.Lput(uint32(loadsize))
	Thearch.Lput(1) /* flags - no undefines */
	if macho64 {
		Thearch.Lput(0) /* reserved */
	}

	var j int
	var s *MachoSeg
	var t *MachoSect
	for i := 0; i < nseg; i++ {
		s = &seg[i]
		if macho64 {
			Thearch.Lput(25) /* segment 64 */
			Thearch.Lput(72 + 80*s.nsect)
			strnput(s.name, 16)
			Thearch.Vput(s.vaddr)
			Thearch.Vput(s.vsize)
			Thearch.Vput(s.fileoffset)
			Thearch.Vput(s.filesize)
			Thearch.Lput(s.prot1)
			Thearch.Lput(s.prot2)
			Thearch.Lput(s.nsect)
			Thearch.Lput(s.flag)
		} else {
			Thearch.Lput(1) /* segment 32 */
			Thearch.Lput(56 + 68*s.nsect)
			strnput(s.name, 16)
			Thearch.Lput(uint32(s.vaddr))
			Thearch.Lput(uint32(s.vsize))
			Thearch.Lput(uint32(s.fileoffset))
			Thearch.Lput(uint32(s.filesize))
			Thearch.Lput(s.prot1)
			Thearch.Lput(s.prot2)
			Thearch.Lput(s.nsect)
			Thearch.Lput(s.flag)
		}

		for j = 0; uint32(j) < s.nsect; j++ {
			t = &s.sect[j]
			if macho64 {
				strnput(t.name, 16)
				strnput(t.segname, 16)
				Thearch.Vput(t.addr)
				Thearch.Vput(t.size)
				Thearch.Lput(t.off)
				Thearch.Lput(t.align)
				Thearch.Lput(t.reloc)
				Thearch.Lput(t.nreloc)
				Thearch.Lput(t.flag)
				Thearch.Lput(t.res1) /* reserved */
				Thearch.Lput(t.res2) /* reserved */
				Thearch.Lput(0)      /* reserved */
			} else {
				strnput(t.name, 16)
				strnput(t.segname, 16)
				Thearch.Lput(uint32(t.addr))
				Thearch.Lput(uint32(t.size))
				Thearch.Lput(t.off)
				Thearch.Lput(t.align)
				Thearch.Lput(t.reloc)
				Thearch.Lput(t.nreloc)
				Thearch.Lput(t.flag)
				Thearch.Lput(t.res1) /* reserved */
				Thearch.Lput(t.res2) /* reserved */
			}
		}
	}

	var l *MachoLoad
	for i := 0; i < len(load); i++ {
		l = &load[i]
		Thearch.Lput(l.type_)
		Thearch.Lput(4 * (uint32(len(l.data)) + 2))
		for j = 0; j < len(l.data); j++ {
			Thearch.Lput(l.data[j])
		}
	}

	return int(Cpos() - o1)
}

func domacho() {
	if Debug['d'] != 0 {
		return
	}

	// empirically, string table must begin with " \x00".
	s := Linklookup(Ctxt, ".machosymstr", 0)

	s.Type = SMACHOSYMSTR
	s.Reachable = true
	Adduint8(Ctxt, s, ' ')
	Adduint8(Ctxt, s, '\x00')

	s = Linklookup(Ctxt, ".machosymtab", 0)
	s.Type = SMACHOSYMTAB
	s.Reachable = true

	if Linkmode != LinkExternal {
		s := Linklookup(Ctxt, ".plt", 0) // will be __symbol_stub
		s.Type = SMACHOPLT
		s.Reachable = true

		s = Linklookup(Ctxt, ".got", 0) // will be __nl_symbol_ptr
		s.Type = SMACHOGOT
		s.Reachable = true
		s.Align = 4

		s = Linklookup(Ctxt, ".linkedit.plt", 0) // indirect table for .plt
		s.Type = SMACHOINDIRECTPLT
		s.Reachable = true

		s = Linklookup(Ctxt, ".linkedit.got", 0) // indirect table for .got
		s.Type = SMACHOINDIRECTGOT
		s.Reachable = true
	}
}

func Machoadddynlib(lib string) {
	// Will need to store the library name rounded up
	// and 24 bytes of header metadata.  If not enough
	// space, grab another page of initial space at the
	// beginning of the output file.
	load_budget -= (len(lib)+7)/8*8 + 24

	if load_budget < 0 {
		HEADR += 4096
		INITTEXT += 4096
		load_budget += 4096
	}

	dylib = append(dylib, lib)
}

func machoshbits(mseg *MachoSeg, sect *Section, segname string) {
	buf := "__" + strings.Replace(sect.Name[1:], ".", "_", -1)

	var msect *MachoSect
	if Thearch.Thechar == '7' && sect.Rwx&1 == 0 {
		// darwin/arm64 forbids absolute relocs in __TEXT, so if
		// the section is not executable, put it in __DATA segment.
		msect = newMachoSect(mseg, buf, "__DATA")
	} else {
		msect = newMachoSect(mseg, buf, segname)
	}

	if sect.Rellen > 0 {
		msect.reloc = uint32(sect.Reloff)
		msect.nreloc = uint32(sect.Rellen / 8)
	}

	for 1<<msect.align < sect.Align {
		msect.align++
	}
	msect.addr = sect.Vaddr
	msect.size = sect.Length

	if sect.Vaddr < sect.Seg.Vaddr+sect.Seg.Filelen {
		// data in file
		if sect.Length > sect.Seg.Vaddr+sect.Seg.Filelen-sect.Vaddr {
			Diag("macho cannot represent section %s crossing data and bss", sect.Name)
		}
		msect.off = uint32(sect.Seg.Fileoff + sect.Vaddr - sect.Seg.Vaddr)
	} else {
		// zero fill
		msect.off = 0

		msect.flag |= 1
	}

	if sect.Rwx&1 != 0 {
		msect.flag |= 0x400 /* has instructions */
	}

	if sect.Name == ".plt" {
		msect.name = "__symbol_stub1"
		msect.flag = 0x80000408 /* only instructions, code, symbol stubs */
		msect.res1 = 0          //nkind[SymKindLocal];
		msect.res2 = 6
	}

	if sect.Name == ".got" {
		msect.name = "__nl_symbol_ptr"
		msect.flag = 6                                                     /* section with nonlazy symbol pointers */
		msect.res1 = uint32(Linklookup(Ctxt, ".linkedit.plt", 0).Size / 4) /* offset into indirect symbol table */
	}

	if sect.Name == ".init_array" {
		msect.name = "__mod_init_func"
		msect.flag = 9 // S_MOD_INIT_FUNC_POINTERS
	}
}

func Asmbmacho() {
	/* apple MACH */
	va := INITTEXT - int64(HEADR)

	mh := getMachoHdr()
	switch Thearch.Thechar {
	default:
		Diag("unknown mach architecture")
		Errorexit()
		fallthrough

	case '5':
		mh.cpu = MACHO_CPU_ARM
		mh.subcpu = MACHO_SUBCPU_ARMV7

	case '6':
		mh.cpu = MACHO_CPU_AMD64
		mh.subcpu = MACHO_SUBCPU_X86

	case '7':
		mh.cpu = MACHO_CPU_ARM64
		mh.subcpu = MACHO_SUBCPU_ARM64_ALL

	case '8':
		mh.cpu = MACHO_CPU_386
		mh.subcpu = MACHO_SUBCPU_X86
	}

	var ms *MachoSeg
	if Linkmode == LinkExternal {
		/* segment for entire file */
		ms = newMachoSeg("", 40)

		ms.fileoffset = Segtext.Fileoff
		ms.filesize = Segdata.Fileoff + Segdata.Filelen - Segtext.Fileoff
	}

	/* segment for zero page */
	if Linkmode != LinkExternal {
		ms = newMachoSeg("__PAGEZERO", 0)
		ms.vsize = uint64(va)
	}

	/* text */
	v := Rnd(int64(uint64(HEADR)+Segtext.Length), int64(INITRND))

	if Linkmode != LinkExternal {
		ms = newMachoSeg("__TEXT", 20)
		ms.vaddr = uint64(va)
		ms.vsize = uint64(v)
		ms.fileoffset = 0
		ms.filesize = uint64(v)
		ms.prot1 = 7
		ms.prot2 = 5
	}

	for sect := Segtext.Sect; sect != nil; sect = sect.Next {
		machoshbits(ms, sect, "__TEXT")
	}

	/* data */
	if Linkmode != LinkExternal {
		w := int64(Segdata.Length)
		ms = newMachoSeg("__DATA", 20)
		ms.vaddr = uint64(va) + uint64(v)
		ms.vsize = uint64(w)
		ms.fileoffset = uint64(v)
		ms.filesize = Segdata.Filelen
		ms.prot1 = 3
		ms.prot2 = 3
	}

	for sect := Segdata.Sect; sect != nil; sect = sect.Next {
		machoshbits(ms, sect, "__DATA")
	}

	if Linkmode != LinkExternal {
		switch Thearch.Thechar {
		default:
			Diag("unknown macho architecture")
			Errorexit()
			fallthrough

		case '5':
			ml := newMachoLoad(5, 17+2)          /* unix thread */
			ml.data[0] = 1                       /* thread type */
			ml.data[1] = 17                      /* word count */
			ml.data[2+15] = uint32(Entryvalue()) /* start pc */

		case '6':
			ml := newMachoLoad(5, 42+2)          /* unix thread */
			ml.data[0] = 4                       /* thread type */
			ml.data[1] = 42                      /* word count */
			ml.data[2+32] = uint32(Entryvalue()) /* start pc */
			ml.data[2+32+1] = uint32(Entryvalue() >> 32)

		case '7':
			ml := newMachoLoad(5, 68+2)          /* unix thread */
			ml.data[0] = 6                       /* thread type */
			ml.data[1] = 68                      /* word count */
			ml.data[2+64] = uint32(Entryvalue()) /* start pc */
			ml.data[2+64+1] = uint32(Entryvalue() >> 32)

		case '8':
			ml := newMachoLoad(5, 16+2)          /* unix thread */
			ml.data[0] = 1                       /* thread type */
			ml.data[1] = 16                      /* word count */
			ml.data[2+10] = uint32(Entryvalue()) /* start pc */
		}
	}

	if Debug['d'] == 0 {
		// must match domacholink below
		s1 := Linklookup(Ctxt, ".machosymtab", 0)

		s2 := Linklookup(Ctxt, ".linkedit.plt", 0)
		s3 := Linklookup(Ctxt, ".linkedit.got", 0)
		s4 := Linklookup(Ctxt, ".machosymstr", 0)

		if Linkmode != LinkExternal {
			ms := newMachoSeg("__LINKEDIT", 0)
			ms.vaddr = uint64(va) + uint64(v) + uint64(Rnd(int64(Segdata.Length), int64(INITRND)))
			ms.vsize = uint64(s1.Size) + uint64(s2.Size) + uint64(s3.Size) + uint64(s4.Size)
			ms.fileoffset = uint64(linkoff)
			ms.filesize = ms.vsize
			ms.prot1 = 7
			ms.prot2 = 3
		}

		ml := newMachoLoad(2, 4)                                   /* LC_SYMTAB */
		ml.data[0] = uint32(linkoff)                               /* symoff */
		ml.data[1] = uint32(nsortsym)                              /* nsyms */
		ml.data[2] = uint32(linkoff + s1.Size + s2.Size + s3.Size) /* stroff */
		ml.data[3] = uint32(s4.Size)                               /* strsize */

		machodysymtab()

		if Linkmode != LinkExternal {
			ml := newMachoLoad(14, 6) /* LC_LOAD_DYLINKER */
			ml.data[0] = 12           /* offset to string */
			stringtouint32(ml.data[1:], "/usr/lib/dyld")

			for i := 0; i < len(dylib); i++ {
				ml = newMachoLoad(12, 4+(uint32(len(dylib[i]))+1+7)/8*2) /* LC_LOAD_DYLIB */
				ml.data[0] = 24                                          /* offset of string from beginning of load */
				ml.data[1] = 0                                           /* time stamp */
				ml.data[2] = 0                                           /* version */
				ml.data[3] = 0                                           /* compatibility version */
				stringtouint32(ml.data[4:], dylib[i])
			}
		}
	}

	// TODO: dwarf headers go in ms too
	if Debug['s'] == 0 && Linkmode != LinkExternal {
		dwarfaddmachoheaders()
	}

	a := machowrite()
	if int32(a) > HEADR {
		Diag("HEADR too small: %d > %d", a, HEADR)
	}
}

func symkind(s *LSym) int {
	if s.Type == SDYNIMPORT {
		return SymKindUndef
	}
	if s.Cgoexport != 0 {
		return SymKindExtdef
	}
	return SymKindLocal
}

func addsym(s *LSym, name string, type_ int, addr int64, size int64, ver int, gotype *LSym) {
	if s == nil {
		return
	}

	switch type_ {
	default:
		return

	case 'D', 'B', 'T':
		break
	}

	if sortsym != nil {
		sortsym[nsortsym] = s
		nkind[symkind(s)]++
	}

	nsortsym++
}

type machoscmp []*LSym

func (x machoscmp) Len() int {
	return len(x)
}

func (x machoscmp) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}

func (x machoscmp) Less(i, j int) bool {
	s1 := x[i]
	s2 := x[j]

	k1 := symkind(s1)
	k2 := symkind(s2)
	if k1 != k2 {
		return k1-k2 < 0
	}

	return stringsCompare(s1.Extname, s2.Extname) < 0
}

func machogenasmsym(put func(*LSym, string, int, int64, int64, int, *LSym)) {
	genasmsym(put)
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if s.Type == SDYNIMPORT || s.Type == SHOSTOBJ {
			if s.Reachable {
				put(s, "", 'D', 0, 0, 0, nil)
			}
		}
	}
}

func machosymorder() {
	// On Mac OS X Mountain Lion, we must sort exported symbols
	// So we sort them here and pre-allocate dynid for them
	// See http://golang.org/issue/4029
	for i := 0; i < len(dynexp); i++ {
		dynexp[i].Reachable = true
	}
	machogenasmsym(addsym)
	sortsym = make([]*LSym, nsortsym)
	nsortsym = 0
	machogenasmsym(addsym)
	sort.Sort(machoscmp(sortsym[:nsortsym]))
	for i := 0; i < nsortsym; i++ {
		sortsym[i].Dynid = int32(i)
	}
}

func machosymtab() {
	var s *LSym
	var o *LSym
	var p string

	symtab := Linklookup(Ctxt, ".machosymtab", 0)
	symstr := Linklookup(Ctxt, ".machosymstr", 0)

	for i := 0; i < nsortsym; i++ {
		s = sortsym[i]
		Adduint32(Ctxt, symtab, uint32(symstr.Size))

		// Only add _ to C symbols. Go symbols have dot in the name.
		if !strings.Contains(s.Extname, ".") {
			Adduint8(Ctxt, symstr, '_')
		}

		// replace "·" as ".", because DTrace cannot handle it.
		if !strings.Contains(s.Extname, "·") {
			Addstring(symstr, s.Extname)
		} else {
			for p = s.Extname; p != ""; p = p[1:] {
				if uint8(p[0]) == 0xc2 && uint8((p[1:])[0]) == 0xb7 {
					Adduint8(Ctxt, symstr, '.')
					p = p[1:]
				} else {
					Adduint8(Ctxt, symstr, uint8(p[0]))
				}
			}

			Adduint8(Ctxt, symstr, '\x00')
		}

		if s.Type == SDYNIMPORT || s.Type == SHOSTOBJ {
			Adduint8(Ctxt, symtab, 0x01)                // type N_EXT, external symbol
			Adduint8(Ctxt, symtab, 0)                   // no section
			Adduint16(Ctxt, symtab, 0)                  // desc
			adduintxx(Ctxt, symtab, 0, Thearch.Ptrsize) // no value
		} else {
			if s.Cgoexport != 0 {
				Adduint8(Ctxt, symtab, 0x0f)
			} else {
				Adduint8(Ctxt, symtab, 0x0e)
			}
			o = s
			for o.Outer != nil {
				o = o.Outer
			}
			if o.Sect == nil {
				Diag("missing section for %s", s.Name)
				Adduint8(Ctxt, symtab, 0)
			} else {
				Adduint8(Ctxt, symtab, uint8((o.Sect.(*Section)).Extnum))
			}
			Adduint16(Ctxt, symtab, 0) // desc
			adduintxx(Ctxt, symtab, uint64(Symaddr(s)), Thearch.Ptrsize)
		}
	}
}

func machodysymtab() {
	ml := newMachoLoad(11, 18) /* LC_DYSYMTAB */

	n := 0
	ml.data[0] = uint32(n)                   /* ilocalsym */
	ml.data[1] = uint32(nkind[SymKindLocal]) /* nlocalsym */
	n += nkind[SymKindLocal]

	ml.data[2] = uint32(n)                    /* iextdefsym */
	ml.data[3] = uint32(nkind[SymKindExtdef]) /* nextdefsym */
	n += nkind[SymKindExtdef]

	ml.data[4] = uint32(n)                   /* iundefsym */
	ml.data[5] = uint32(nkind[SymKindUndef]) /* nundefsym */

	ml.data[6] = 0  /* tocoffset */
	ml.data[7] = 0  /* ntoc */
	ml.data[8] = 0  /* modtaboff */
	ml.data[9] = 0  /* nmodtab */
	ml.data[10] = 0 /* extrefsymoff */
	ml.data[11] = 0 /* nextrefsyms */

	// must match domacholink below
	s1 := Linklookup(Ctxt, ".machosymtab", 0)

	s2 := Linklookup(Ctxt, ".linkedit.plt", 0)
	s3 := Linklookup(Ctxt, ".linkedit.got", 0)
	ml.data[12] = uint32(linkoff + s1.Size)       /* indirectsymoff */
	ml.data[13] = uint32((s2.Size + s3.Size) / 4) /* nindirectsyms */

	ml.data[14] = 0 /* extreloff */
	ml.data[15] = 0 /* nextrel */
	ml.data[16] = 0 /* locreloff */
	ml.data[17] = 0 /* nlocrel */
}

func Domacholink() int64 {
	machosymtab()

	// write data that will be linkedit section
	s1 := Linklookup(Ctxt, ".machosymtab", 0)

	s2 := Linklookup(Ctxt, ".linkedit.plt", 0)
	s3 := Linklookup(Ctxt, ".linkedit.got", 0)
	s4 := Linklookup(Ctxt, ".machosymstr", 0)

	// Force the linkedit section to end on a 16-byte
	// boundary.  This allows pure (non-cgo) Go binaries
	// to be code signed correctly.
	//
	// Apple's codesign_allocate (a helper utility for
	// the codesign utility) can do this fine itself if
	// it is run on a dynamic Mach-O binary.  However,
	// when it is run on a pure (non-cgo) Go binary, where
	// the linkedit section is mostly empty, it fails to
	// account for the extra padding that it itself adds
	// when adding the LC_CODE_SIGNATURE load command
	// (which must be aligned on a 16-byte boundary).
	//
	// By forcing the linkedit section to end on a 16-byte
	// boundary, codesign_allocate will not need to apply
	// any alignment padding itself, working around the
	// issue.
	for s4.Size%16 != 0 {
		Adduint8(Ctxt, s4, 0)
	}

	size := int(s1.Size + s2.Size + s3.Size + s4.Size)

	if size > 0 {
		linkoff = Rnd(int64(uint64(HEADR)+Segtext.Length), int64(INITRND)) + Rnd(int64(Segdata.Filelen), int64(INITRND)) + Rnd(int64(Segdwarf.Filelen), int64(INITRND))
		Cseek(linkoff)

		Cwrite(s1.P[:s1.Size])
		Cwrite(s2.P[:s2.Size])
		Cwrite(s3.P[:s3.Size])
		Cwrite(s4.P[:s4.Size])
	}

	return Rnd(int64(size), int64(INITRND))
}

func machorelocsect(sect *Section, first *LSym) {
	// If main section has no bits, nothing to relocate.
	if sect.Vaddr >= sect.Seg.Vaddr+sect.Seg.Filelen {
		return
	}

	sect.Reloff = uint64(Cpos())
	var sym *LSym
	for sym = first; sym != nil; sym = sym.Next {
		if !sym.Reachable {
			continue
		}
		if uint64(sym.Value) >= sect.Vaddr {
			break
		}
	}

	eaddr := int32(sect.Vaddr + sect.Length)
	var r *Reloc
	var ri int
	for ; sym != nil; sym = sym.Next {
		if !sym.Reachable {
			continue
		}
		if sym.Value >= int64(eaddr) {
			break
		}
		Ctxt.Cursym = sym

		for ri = 0; ri < len(sym.R); ri++ {
			r = &sym.R[ri]
			if r.Done != 0 {
				continue
			}
			if Thearch.Machoreloc1(r, int64(uint64(sym.Value+int64(r.Off))-sect.Vaddr)) < 0 {
				Diag("unsupported obj reloc %d/%d to %s", r.Type, r.Siz, r.Sym.Name)
			}
		}
	}

	sect.Rellen = uint64(Cpos()) - sect.Reloff
}

func Machoemitreloc() {
	for Cpos()&7 != 0 {
		Cput(0)
	}

	machorelocsect(Segtext.Sect, Ctxt.Textp)
	for sect := Segtext.Sect.Next; sect != nil; sect = sect.Next {
		machorelocsect(sect, datap)
	}
	for sect := Segdata.Sect; sect != nil; sect = sect.Next {
		machorelocsect(sect, datap)
	}
}
