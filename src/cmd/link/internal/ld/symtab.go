// Inferno utils/6l/span.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/span.c
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

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"fmt"
	"path/filepath"
	"strings"
)

// Symbol table.

func putelfstr(s string) int {
	if len(Elfstrdat) == 0 && s != "" {
		// first entry must be empty string
		putelfstr("")
	}

	off := len(Elfstrdat)
	Elfstrdat = append(Elfstrdat, s...)
	Elfstrdat = append(Elfstrdat, 0)
	return off
}

func putelfsyment(off int, addr int64, size int64, info int, shndx int, other int) {
	if elf64 {
		Thearch.Lput(uint32(off))
		Cput(uint8(info))
		Cput(uint8(other))
		Thearch.Wput(uint16(shndx))
		Thearch.Vput(uint64(addr))
		Thearch.Vput(uint64(size))
		Symsize += ELF64SYMSIZE
	} else {
		Thearch.Lput(uint32(off))
		Thearch.Lput(uint32(addr))
		Thearch.Lput(uint32(size))
		Cput(uint8(info))
		Cput(uint8(other))
		Thearch.Wput(uint16(shndx))
		Symsize += ELF32SYMSIZE
	}
}

var numelfsym int = 1 // 0 is reserved

var elfbind int

func putelfsym(ctxt *Link, x *Symbol, s string, t SymbolType, addr int64, go_ *Symbol) {
	var typ int

	switch t {
	default:
		return

	case TextSym:
		typ = STT_FUNC

	case DataSym, BSSSym:
		typ = STT_OBJECT

	case UndefinedSym:
		// ElfType is only set for symbols read from Go shared libraries, but
		// for other symbols it is left as STT_NOTYPE which is fine.
		typ = int(x.ElfType)

	case TLSSym:
		typ = STT_TLS
	}

	size := x.Size
	if t == UndefinedSym {
		size = 0
	}

	xo := x
	for xo.Outer != nil {
		xo = xo.Outer
	}

	var elfshnum int
	if xo.Type == SDYNIMPORT || xo.Type == SHOSTOBJ {
		elfshnum = SHN_UNDEF
	} else {
		if xo.Sect == nil {
			Errorf(x, "missing section in putelfsym")
			return
		}
		if xo.Sect.Elfsect == nil {
			Errorf(x, "missing ELF section in putelfsym")
			return
		}
		elfshnum = xo.Sect.Elfsect.shnum
	}

	// One pass for each binding: STB_LOCAL, STB_GLOBAL,
	// maybe one day STB_WEAK.
	bind := STB_GLOBAL

	if x.Version != 0 || (x.Type&SHIDDEN != 0) || x.Attr.Local() {
		bind = STB_LOCAL
	}

	// In external linking mode, we have to invoke gcc with -rdynamic
	// to get the exported symbols put into the dynamic symbol table.
	// To avoid filling the dynamic table with lots of unnecessary symbols,
	// mark all Go symbols local (not global) in the final executable.
	// But when we're dynamically linking, we need all those global symbols.
	if !ctxt.DynlinkingGo() && Linkmode == LinkExternal && !x.Attr.CgoExportStatic() && elfshnum != SHN_UNDEF {
		bind = STB_LOCAL
	}

	if Linkmode == LinkExternal && elfshnum != SHN_UNDEF {
		addr -= int64(xo.Sect.Vaddr)
	}
	other := STV_DEFAULT
	if x.Type&SHIDDEN != 0 {
		other = STV_HIDDEN
	}
	if SysArch.Family == sys.PPC64 && typ == STT_FUNC && x.Attr.Shared() && x.Name != "runtime.duffzero" && x.Name != "runtime.duffcopy" {
		// On ppc64 the top three bits of the st_other field indicate how
		// many instructions separate the global and local entry points. In
		// our case it is two instructions, indicated by the value 3.
		// The conditions here match those in preprocess in
		// cmd/internal/obj/ppc64/obj9.go, which is where the
		// instructions are inserted.
		other |= 3 << 5
	}

	// When dynamically linking, we create Symbols by reading the names from
	// the symbol tables of the shared libraries and so the names need to
	// match exactly. Tools like DTrace will have to wait for now.
	if !ctxt.DynlinkingGo() {
		// Rewrite · to . for ASCII-only tools like DTrace (sigh)
		s = strings.Replace(s, "·", ".", -1)
	}

	if ctxt.DynlinkingGo() && bind == STB_GLOBAL && elfbind == STB_LOCAL && x.Type == STEXT {
		// When dynamically linking, we want references to functions defined
		// in this module to always be to the function object, not to the
		// PLT. We force this by writing an additional local symbol for every
		// global function symbol and making all relocations against the
		// global symbol refer to this local symbol instead (see
		// (*Symbol).ElfsymForReloc). This is approximately equivalent to the
		// ELF linker -Bsymbolic-functions option, but that is buggy on
		// several platforms.
		putelfsyment(putelfstr("local."+s), addr, size, STB_LOCAL<<4|typ&0xf, elfshnum, other)
		x.LocalElfsym = int32(numelfsym)
		numelfsym++
		return
	} else if bind != elfbind {
		return
	}

	putelfsyment(putelfstr(s), addr, size, bind<<4|typ&0xf, elfshnum, other)
	x.Elfsym = int32(numelfsym)
	numelfsym++
}

func putelfsectionsym(s *Symbol, shndx int) {
	putelfsyment(0, 0, 0, STB_LOCAL<<4|STT_SECTION, shndx, 0)
	s.Elfsym = int32(numelfsym)
	numelfsym++
}

func Asmelfsym(ctxt *Link) {
	// the first symbol entry is reserved
	putelfsyment(0, 0, 0, STB_LOCAL<<4|STT_NOTYPE, 0, 0)

	dwarfaddelfsectionsyms(ctxt)

	// Some linkers will add a FILE sym if one is not present.
	// Avoid having the working directory inserted into the symbol table.
	// It is added with a name to avoid problems with external linking
	// encountered on some versions of Solaris. See issue #14957.
	putelfsyment(putelfstr("go.go"), 0, 0, STB_LOCAL<<4|STT_FILE, SHN_ABS, 0)
	numelfsym++

	elfbind = STB_LOCAL
	genasmsym(ctxt, putelfsym)

	elfbind = STB_GLOBAL
	elfglobalsymndx = numelfsym
	genasmsym(ctxt, putelfsym)
}

func putplan9sym(ctxt *Link, x *Symbol, s string, typ SymbolType, addr int64, go_ *Symbol) {
	t := int(typ)
	switch typ {
	case TextSym, DataSym, BSSSym:
		if x.Version != 0 {
			t += 'a' - 'A'
		}
		fallthrough

	case AutoSym, ParamSym, FileSym, FrameSym:
		l := 4
		if Headtype == objabi.Hplan9 && SysArch.Family == sys.AMD64 && !Flag8 {
			Lputb(uint32(addr >> 32))
			l = 8
		}

		Lputb(uint32(addr))
		Cput(uint8(t + 0x80)) /* 0x80 is variable length */

		var i int

		/* skip the '<' in filenames */
		if t == FileSym {
			s = s[1:]
		}
		for i = 0; i < len(s); i++ {
			Cput(s[i])
		}
		Cput(0)

		Symsize += int32(l) + 1 + int32(i) + 1

	default:
		return
	}
}

func Asmplan9sym(ctxt *Link) {
	genasmsym(ctxt, putplan9sym)
}

var symt *Symbol

var encbuf [10]byte

func Wputb(w uint16) { Cwrite(Append16b(encbuf[:0], w)) }
func Lputb(l uint32) { Cwrite(Append32b(encbuf[:0], l)) }
func Vputb(v uint64) { Cwrite(Append64b(encbuf[:0], v)) }

func Wputl(w uint16) { Cwrite(Append16l(encbuf[:0], w)) }
func Lputl(l uint32) { Cwrite(Append32l(encbuf[:0], l)) }
func Vputl(v uint64) { Cwrite(Append64l(encbuf[:0], v)) }

func Append16b(b []byte, v uint16) []byte {
	return append(b, uint8(v>>8), uint8(v))
}
func Append16l(b []byte, v uint16) []byte {
	return append(b, uint8(v), uint8(v>>8))
}

func Append32b(b []byte, v uint32) []byte {
	return append(b, uint8(v>>24), uint8(v>>16), uint8(v>>8), uint8(v))
}
func Append32l(b []byte, v uint32) []byte {
	return append(b, uint8(v), uint8(v>>8), uint8(v>>16), uint8(v>>24))
}

func Append64b(b []byte, v uint64) []byte {
	return append(b, uint8(v>>56), uint8(v>>48), uint8(v>>40), uint8(v>>32),
		uint8(v>>24), uint8(v>>16), uint8(v>>8), uint8(v))
}

func Append64l(b []byte, v uint64) []byte {
	return append(b, uint8(v), uint8(v>>8), uint8(v>>16), uint8(v>>24),
		uint8(v>>32), uint8(v>>40), uint8(v>>48), uint8(v>>56))
}

type byPkg []*Library

func (libs byPkg) Len() int {
	return len(libs)
}

func (libs byPkg) Less(a, b int) bool {
	return libs[a].Pkg < libs[b].Pkg
}

func (libs byPkg) Swap(a, b int) {
	libs[a], libs[b] = libs[b], libs[a]
}

// Create a table with information on the text sections.

func textsectionmap(ctxt *Link) uint32 {

	t := ctxt.Syms.Lookup("runtime.textsectionmap", 0)
	t.Type = SRODATA
	t.Attr |= AttrReachable
	nsections := int64(0)

	for _, sect := range Segtext.Sections {
		if sect.Name == ".text" {
			nsections++
		} else {
			break
		}
	}
	Symgrow(t, 3*nsections*int64(SysArch.PtrSize))

	off := int64(0)
	n := 0

	// The vaddr for each text section is the difference between the section's
	// Vaddr and the Vaddr for the first text section as determined at compile
	// time.

	// The symbol for the first text section is named runtime.text as before.
	// Additional text sections are named runtime.text.n where n is the
	// order of creation starting with 1. These symbols provide the section's
	// address after relocation by the linker.

	textbase := Segtext.Sections[0].Vaddr
	for _, sect := range Segtext.Sections {
		if sect.Name != ".text" {
			break
		}
		off = setuint(ctxt, t, off, sect.Vaddr-textbase)
		off = setuint(ctxt, t, off, sect.Length)
		if n == 0 {
			s := ctxt.Syms.ROLookup("runtime.text", 0)
			if s == nil {
				Errorf(nil, "Unable to find symbol runtime.text\n")
			}
			off = setaddr(ctxt, t, off, s)

		} else {
			s := ctxt.Syms.Lookup(fmt.Sprintf("runtime.text.%d", n), 0)
			if s == nil {
				Errorf(nil, "Unable to find symbol runtime.text.%d\n", n)
			}
			off = setaddr(ctxt, t, off, s)
		}
		n++
	}
	return uint32(n)
}

func (ctxt *Link) symtab() {
	dosymtype(ctxt)

	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	ctxt.xdefine("runtime.text", STEXT, 0)

	ctxt.xdefine("runtime.etext", STEXT, 0)
	ctxt.xdefine("runtime.itablink", SRODATA, 0)
	ctxt.xdefine("runtime.eitablink", SRODATA, 0)
	ctxt.xdefine("runtime.rodata", SRODATA, 0)
	ctxt.xdefine("runtime.erodata", SRODATA, 0)
	ctxt.xdefine("runtime.types", SRODATA, 0)
	ctxt.xdefine("runtime.etypes", SRODATA, 0)
	ctxt.xdefine("runtime.noptrdata", SNOPTRDATA, 0)
	ctxt.xdefine("runtime.enoptrdata", SNOPTRDATA, 0)
	ctxt.xdefine("runtime.data", SDATA, 0)
	ctxt.xdefine("runtime.edata", SDATA, 0)
	ctxt.xdefine("runtime.bss", SBSS, 0)
	ctxt.xdefine("runtime.ebss", SBSS, 0)
	ctxt.xdefine("runtime.noptrbss", SNOPTRBSS, 0)
	ctxt.xdefine("runtime.enoptrbss", SNOPTRBSS, 0)
	ctxt.xdefine("runtime.end", SBSS, 0)
	ctxt.xdefine("runtime.epclntab", SRODATA, 0)
	ctxt.xdefine("runtime.esymtab", SRODATA, 0)

	// garbage collection symbols
	s := ctxt.Syms.Lookup("runtime.gcdata", 0)

	s.Type = SRODATA
	s.Size = 0
	s.Attr |= AttrReachable
	ctxt.xdefine("runtime.egcdata", SRODATA, 0)

	s = ctxt.Syms.Lookup("runtime.gcbss", 0)
	s.Type = SRODATA
	s.Size = 0
	s.Attr |= AttrReachable
	ctxt.xdefine("runtime.egcbss", SRODATA, 0)

	// pseudo-symbols to mark locations of type, string, and go string data.
	var symtype *Symbol
	var symtyperel *Symbol
	if UseRelro() && (Buildmode == BuildmodeCArchive || Buildmode == BuildmodeCShared || Buildmode == BuildmodePIE) {
		s = ctxt.Syms.Lookup("type.*", 0)

		s.Type = STYPE
		s.Size = 0
		s.Attr |= AttrReachable
		symtype = s

		s = ctxt.Syms.Lookup("typerel.*", 0)

		s.Type = STYPERELRO
		s.Size = 0
		s.Attr |= AttrReachable
		symtyperel = s
	} else if !ctxt.DynlinkingGo() {
		s = ctxt.Syms.Lookup("type.*", 0)

		s.Type = STYPE
		s.Size = 0
		s.Attr |= AttrReachable
		symtype = s
		symtyperel = s
	}

	groupSym := func(name string, t SymKind) *Symbol {
		s := ctxt.Syms.Lookup(name, 0)
		s.Type = t
		s.Size = 0
		s.Attr |= AttrLocal | AttrReachable
		return s
	}
	var (
		symgostring = groupSym("go.string.*", SGOSTRING)
		symgofunc   = groupSym("go.func.*", SGOFUNC)
		symgcbits   = groupSym("runtime.gcbits.*", SGCBITS)
	)

	var symgofuncrel *Symbol
	if !ctxt.DynlinkingGo() {
		if UseRelro() {
			symgofuncrel = groupSym("go.funcrel.*", SGOFUNCRELRO)
		} else {
			symgofuncrel = symgofunc
		}
	}

	symitablink := ctxt.Syms.Lookup("runtime.itablink", 0)
	symitablink.Type = SITABLINK

	symt = ctxt.Syms.Lookup("runtime.symtab", 0)
	symt.Attr |= AttrLocal
	symt.Type = SSYMTAB
	symt.Size = 0
	symt.Attr |= AttrReachable

	nitablinks := 0

	// assign specific types so that they sort together.
	// within a type they sort by size, so the .* symbols
	// just defined above will be first.
	// hide the specific symbols.
	for _, s := range ctxt.Syms.Allsym {
		if !s.Attr.Reachable() || s.Attr.Special() || s.Type != SRODATA {
			continue
		}

		switch {
		case strings.HasPrefix(s.Name, "type."):
			if !ctxt.DynlinkingGo() {
				s.Attr |= AttrNotInSymbolTable
			}
			if UseRelro() {
				s.Type = STYPERELRO
				s.Outer = symtyperel
			} else {
				s.Type = STYPE
				s.Outer = symtype
			}

		case strings.HasPrefix(s.Name, "go.importpath.") && UseRelro():
			// Keep go.importpath symbols in the same section as types and
			// names, as they can be referred to by a section offset.
			s.Type = STYPERELRO

		case strings.HasPrefix(s.Name, "go.itablink."):
			nitablinks++
			s.Type = SITABLINK
			s.Attr |= AttrNotInSymbolTable
			s.Outer = symitablink

		case strings.HasPrefix(s.Name, "go.string."):
			s.Type = SGOSTRING
			s.Attr |= AttrNotInSymbolTable
			s.Outer = symgostring

		case strings.HasPrefix(s.Name, "runtime.gcbits."):
			s.Type = SGCBITS
			s.Attr |= AttrNotInSymbolTable
			s.Outer = symgcbits

		case strings.HasSuffix(s.Name, "·f"):
			if !ctxt.DynlinkingGo() {
				s.Attr |= AttrNotInSymbolTable
			}
			if UseRelro() {
				s.Type = SGOFUNCRELRO
				s.Outer = symgofuncrel
			} else {
				s.Type = SGOFUNC
				s.Outer = symgofunc
			}

		case strings.HasPrefix(s.Name, "gcargs."),
			strings.HasPrefix(s.Name, "gclocals."),
			strings.HasPrefix(s.Name, "gclocals·"),
			strings.HasPrefix(s.Name, "inltree."):
			s.Type = SGOFUNC
			s.Attr |= AttrNotInSymbolTable
			s.Outer = symgofunc
			s.Align = 4
			liveness += (s.Size + int64(s.Align) - 1) &^ (int64(s.Align) - 1)
		}
	}

	if Buildmode == BuildmodeShared {
		abihashgostr := ctxt.Syms.Lookup("go.link.abihash."+filepath.Base(*flagOutfile), 0)
		abihashgostr.Attr |= AttrReachable
		abihashgostr.Type = SRODATA
		hashsym := ctxt.Syms.Lookup("go.link.abihashbytes", 0)
		Addaddr(ctxt, abihashgostr, hashsym)
		adduint(ctxt, abihashgostr, uint64(hashsym.Size))
	}
	if Buildmode == BuildmodePlugin || ctxt.Syms.ROLookup("plugin.Open", 0) != nil {
		for _, l := range ctxt.Library {
			s := ctxt.Syms.Lookup("go.link.pkghashbytes."+l.Pkg, 0)
			s.Attr |= AttrReachable
			s.Type = SRODATA
			s.Size = int64(len(l.hash))
			s.P = []byte(l.hash)
			str := ctxt.Syms.Lookup("go.link.pkghash."+l.Pkg, 0)
			str.Attr |= AttrReachable
			str.Type = SRODATA
			Addaddr(ctxt, str, s)
			adduint(ctxt, str, uint64(len(l.hash)))
		}
	}

	nsections := textsectionmap(ctxt)

	// Information about the layout of the executable image for the
	// runtime to use. Any changes here must be matched by changes to
	// the definition of moduledata in runtime/symtab.go.
	// This code uses several global variables that are set by pcln.go:pclntab.
	moduledata := ctxt.Moduledata
	// The pclntab slice
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.pclntab", 0))
	adduint(ctxt, moduledata, uint64(ctxt.Syms.Lookup("runtime.pclntab", 0).Size))
	adduint(ctxt, moduledata, uint64(ctxt.Syms.Lookup("runtime.pclntab", 0).Size))
	// The ftab slice
	Addaddrplus(ctxt, moduledata, ctxt.Syms.Lookup("runtime.pclntab", 0), int64(pclntabPclntabOffset))
	adduint(ctxt, moduledata, uint64(pclntabNfunc+1))
	adduint(ctxt, moduledata, uint64(pclntabNfunc+1))
	// The filetab slice
	Addaddrplus(ctxt, moduledata, ctxt.Syms.Lookup("runtime.pclntab", 0), int64(pclntabFiletabOffset))
	adduint(ctxt, moduledata, uint64(len(ctxt.Filesyms))+1)
	adduint(ctxt, moduledata, uint64(len(ctxt.Filesyms))+1)
	// findfunctab
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.findfunctab", 0))
	// minpc, maxpc
	Addaddr(ctxt, moduledata, pclntabFirstFunc)
	Addaddrplus(ctxt, moduledata, pclntabLastFunc, pclntabLastFunc.Size)
	// pointers to specific parts of the module
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.text", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.etext", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.noptrdata", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.enoptrdata", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.data", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.edata", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.bss", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.ebss", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.noptrbss", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.enoptrbss", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.end", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.gcdata", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.gcbss", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.types", 0))
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.etypes", 0))

	// text section information
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.textsectionmap", 0))
	adduint(ctxt, moduledata, uint64(nsections))
	adduint(ctxt, moduledata, uint64(nsections))

	// The typelinks slice
	typelinkSym := ctxt.Syms.Lookup("runtime.typelink", 0)
	ntypelinks := uint64(typelinkSym.Size) / 4
	Addaddr(ctxt, moduledata, typelinkSym)
	adduint(ctxt, moduledata, ntypelinks)
	adduint(ctxt, moduledata, ntypelinks)
	// The itablinks slice
	Addaddr(ctxt, moduledata, ctxt.Syms.Lookup("runtime.itablink", 0))
	adduint(ctxt, moduledata, uint64(nitablinks))
	adduint(ctxt, moduledata, uint64(nitablinks))
	// The ptab slice
	if ptab := ctxt.Syms.ROLookup("go.plugin.tabs", 0); ptab != nil && ptab.Attr.Reachable() {
		ptab.Attr |= AttrLocal
		ptab.Type = SRODATA

		nentries := uint64(len(ptab.P) / 8) // sizeof(nameOff) + sizeof(typeOff)
		Addaddr(ctxt, moduledata, ptab)
		adduint(ctxt, moduledata, nentries)
		adduint(ctxt, moduledata, nentries)
	} else {
		adduint(ctxt, moduledata, 0)
		adduint(ctxt, moduledata, 0)
		adduint(ctxt, moduledata, 0)
	}
	if Buildmode == BuildmodePlugin {
		addgostring(ctxt, moduledata, "go.link.thispluginpath", *flagPluginPath)

		pkghashes := ctxt.Syms.Lookup("go.link.pkghashes", 0)
		pkghashes.Attr |= AttrReachable
		pkghashes.Attr |= AttrLocal
		pkghashes.Type = SRODATA

		for i, l := range ctxt.Library {
			// pkghashes[i].name
			addgostring(ctxt, pkghashes, fmt.Sprintf("go.link.pkgname.%d", i), l.Pkg)
			// pkghashes[i].linktimehash
			addgostring(ctxt, pkghashes, fmt.Sprintf("go.link.pkglinkhash.%d", i), string(l.hash))
			// pkghashes[i].runtimehash
			hash := ctxt.Syms.ROLookup("go.link.pkghash."+l.Pkg, 0)
			Addaddr(ctxt, pkghashes, hash)
		}
		Addaddr(ctxt, moduledata, pkghashes)
		adduint(ctxt, moduledata, uint64(len(ctxt.Library)))
		adduint(ctxt, moduledata, uint64(len(ctxt.Library)))
	} else {
		adduint(ctxt, moduledata, 0) // pluginpath
		adduint(ctxt, moduledata, 0)
		adduint(ctxt, moduledata, 0) // pkghashes slice
		adduint(ctxt, moduledata, 0)
		adduint(ctxt, moduledata, 0)
	}
	if len(ctxt.Shlibs) > 0 {
		thismodulename := filepath.Base(*flagOutfile)
		switch Buildmode {
		case BuildmodeExe, BuildmodePIE:
			// When linking an executable, outfile is just "a.out". Make
			// it something slightly more comprehensible.
			thismodulename = "the executable"
		}
		addgostring(ctxt, moduledata, "go.link.thismodulename", thismodulename)

		modulehashes := ctxt.Syms.Lookup("go.link.abihashes", 0)
		modulehashes.Attr |= AttrReachable
		modulehashes.Attr |= AttrLocal
		modulehashes.Type = SRODATA

		for i, shlib := range ctxt.Shlibs {
			// modulehashes[i].modulename
			modulename := filepath.Base(shlib.Path)
			addgostring(ctxt, modulehashes, fmt.Sprintf("go.link.libname.%d", i), modulename)

			// modulehashes[i].linktimehash
			addgostring(ctxt, modulehashes, fmt.Sprintf("go.link.linkhash.%d", i), string(shlib.Hash))

			// modulehashes[i].runtimehash
			abihash := ctxt.Syms.Lookup("go.link.abihash."+modulename, 0)
			abihash.Attr |= AttrReachable
			Addaddr(ctxt, modulehashes, abihash)
		}

		Addaddr(ctxt, moduledata, modulehashes)
		adduint(ctxt, moduledata, uint64(len(ctxt.Shlibs)))
		adduint(ctxt, moduledata, uint64(len(ctxt.Shlibs)))
	}

	// The rest of moduledata is zero initialized.
	// When linking an object that does not contain the runtime we are
	// creating the moduledata from scratch and it does not have a
	// compiler-provided size, so read it from the type data.
	moduledatatype := ctxt.Syms.ROLookup("type.runtime.moduledata", 0)
	moduledata.Size = decodetypeSize(ctxt.Arch, moduledatatype)
	Symgrow(moduledata, moduledata.Size)

	lastmoduledatap := ctxt.Syms.Lookup("runtime.lastmoduledatap", 0)
	if lastmoduledatap.Type != SDYNIMPORT {
		lastmoduledatap.Type = SNOPTRDATA
		lastmoduledatap.Size = 0 // overwrite existing value
		Addaddr(ctxt, lastmoduledatap, moduledata)
	}
}
