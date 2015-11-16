// Inferno utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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
	"cmd/internal/obj"
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

	// When dynamically linking, we create LSym's by reading the names from
	// the symbol tables of the shared libraries and so the names need to
	// match exactly.  Tools like DTrace will have to wait for now.
	if !DynlinkingGo() {
		// Rewrite · to . for ASCII-only tools like DTrace (sigh)
		s = strings.Replace(s, "·", ".", -1)
	}

	n := len(s) + 1
	for len(Elfstrdat)+n > cap(Elfstrdat) {
		Elfstrdat = append(Elfstrdat[:cap(Elfstrdat)], 0)[:len(Elfstrdat)]
	}

	off := len(Elfstrdat)
	Elfstrdat = Elfstrdat[:off+n]
	copy(Elfstrdat[off:], s)

	return off
}

func putelfsyment(off int, addr int64, size int64, info int, shndx int, other int) {
	switch Thearch.Thechar {
	case '0', '6', '7', '9':
		Thearch.Lput(uint32(off))
		Cput(uint8(info))
		Cput(uint8(other))
		Thearch.Wput(uint16(shndx))
		Thearch.Vput(uint64(addr))
		Thearch.Vput(uint64(size))
		Symsize += ELF64SYMSIZE

	default:
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

func putelfsym(x *LSym, s string, t int, addr int64, size int64, ver int, go_ *LSym) {
	var type_ int

	switch t {
	default:
		return

	case 'T':
		type_ = STT_FUNC

	case 'D':
		type_ = STT_OBJECT

	case 'B':
		type_ = STT_OBJECT

	case 'U':
		// ElfType is only set for symbols read from Go shared libraries, but
		// for other symbols it is left as STT_NOTYPE which is fine.
		type_ = int(x.ElfType)

	case 't':
		type_ = STT_TLS
	}

	xo := x
	for xo.Outer != nil {
		xo = xo.Outer
	}

	var elfshnum int
	if xo.Type == obj.SDYNIMPORT || xo.Type == obj.SHOSTOBJ {
		elfshnum = SHN_UNDEF
	} else {
		if xo.Sect == nil {
			Ctxt.Cursym = x
			Diag("missing section in putelfsym")
			return
		}
		if xo.Sect.Elfsect == nil {
			Ctxt.Cursym = x
			Diag("missing ELF section in putelfsym")
			return
		}
		elfshnum = xo.Sect.Elfsect.shnum
	}

	// One pass for each binding: STB_LOCAL, STB_GLOBAL,
	// maybe one day STB_WEAK.
	bind := STB_GLOBAL

	if ver != 0 || (x.Type&obj.SHIDDEN != 0) || x.Local {
		bind = STB_LOCAL
	}

	// In external linking mode, we have to invoke gcc with -rdynamic
	// to get the exported symbols put into the dynamic symbol table.
	// To avoid filling the dynamic table with lots of unnecessary symbols,
	// mark all Go symbols local (not global) in the final executable.
	// But when we're dynamically linking, we need all those global symbols.
	if !DynlinkingGo() && Linkmode == LinkExternal && x.Cgoexport&CgoExportStatic == 0 && elfshnum != SHN_UNDEF {
		bind = STB_LOCAL
	}

	if Linkmode == LinkExternal && elfshnum != SHN_UNDEF {
		addr -= int64(xo.Sect.Vaddr)
	}
	other := STV_DEFAULT
	if x.Type&obj.SHIDDEN != 0 {
		other = STV_HIDDEN
	}
	if (Buildmode == BuildmodePIE || DynlinkingGo()) && Thearch.Thechar == '9' && type_ == STT_FUNC && x.Name != "runtime.duffzero" && x.Name != "runtime.duffcopy" {
		// On ppc64 the top three bits of the st_other field indicate how
		// many instructions separate the global and local entry points. In
		// our case it is two instructions, indicated by the value 3.
		other |= 3 << 5
	}

	if DynlinkingGo() && bind == STB_GLOBAL && elfbind == STB_LOCAL && x.Type == obj.STEXT {
		// When dynamically linking, we want references to functions defined
		// in this module to always be to the function object, not to the
		// PLT. We force this by writing an additional local symbol for every
		// global function symbol and making all relocations against the
		// global symbol refer to this local symbol instead (see
		// (*LSym).ElfsymForReloc). This is approximately equivalent to the
		// ELF linker -Bsymbolic-functions option, but that is buggy on
		// several platforms.
		putelfsyment(putelfstr("local."+s), addr, size, STB_LOCAL<<4|type_&0xf, elfshnum, other)
		x.LocalElfsym = int32(numelfsym)
		numelfsym++
		return
	} else if bind != elfbind {
		return
	}

	putelfsyment(putelfstr(s), addr, size, bind<<4|type_&0xf, elfshnum, other)
	x.Elfsym = int32(numelfsym)
	numelfsym++
}

func putelfsectionsym(s *LSym, shndx int) {
	putelfsyment(0, 0, 0, STB_LOCAL<<4|STT_SECTION, shndx, 0)
	s.Elfsym = int32(numelfsym)
	numelfsym++
}

func putelfsymshndx(sympos int64, shndx int) {
	here := Cpos()
	if elf64 {
		Cseek(sympos + 6)
	} else {
		Cseek(sympos + 14)
	}

	Thearch.Wput(uint16(shndx))
	Cseek(here)
}

func Asmelfsym() {
	// the first symbol entry is reserved
	putelfsyment(0, 0, 0, STB_LOCAL<<4|STT_NOTYPE, 0, 0)

	dwarfaddelfsectionsyms()

	elfbind = STB_LOCAL
	genasmsym(putelfsym)

	elfbind = STB_GLOBAL
	elfglobalsymndx = numelfsym
	genasmsym(putelfsym)
}

func putplan9sym(x *LSym, s string, t int, addr int64, size int64, ver int, go_ *LSym) {
	switch t {
	case 'T', 'L', 'D', 'B':
		if ver != 0 {
			t += 'a' - 'A'
		}
		fallthrough

	case 'a',
		'p',
		'f',
		'z',
		'Z',
		'm':
		l := 4
		if HEADTYPE == obj.Hplan9 && Thearch.Thechar == '6' && Debug['8'] == 0 {
			Lputb(uint32(addr >> 32))
			l = 8
		}

		Lputb(uint32(addr))
		Cput(uint8(t + 0x80)) /* 0x80 is variable length */

		var i int
		if t == 'z' || t == 'Z' {
			Cput(uint8(s[0]))
			for i = 1; s[i] != 0 || s[i+1] != 0; i += 2 {
				Cput(uint8(s[i]))
				Cput(uint8(s[i+1]))
			}

			Cput(0)
			Cput(0)
			i++
		} else {
			/* skip the '<' in filenames */
			if t == 'f' {
				s = s[1:]
			}
			for i = 0; i < len(s); i++ {
				Cput(uint8(s[i]))
			}
			Cput(0)
		}

		Symsize += int32(l) + 1 + int32(i) + 1

	default:
		return
	}
}

func Asmplan9sym() {
	genasmsym(putplan9sym)
}

var symt *LSym

func Wputl(w uint16) {
	Cput(uint8(w))
	Cput(uint8(w >> 8))
}

func Wputb(w uint16) {
	Cput(uint8(w >> 8))
	Cput(uint8(w))
}

func Lputb(l uint32) {
	Cput(uint8(l >> 24))
	Cput(uint8(l >> 16))
	Cput(uint8(l >> 8))
	Cput(uint8(l))
}

func Lputl(l uint32) {
	Cput(uint8(l))
	Cput(uint8(l >> 8))
	Cput(uint8(l >> 16))
	Cput(uint8(l >> 24))
}

func Vputb(v uint64) {
	Lputb(uint32(v >> 32))
	Lputb(uint32(v))
}

func Vputl(v uint64) {
	Lputl(uint32(v))
	Lputl(uint32(v >> 32))
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

func symtab() {
	dosymtype()

	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	xdefine("runtime.text", obj.STEXT, 0)

	xdefine("runtime.etext", obj.STEXT, 0)
	xdefine("runtime.typelink", obj.SRODATA, 0)
	xdefine("runtime.etypelink", obj.SRODATA, 0)
	xdefine("runtime.rodata", obj.SRODATA, 0)
	xdefine("runtime.erodata", obj.SRODATA, 0)
	xdefine("runtime.noptrdata", obj.SNOPTRDATA, 0)
	xdefine("runtime.enoptrdata", obj.SNOPTRDATA, 0)
	xdefine("runtime.data", obj.SDATA, 0)
	xdefine("runtime.edata", obj.SDATA, 0)
	xdefine("runtime.bss", obj.SBSS, 0)
	xdefine("runtime.ebss", obj.SBSS, 0)
	xdefine("runtime.noptrbss", obj.SNOPTRBSS, 0)
	xdefine("runtime.enoptrbss", obj.SNOPTRBSS, 0)
	xdefine("runtime.end", obj.SBSS, 0)
	xdefine("runtime.epclntab", obj.SRODATA, 0)
	xdefine("runtime.esymtab", obj.SRODATA, 0)

	// garbage collection symbols
	s := Linklookup(Ctxt, "runtime.gcdata", 0)

	s.Type = obj.SRODATA
	s.Size = 0
	s.Reachable = true
	xdefine("runtime.egcdata", obj.SRODATA, 0)

	s = Linklookup(Ctxt, "runtime.gcbss", 0)
	s.Type = obj.SRODATA
	s.Size = 0
	s.Reachable = true
	xdefine("runtime.egcbss", obj.SRODATA, 0)

	// pseudo-symbols to mark locations of type, string, and go string data.
	var symtype *LSym
	var symtyperel *LSym
	if UseRelro() && (Buildmode == BuildmodeCShared || Buildmode == BuildmodePIE) {
		s = Linklookup(Ctxt, "type.*", 0)

		s.Type = obj.STYPE
		s.Size = 0
		s.Reachable = true
		symtype = s

		s = Linklookup(Ctxt, "typerel.*", 0)

		s.Type = obj.STYPERELRO
		s.Size = 0
		s.Reachable = true
		symtyperel = s
	} else if !DynlinkingGo() {
		s = Linklookup(Ctxt, "type.*", 0)

		s.Type = obj.STYPE
		s.Size = 0
		s.Reachable = true
		symtype = s
		symtyperel = s
	}

	s = Linklookup(Ctxt, "go.string.*", 0)
	s.Type = obj.SGOSTRING
	s.Local = true
	s.Size = 0
	s.Reachable = true
	symgostring := s

	s = Linklookup(Ctxt, "go.func.*", 0)
	s.Type = obj.SGOFUNC
	s.Local = true
	s.Size = 0
	s.Reachable = true
	symgofunc := s

	s = Linklookup(Ctxt, "runtime.gcbits.*", 0)
	s.Type = obj.SGCBITS
	s.Local = true
	s.Size = 0
	s.Reachable = true
	symgcbits := s

	symtypelink := Linklookup(Ctxt, "runtime.typelink", 0)
	symtypelink.Type = obj.STYPELINK

	symt = Linklookup(Ctxt, "runtime.symtab", 0)
	symt.Local = true
	symt.Type = obj.SSYMTAB
	symt.Size = 0
	symt.Reachable = true

	ntypelinks := 0

	// assign specific types so that they sort together.
	// within a type they sort by size, so the .* symbols
	// just defined above will be first.
	// hide the specific symbols.
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if !s.Reachable || s.Special != 0 || s.Type != obj.SRODATA {
			continue
		}

		if strings.HasPrefix(s.Name, "type.") && !DynlinkingGo() {
			s.Hide = 1
			if UseRelro() && len(s.R) > 0 {
				s.Type = obj.STYPERELRO
				s.Outer = symtyperel
			} else {
				s.Type = obj.STYPE
				s.Outer = symtype
			}
		}

		if strings.HasPrefix(s.Name, "go.typelink.") {
			ntypelinks++
			s.Type = obj.STYPELINK
			s.Hide = 1
			s.Outer = symtypelink
		}

		if strings.HasPrefix(s.Name, "go.string.") {
			s.Type = obj.SGOSTRING
			s.Hide = 1
			s.Outer = symgostring
		}

		if strings.HasPrefix(s.Name, "runtime.gcbits.") {
			s.Type = obj.SGCBITS
			s.Hide = 1
			s.Outer = symgcbits
		}

		if strings.HasPrefix(s.Name, "go.func.") {
			s.Type = obj.SGOFUNC
			s.Hide = 1
			s.Outer = symgofunc
		}

		if strings.HasPrefix(s.Name, "gcargs.") || strings.HasPrefix(s.Name, "gclocals.") || strings.HasPrefix(s.Name, "gclocals·") {
			s.Type = obj.SGOFUNC
			s.Hide = 1
			s.Outer = symgofunc
			s.Align = 4
			liveness += (s.Size + int64(s.Align) - 1) &^ (int64(s.Align) - 1)
		}
	}

	if Buildmode == BuildmodeShared {
		abihashgostr := Linklookup(Ctxt, "go.link.abihash."+filepath.Base(outfile), 0)
		abihashgostr.Reachable = true
		abihashgostr.Type = obj.SRODATA
		hashsym := Linklookup(Ctxt, "go.link.abihashbytes", 0)
		Addaddr(Ctxt, abihashgostr, hashsym)
		adduint(Ctxt, abihashgostr, uint64(hashsym.Size))
	}

	// Information about the layout of the executable image for the
	// runtime to use. Any changes here must be matched by changes to
	// the definition of moduledata in runtime/symtab.go.
	// This code uses several global variables that are set by pcln.go:pclntab.
	moduledata := Ctxt.Moduledata
	// The pclntab slice
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.pclntab", 0))
	adduint(Ctxt, moduledata, uint64(Linklookup(Ctxt, "runtime.pclntab", 0).Size))
	adduint(Ctxt, moduledata, uint64(Linklookup(Ctxt, "runtime.pclntab", 0).Size))
	// The ftab slice
	Addaddrplus(Ctxt, moduledata, Linklookup(Ctxt, "runtime.pclntab", 0), int64(pclntabPclntabOffset))
	adduint(Ctxt, moduledata, uint64(pclntabNfunc+1))
	adduint(Ctxt, moduledata, uint64(pclntabNfunc+1))
	// The filetab slice
	Addaddrplus(Ctxt, moduledata, Linklookup(Ctxt, "runtime.pclntab", 0), int64(pclntabFiletabOffset))
	adduint(Ctxt, moduledata, uint64(Ctxt.Nhistfile)+1)
	adduint(Ctxt, moduledata, uint64(Ctxt.Nhistfile)+1)
	// findfunctab
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.findfunctab", 0))
	// minpc, maxpc
	Addaddr(Ctxt, moduledata, pclntabFirstFunc)
	Addaddrplus(Ctxt, moduledata, pclntabLastFunc, pclntabLastFunc.Size)
	// pointers to specific parts of the module
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.text", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.etext", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.noptrdata", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.enoptrdata", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.data", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.edata", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.bss", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.ebss", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.noptrbss", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.enoptrbss", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.end", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.gcdata", 0))
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.gcbss", 0))
	// The typelinks slice
	Addaddr(Ctxt, moduledata, Linklookup(Ctxt, "runtime.typelink", 0))
	adduint(Ctxt, moduledata, uint64(ntypelinks))
	adduint(Ctxt, moduledata, uint64(ntypelinks))
	if len(Ctxt.Shlibs) > 0 {
		thismodulename := filepath.Base(outfile)
		switch Buildmode {
		case BuildmodeExe, BuildmodePIE:
			// When linking an executable, outfile is just "a.out". Make
			// it something slightly more comprehensible.
			thismodulename = "the executable"
		}
		addgostring(moduledata, "go.link.thismodulename", thismodulename)

		modulehashes := Linklookup(Ctxt, "go.link.abihashes", 0)
		modulehashes.Reachable = true
		modulehashes.Local = true
		modulehashes.Type = obj.SRODATA

		for i, shlib := range Ctxt.Shlibs {
			// modulehashes[i].modulename
			modulename := filepath.Base(shlib.Path)
			addgostring(modulehashes, fmt.Sprintf("go.link.libname.%d", i), modulename)

			// modulehashes[i].linktimehash
			addgostring(modulehashes, fmt.Sprintf("go.link.linkhash.%d", i), string(shlib.Hash))

			// modulehashes[i].runtimehash
			abihash := Linklookup(Ctxt, "go.link.abihash."+modulename, 0)
			abihash.Reachable = true
			Addaddr(Ctxt, modulehashes, abihash)
		}

		Addaddr(Ctxt, moduledata, modulehashes)
		adduint(Ctxt, moduledata, uint64(len(Ctxt.Shlibs)))
		adduint(Ctxt, moduledata, uint64(len(Ctxt.Shlibs)))
	}
	// The rest of moduledata is zero initialized.
	// When linking an object that does not contain the runtime we are
	// creating the moduledata from scratch and it does not have a
	// compiler-provided size, so read it from the type data.
	moduledatatype := Linkrlookup(Ctxt, "type.runtime.moduledata", 0)
	moduledata.Size = decodetype_size(moduledatatype)
	Symgrow(Ctxt, moduledata, moduledata.Size)

	lastmoduledatap := Linklookup(Ctxt, "runtime.lastmoduledatap", 0)
	if lastmoduledatap.Type != obj.SDYNIMPORT {
		lastmoduledatap.Type = obj.SNOPTRDATA
		lastmoduledatap.Size = 0 // overwrite existing value
		Addaddr(Ctxt, lastmoduledatap, moduledata)
	}
}
