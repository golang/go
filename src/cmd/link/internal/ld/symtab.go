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
	"cmd/link/internal/sym"
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

func putelfsyment(out *OutBuf, off int, addr int64, size int64, info int, shndx int, other int) {
	if elf64 {
		out.Write32(uint32(off))
		out.Write8(uint8(info))
		out.Write8(uint8(other))
		out.Write16(uint16(shndx))
		out.Write64(uint64(addr))
		out.Write64(uint64(size))
		Symsize += ELF64SYMSIZE
	} else {
		out.Write32(uint32(off))
		out.Write32(uint32(addr))
		out.Write32(uint32(size))
		out.Write8(uint8(info))
		out.Write8(uint8(other))
		out.Write16(uint16(shndx))
		Symsize += ELF32SYMSIZE
	}
}

var numelfsym = 1 // 0 is reserved

var elfbind int

func putelfsym(ctxt *Link, x *sym.Symbol, s string, t SymbolType, addr int64, go_ *sym.Symbol) {
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
		typ = int(x.ElfType())

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
	if xo.Type == sym.SDYNIMPORT || xo.Type == sym.SHOSTOBJ {
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
		elfshnum = xo.Sect.Elfsect.(*ElfShdr).shnum
	}

	// One pass for each binding: STB_LOCAL, STB_GLOBAL,
	// maybe one day STB_WEAK.
	bind := STB_GLOBAL

	if x.IsFileLocal() || x.Attr.VisibilityHidden() || x.Attr.Local() {
		bind = STB_LOCAL
	}

	// In external linking mode, we have to invoke gcc with -rdynamic
	// to get the exported symbols put into the dynamic symbol table.
	// To avoid filling the dynamic table with lots of unnecessary symbols,
	// mark all Go symbols local (not global) in the final executable.
	// But when we're dynamically linking, we need all those global symbols.
	if !ctxt.DynlinkingGo() && ctxt.LinkMode == LinkExternal && !x.Attr.CgoExportStatic() && elfshnum != SHN_UNDEF {
		bind = STB_LOCAL
	}

	if ctxt.LinkMode == LinkExternal && elfshnum != SHN_UNDEF {
		addr -= int64(xo.Sect.Vaddr)
	}
	other := STV_DEFAULT
	if x.Attr.VisibilityHidden() {
		// TODO(mwhudson): We only set AttrVisibilityHidden in ldelf, i.e. when
		// internally linking. But STV_HIDDEN visibility only matters in object
		// files and shared libraries, and as we are a long way from implementing
		// internal linking for shared libraries and only create object files when
		// externally linking, I don't think this makes a lot of sense.
		other = STV_HIDDEN
	}
	if ctxt.Arch.Family == sys.PPC64 && typ == STT_FUNC && x.Attr.Shared() && x.Name != "runtime.duffzero" && x.Name != "runtime.duffcopy" {
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

	if ctxt.DynlinkingGo() && bind == STB_GLOBAL && elfbind == STB_LOCAL && x.Type == sym.STEXT {
		// When dynamically linking, we want references to functions defined
		// in this module to always be to the function object, not to the
		// PLT. We force this by writing an additional local symbol for every
		// global function symbol and making all relocations against the
		// global symbol refer to this local symbol instead (see
		// (*sym.Symbol).ElfsymForReloc). This is approximately equivalent to the
		// ELF linker -Bsymbolic-functions option, but that is buggy on
		// several platforms.
		putelfsyment(ctxt.Out, putelfstr("local."+s), addr, size, STB_LOCAL<<4|typ&0xf, elfshnum, other)
		x.LocalElfsym = int32(numelfsym)
		numelfsym++
		return
	} else if bind != elfbind {
		return
	}

	putelfsyment(ctxt.Out, putelfstr(s), addr, size, bind<<4|typ&0xf, elfshnum, other)
	x.Elfsym = int32(numelfsym)
	numelfsym++
}

func putelfsectionsym(out *OutBuf, s *sym.Symbol, shndx int) {
	putelfsyment(out, 0, 0, 0, STB_LOCAL<<4|STT_SECTION, shndx, 0)
	s.Elfsym = int32(numelfsym)
	numelfsym++
}

func Asmelfsym(ctxt *Link) {
	// the first symbol entry is reserved
	putelfsyment(ctxt.Out, 0, 0, 0, STB_LOCAL<<4|STT_NOTYPE, 0, 0)

	dwarfaddelfsectionsyms(ctxt)

	// Some linkers will add a FILE sym if one is not present.
	// Avoid having the working directory inserted into the symbol table.
	// It is added with a name to avoid problems with external linking
	// encountered on some versions of Solaris. See issue #14957.
	putelfsyment(ctxt.Out, putelfstr("go.go"), 0, 0, STB_LOCAL<<4|STT_FILE, SHN_ABS, 0)
	numelfsym++

	elfbind = STB_LOCAL
	genasmsym(ctxt, putelfsym)

	elfbind = STB_GLOBAL
	elfglobalsymndx = numelfsym
	genasmsym(ctxt, putelfsym)
}

func putplan9sym(ctxt *Link, x *sym.Symbol, s string, typ SymbolType, addr int64, go_ *sym.Symbol) {
	t := int(typ)
	switch typ {
	case TextSym, DataSym, BSSSym:
		if x.IsFileLocal() {
			t += 'a' - 'A'
		}
		fallthrough

	case AutoSym, ParamSym, FrameSym:
		l := 4
		if ctxt.HeadType == objabi.Hplan9 && ctxt.Arch.Family == sys.AMD64 && !Flag8 {
			ctxt.Out.Write32b(uint32(addr >> 32))
			l = 8
		}

		ctxt.Out.Write32b(uint32(addr))
		ctxt.Out.Write8(uint8(t + 0x80)) /* 0x80 is variable length */

		ctxt.Out.WriteString(s)
		ctxt.Out.Write8(0)

		Symsize += int32(l) + 1 + int32(len(s)) + 1

	default:
		return
	}
}

func Asmplan9sym(ctxt *Link) {
	genasmsym(ctxt, putplan9sym)
}

var symt *sym.Symbol

type byPkg []*sym.Library

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
	t.Type = sym.SRODATA
	t.Attr |= sym.AttrReachable
	nsections := int64(0)

	for _, sect := range Segtext.Sections {
		if sect.Name == ".text" {
			nsections++
		} else {
			break
		}
	}
	t.Grow(3 * nsections * int64(ctxt.Arch.PtrSize))

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
		off = t.SetUint(ctxt.Arch, off, sect.Vaddr-textbase)
		off = t.SetUint(ctxt.Arch, off, sect.Length)
		if n == 0 {
			s := ctxt.Syms.ROLookup("runtime.text", 0)
			if s == nil {
				Errorf(nil, "Unable to find symbol runtime.text\n")
			}
			off = t.SetAddr(ctxt.Arch, off, s)

		} else {
			s := ctxt.Syms.Lookup(fmt.Sprintf("runtime.text.%d", n), 0)
			if s == nil {
				Errorf(nil, "Unable to find symbol runtime.text.%d\n", n)
			}
			off = t.SetAddr(ctxt.Arch, off, s)
		}
		n++
	}
	return uint32(n)
}

func (ctxt *Link) symtab() {
	dosymtype(ctxt)

	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	ctxt.xdefine("runtime.text", sym.STEXT, 0)

	ctxt.xdefine("runtime.etext", sym.STEXT, 0)
	ctxt.xdefine("runtime.itablink", sym.SRODATA, 0)
	ctxt.xdefine("runtime.eitablink", sym.SRODATA, 0)
	ctxt.xdefine("runtime.rodata", sym.SRODATA, 0)
	ctxt.xdefine("runtime.erodata", sym.SRODATA, 0)
	ctxt.xdefine("runtime.types", sym.SRODATA, 0)
	ctxt.xdefine("runtime.etypes", sym.SRODATA, 0)
	ctxt.xdefine("runtime.noptrdata", sym.SNOPTRDATA, 0)
	ctxt.xdefine("runtime.enoptrdata", sym.SNOPTRDATA, 0)
	ctxt.xdefine("runtime.data", sym.SDATA, 0)
	ctxt.xdefine("runtime.edata", sym.SDATA, 0)
	ctxt.xdefine("runtime.bss", sym.SBSS, 0)
	ctxt.xdefine("runtime.ebss", sym.SBSS, 0)
	ctxt.xdefine("runtime.noptrbss", sym.SNOPTRBSS, 0)
	ctxt.xdefine("runtime.enoptrbss", sym.SNOPTRBSS, 0)
	ctxt.xdefine("runtime.end", sym.SBSS, 0)
	ctxt.xdefine("runtime.epclntab", sym.SRODATA, 0)
	ctxt.xdefine("runtime.esymtab", sym.SRODATA, 0)

	// garbage collection symbols
	s := ctxt.Syms.Lookup("runtime.gcdata", 0)

	s.Type = sym.SRODATA
	s.Size = 0
	s.Attr |= sym.AttrReachable
	ctxt.xdefine("runtime.egcdata", sym.SRODATA, 0)

	s = ctxt.Syms.Lookup("runtime.gcbss", 0)
	s.Type = sym.SRODATA
	s.Size = 0
	s.Attr |= sym.AttrReachable
	ctxt.xdefine("runtime.egcbss", sym.SRODATA, 0)

	// pseudo-symbols to mark locations of type, string, and go string data.
	var symtype *sym.Symbol
	var symtyperel *sym.Symbol
	if !ctxt.DynlinkingGo() {
		if ctxt.UseRelro() && (ctxt.BuildMode == BuildModeCArchive || ctxt.BuildMode == BuildModeCShared || ctxt.BuildMode == BuildModePIE) {
			s = ctxt.Syms.Lookup("type.*", 0)

			s.Type = sym.STYPE
			s.Size = 0
			s.Attr |= sym.AttrReachable
			symtype = s

			s = ctxt.Syms.Lookup("typerel.*", 0)

			s.Type = sym.STYPERELRO
			s.Size = 0
			s.Attr |= sym.AttrReachable
			symtyperel = s
		} else {
			s = ctxt.Syms.Lookup("type.*", 0)

			s.Type = sym.STYPE
			s.Size = 0
			s.Attr |= sym.AttrReachable
			symtype = s
			symtyperel = s
		}
	}

	groupSym := func(name string, t sym.SymKind) *sym.Symbol {
		s := ctxt.Syms.Lookup(name, 0)
		s.Type = t
		s.Size = 0
		s.Attr |= sym.AttrLocal | sym.AttrReachable
		return s
	}
	var (
		symgostring = groupSym("go.string.*", sym.SGOSTRING)
		symgofunc   = groupSym("go.func.*", sym.SGOFUNC)
		symgcbits   = groupSym("runtime.gcbits.*", sym.SGCBITS)
	)

	var symgofuncrel *sym.Symbol
	if !ctxt.DynlinkingGo() {
		if ctxt.UseRelro() {
			symgofuncrel = groupSym("go.funcrel.*", sym.SGOFUNCRELRO)
		} else {
			symgofuncrel = symgofunc
		}
	}

	symitablink := ctxt.Syms.Lookup("runtime.itablink", 0)
	symitablink.Type = sym.SITABLINK

	symt = ctxt.Syms.Lookup("runtime.symtab", 0)
	symt.Attr |= sym.AttrLocal
	symt.Type = sym.SSYMTAB
	symt.Size = 0
	symt.Attr |= sym.AttrReachable

	nitablinks := 0

	// assign specific types so that they sort together.
	// within a type they sort by size, so the .* symbols
	// just defined above will be first.
	// hide the specific symbols.
	for _, s := range ctxt.Syms.Allsym {
		if ctxt.LinkMode != LinkExternal && isStaticTemp(s.Name) {
			s.Attr |= sym.AttrNotInSymbolTable
		}

		if !s.Attr.Reachable() || s.Attr.Special() || s.Type != sym.SRODATA {
			continue
		}

		switch {
		case strings.HasPrefix(s.Name, "type."):
			if !ctxt.DynlinkingGo() {
				s.Attr |= sym.AttrNotInSymbolTable
			}
			if ctxt.UseRelro() {
				s.Type = sym.STYPERELRO
				s.Outer = symtyperel
			} else {
				s.Type = sym.STYPE
				s.Outer = symtype
			}

		case strings.HasPrefix(s.Name, "go.importpath.") && ctxt.UseRelro():
			// Keep go.importpath symbols in the same section as types and
			// names, as they can be referred to by a section offset.
			s.Type = sym.STYPERELRO

		case strings.HasPrefix(s.Name, "go.itablink."):
			nitablinks++
			s.Type = sym.SITABLINK
			s.Attr |= sym.AttrNotInSymbolTable
			s.Outer = symitablink

		case strings.HasPrefix(s.Name, "go.string."):
			s.Type = sym.SGOSTRING
			s.Attr |= sym.AttrNotInSymbolTable
			s.Outer = symgostring

		case strings.HasPrefix(s.Name, "runtime.gcbits."):
			s.Type = sym.SGCBITS
			s.Attr |= sym.AttrNotInSymbolTable
			s.Outer = symgcbits

		case strings.HasSuffix(s.Name, "·f"):
			if !ctxt.DynlinkingGo() {
				s.Attr |= sym.AttrNotInSymbolTable
			}
			if ctxt.UseRelro() {
				s.Type = sym.SGOFUNCRELRO
				s.Outer = symgofuncrel
			} else {
				s.Type = sym.SGOFUNC
				s.Outer = symgofunc
			}

		case strings.HasPrefix(s.Name, "gcargs."),
			strings.HasPrefix(s.Name, "gclocals."),
			strings.HasPrefix(s.Name, "gclocals·"),
			strings.HasPrefix(s.Name, "inltree."):
			s.Type = sym.SGOFUNC
			s.Attr |= sym.AttrNotInSymbolTable
			s.Outer = symgofunc
			s.Align = 4
			liveness += (s.Size + int64(s.Align) - 1) &^ (int64(s.Align) - 1)
		}
	}

	if ctxt.BuildMode == BuildModeShared {
		abihashgostr := ctxt.Syms.Lookup("go.link.abihash."+filepath.Base(*flagOutfile), 0)
		abihashgostr.Attr |= sym.AttrReachable
		abihashgostr.Type = sym.SRODATA
		hashsym := ctxt.Syms.Lookup("go.link.abihashbytes", 0)
		abihashgostr.AddAddr(ctxt.Arch, hashsym)
		abihashgostr.AddUint(ctxt.Arch, uint64(hashsym.Size))
	}
	if ctxt.BuildMode == BuildModePlugin || ctxt.CanUsePlugins() {
		for _, l := range ctxt.Library {
			s := ctxt.Syms.Lookup("go.link.pkghashbytes."+l.Pkg, 0)
			s.Attr |= sym.AttrReachable
			s.Type = sym.SRODATA
			s.Size = int64(len(l.Hash))
			s.P = []byte(l.Hash)
			str := ctxt.Syms.Lookup("go.link.pkghash."+l.Pkg, 0)
			str.Attr |= sym.AttrReachable
			str.Type = sym.SRODATA
			str.AddAddr(ctxt.Arch, s)
			str.AddUint(ctxt.Arch, uint64(len(l.Hash)))
		}
	}

	nsections := textsectionmap(ctxt)

	// Information about the layout of the executable image for the
	// runtime to use. Any changes here must be matched by changes to
	// the definition of moduledata in runtime/symtab.go.
	// This code uses several global variables that are set by pcln.go:pclntab.
	moduledata := ctxt.Moduledata
	// The pclntab slice
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.pclntab", 0))
	moduledata.AddUint(ctxt.Arch, uint64(ctxt.Syms.Lookup("runtime.pclntab", 0).Size))
	moduledata.AddUint(ctxt.Arch, uint64(ctxt.Syms.Lookup("runtime.pclntab", 0).Size))
	// The ftab slice
	moduledata.AddAddrPlus(ctxt.Arch, ctxt.Syms.Lookup("runtime.pclntab", 0), int64(pclntabPclntabOffset))
	moduledata.AddUint(ctxt.Arch, uint64(pclntabNfunc+1))
	moduledata.AddUint(ctxt.Arch, uint64(pclntabNfunc+1))
	// The filetab slice
	moduledata.AddAddrPlus(ctxt.Arch, ctxt.Syms.Lookup("runtime.pclntab", 0), int64(pclntabFiletabOffset))
	moduledata.AddUint(ctxt.Arch, uint64(len(ctxt.Filesyms))+1)
	moduledata.AddUint(ctxt.Arch, uint64(len(ctxt.Filesyms))+1)
	// findfunctab
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.findfunctab", 0))
	// minpc, maxpc
	moduledata.AddAddr(ctxt.Arch, pclntabFirstFunc)
	moduledata.AddAddrPlus(ctxt.Arch, pclntabLastFunc, pclntabLastFunc.Size)
	// pointers to specific parts of the module
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.text", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.etext", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.noptrdata", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.enoptrdata", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.data", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.edata", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.bss", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.ebss", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.noptrbss", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.enoptrbss", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.end", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.gcdata", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.gcbss", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.types", 0))
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.etypes", 0))

	// text section information
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.textsectionmap", 0))
	moduledata.AddUint(ctxt.Arch, uint64(nsections))
	moduledata.AddUint(ctxt.Arch, uint64(nsections))

	// The typelinks slice
	typelinkSym := ctxt.Syms.Lookup("runtime.typelink", 0)
	ntypelinks := uint64(typelinkSym.Size) / 4
	moduledata.AddAddr(ctxt.Arch, typelinkSym)
	moduledata.AddUint(ctxt.Arch, ntypelinks)
	moduledata.AddUint(ctxt.Arch, ntypelinks)
	// The itablinks slice
	moduledata.AddAddr(ctxt.Arch, ctxt.Syms.Lookup("runtime.itablink", 0))
	moduledata.AddUint(ctxt.Arch, uint64(nitablinks))
	moduledata.AddUint(ctxt.Arch, uint64(nitablinks))
	// The ptab slice
	if ptab := ctxt.Syms.ROLookup("go.plugin.tabs", 0); ptab != nil && ptab.Attr.Reachable() {
		ptab.Attr |= sym.AttrLocal
		ptab.Type = sym.SRODATA

		nentries := uint64(len(ptab.P) / 8) // sizeof(nameOff) + sizeof(typeOff)
		moduledata.AddAddr(ctxt.Arch, ptab)
		moduledata.AddUint(ctxt.Arch, nentries)
		moduledata.AddUint(ctxt.Arch, nentries)
	} else {
		moduledata.AddUint(ctxt.Arch, 0)
		moduledata.AddUint(ctxt.Arch, 0)
		moduledata.AddUint(ctxt.Arch, 0)
	}
	if ctxt.BuildMode == BuildModePlugin {
		addgostring(ctxt, moduledata, "go.link.thispluginpath", objabi.PathToPrefix(*flagPluginPath))

		pkghashes := ctxt.Syms.Lookup("go.link.pkghashes", 0)
		pkghashes.Attr |= sym.AttrReachable
		pkghashes.Attr |= sym.AttrLocal
		pkghashes.Type = sym.SRODATA

		for i, l := range ctxt.Library {
			// pkghashes[i].name
			addgostring(ctxt, pkghashes, fmt.Sprintf("go.link.pkgname.%d", i), l.Pkg)
			// pkghashes[i].linktimehash
			addgostring(ctxt, pkghashes, fmt.Sprintf("go.link.pkglinkhash.%d", i), l.Hash)
			// pkghashes[i].runtimehash
			hash := ctxt.Syms.ROLookup("go.link.pkghash."+l.Pkg, 0)
			pkghashes.AddAddr(ctxt.Arch, hash)
		}
		moduledata.AddAddr(ctxt.Arch, pkghashes)
		moduledata.AddUint(ctxt.Arch, uint64(len(ctxt.Library)))
		moduledata.AddUint(ctxt.Arch, uint64(len(ctxt.Library)))
	} else {
		moduledata.AddUint(ctxt.Arch, 0) // pluginpath
		moduledata.AddUint(ctxt.Arch, 0)
		moduledata.AddUint(ctxt.Arch, 0) // pkghashes slice
		moduledata.AddUint(ctxt.Arch, 0)
		moduledata.AddUint(ctxt.Arch, 0)
	}
	if len(ctxt.Shlibs) > 0 {
		thismodulename := filepath.Base(*flagOutfile)
		switch ctxt.BuildMode {
		case BuildModeExe, BuildModePIE:
			// When linking an executable, outfile is just "a.out". Make
			// it something slightly more comprehensible.
			thismodulename = "the executable"
		}
		addgostring(ctxt, moduledata, "go.link.thismodulename", thismodulename)

		modulehashes := ctxt.Syms.Lookup("go.link.abihashes", 0)
		modulehashes.Attr |= sym.AttrReachable
		modulehashes.Attr |= sym.AttrLocal
		modulehashes.Type = sym.SRODATA

		for i, shlib := range ctxt.Shlibs {
			// modulehashes[i].modulename
			modulename := filepath.Base(shlib.Path)
			addgostring(ctxt, modulehashes, fmt.Sprintf("go.link.libname.%d", i), modulename)

			// modulehashes[i].linktimehash
			addgostring(ctxt, modulehashes, fmt.Sprintf("go.link.linkhash.%d", i), string(shlib.Hash))

			// modulehashes[i].runtimehash
			abihash := ctxt.Syms.Lookup("go.link.abihash."+modulename, 0)
			abihash.Attr |= sym.AttrReachable
			modulehashes.AddAddr(ctxt.Arch, abihash)
		}

		moduledata.AddAddr(ctxt.Arch, modulehashes)
		moduledata.AddUint(ctxt.Arch, uint64(len(ctxt.Shlibs)))
		moduledata.AddUint(ctxt.Arch, uint64(len(ctxt.Shlibs)))
	} else {
		moduledata.AddUint(ctxt.Arch, 0) // modulename
		moduledata.AddUint(ctxt.Arch, 0)
		moduledata.AddUint(ctxt.Arch, 0) // moduleshashes slice
		moduledata.AddUint(ctxt.Arch, 0)
		moduledata.AddUint(ctxt.Arch, 0)
	}

	hasmain := ctxt.BuildMode == BuildModeExe || ctxt.BuildMode == BuildModePIE
	if hasmain {
		moduledata.AddUint8(1)
	} else {
		moduledata.AddUint8(0)
	}

	// The rest of moduledata is zero initialized.
	// When linking an object that does not contain the runtime we are
	// creating the moduledata from scratch and it does not have a
	// compiler-provided size, so read it from the type data.
	moduledatatype := ctxt.Syms.ROLookup("type.runtime.moduledata", 0)
	moduledata.Size = decodetypeSize(ctxt.Arch, moduledatatype)
	moduledata.Grow(moduledata.Size)

	lastmoduledatap := ctxt.Syms.Lookup("runtime.lastmoduledatap", 0)
	if lastmoduledatap.Type != sym.SDYNIMPORT {
		lastmoduledatap.Type = sym.SNOPTRDATA
		lastmoduledatap.Size = 0 // overwrite existing value
		lastmoduledatap.AddAddr(ctxt.Arch, moduledata)
	}
}

func isStaticTemp(name string) bool {
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i:]
	}
	return strings.Contains(name, "..stmp_")
}
