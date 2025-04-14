// Inferno utils/6l/span.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/span.c
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
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"fmt"
	"internal/buildcfg"
	"path/filepath"
	"strings"
)

// Symbol table.

func putelfstr(s string) int {
	if len(elfstrdat) == 0 && s != "" {
		// first entry must be empty string
		putelfstr("")
	}

	off := len(elfstrdat)
	elfstrdat = append(elfstrdat, s...)
	elfstrdat = append(elfstrdat, 0)
	return off
}

func putelfsyment(out *OutBuf, off int, addr int64, size int64, info uint8, shndx elf.SectionIndex, other int) {
	if elf64 {
		out.Write32(uint32(off))
		out.Write8(info)
		out.Write8(uint8(other))
		out.Write16(uint16(shndx))
		out.Write64(uint64(addr))
		out.Write64(uint64(size))
		symSize += ELF64SYMSIZE
	} else {
		out.Write32(uint32(off))
		out.Write32(uint32(addr))
		out.Write32(uint32(size))
		out.Write8(info)
		out.Write8(uint8(other))
		out.Write16(uint16(shndx))
		symSize += ELF32SYMSIZE
	}
}

func putelfsym(ctxt *Link, x loader.Sym, typ elf.SymType, curbind elf.SymBind) {
	ldr := ctxt.loader
	addr := ldr.SymValue(x)
	size := ldr.SymSize(x)

	xo := x
	if ldr.OuterSym(x) != 0 {
		xo = ldr.OuterSym(x)
	}
	xot := ldr.SymType(xo)
	xosect := ldr.SymSect(xo)

	var elfshnum elf.SectionIndex
	if xot == sym.SDYNIMPORT || xot == sym.SHOSTOBJ || xot == sym.SUNDEFEXT {
		elfshnum = elf.SHN_UNDEF
		size = 0
	} else {
		if xosect == nil {
			ldr.Errorf(x, "missing section in putelfsym")
			return
		}
		if xosect.Elfsect == nil {
			ldr.Errorf(x, "missing ELF section in putelfsym")
			return
		}
		elfshnum = xosect.Elfsect.(*ElfShdr).shnum
	}

	sname := ldr.SymExtname(x)
	sname = mangleABIName(ctxt, ldr, x, sname)

	// One pass for each binding: elf.STB_LOCAL, elf.STB_GLOBAL,
	// maybe one day elf.STB_WEAK.
	bind := elf.STB_GLOBAL
	if ldr.IsFileLocal(x) && !isStaticTmp(sname) || ldr.AttrVisibilityHidden(x) || ldr.AttrLocal(x) {
		// Static tmp is package local, but a package can be shared among multiple DSOs.
		// They need to have a single view of the static tmp that are writable.
		bind = elf.STB_LOCAL
	}

	// In external linking mode, we have to invoke gcc with -rdynamic
	// to get the exported symbols put into the dynamic symbol table.
	// To avoid filling the dynamic table with lots of unnecessary symbols,
	// mark all Go symbols local (not global) in the final executable.
	// But when we're dynamically linking, we need all those global symbols.
	if !ctxt.DynlinkingGo() && ctxt.IsExternal() && !ldr.AttrCgoExportStatic(x) && elfshnum != elf.SHN_UNDEF {
		bind = elf.STB_LOCAL
	}

	if ctxt.LinkMode == LinkExternal && elfshnum != elf.SHN_UNDEF {
		addr -= int64(xosect.Vaddr)
	}
	other := int(elf.STV_DEFAULT)
	if ldr.AttrVisibilityHidden(x) {
		// TODO(mwhudson): We only set AttrVisibilityHidden in ldelf, i.e. when
		// internally linking. But STV_HIDDEN visibility only matters in object
		// files and shared libraries, and as we are a long way from implementing
		// internal linking for shared libraries and only create object files when
		// externally linking, I don't think this makes a lot of sense.
		other = int(elf.STV_HIDDEN)
	}
	if ctxt.IsPPC64() && typ == elf.STT_FUNC && ldr.AttrShared(x) {
		// On ppc64 the top three bits of the st_other field indicate how many
		// bytes separate the global and local entry points. For non-PCrel shared
		// symbols this is always 8 bytes except for some special functions.
		hasPCrel := buildcfg.GOPPC64 >= 10 && buildcfg.GOOS == "linux"

		// This should match the preprocessing behavior in cmd/internal/obj/ppc64/obj9.go
		// where the distinct global entry is inserted.
		if !hasPCrel && ldr.SymName(x) != "runtime.duffzero" && ldr.SymName(x) != "runtime.duffcopy" {
			other |= 3 << 5
		}
	}

	// When dynamically linking, we create Symbols by reading the names from
	// the symbol tables of the shared libraries and so the names need to
	// match exactly. Tools like DTrace will have to wait for now.
	if !ctxt.DynlinkingGo() {
		// Rewrite · to . for ASCII-only tools like DTrace (sigh)
		sname = strings.ReplaceAll(sname, "·", ".")
	}

	if ctxt.DynlinkingGo() && bind == elf.STB_GLOBAL && curbind == elf.STB_LOCAL && ldr.SymType(x).IsText() {
		// When dynamically linking, we want references to functions defined
		// in this module to always be to the function object, not to the
		// PLT. We force this by writing an additional local symbol for every
		// global function symbol and making all relocations against the
		// global symbol refer to this local symbol instead (see
		// (*sym.Symbol).ElfsymForReloc). This is approximately equivalent to the
		// ELF linker -Bsymbolic-functions option, but that is buggy on
		// several platforms.
		putelfsyment(ctxt.Out, putelfstr("local."+sname), addr, size, elf.ST_INFO(elf.STB_LOCAL, typ), elfshnum, other)
		ldr.SetSymLocalElfSym(x, int32(ctxt.numelfsym))
		ctxt.numelfsym++
		return
	} else if bind != curbind {
		return
	}

	putelfsyment(ctxt.Out, putelfstr(sname), addr, size, elf.ST_INFO(bind, typ), elfshnum, other)
	ldr.SetSymElfSym(x, int32(ctxt.numelfsym))
	ctxt.numelfsym++
}

func putelfsectionsym(ctxt *Link, out *OutBuf, s loader.Sym, shndx elf.SectionIndex) {
	putelfsyment(out, 0, 0, 0, elf.ST_INFO(elf.STB_LOCAL, elf.STT_SECTION), shndx, 0)
	ctxt.loader.SetSymElfSym(s, int32(ctxt.numelfsym))
	ctxt.numelfsym++
}

func genelfsym(ctxt *Link, elfbind elf.SymBind) {
	ldr := ctxt.loader

	// runtime.text marker symbol(s).
	s := ldr.Lookup("runtime.text", 0)
	putelfsym(ctxt, s, elf.STT_FUNC, elfbind)
	for k, sect := range Segtext.Sections[1:] {
		n := k + 1
		if sect.Name != ".text" || (ctxt.IsAIX() && ctxt.IsExternal()) {
			// On AIX, runtime.text.X are symbols already in the symtab.
			break
		}
		s = ldr.Lookup(fmt.Sprintf("runtime.text.%d", n), 0)
		if s == 0 {
			break
		}
		if !ldr.SymType(s).IsText() {
			panic("unexpected type for runtime.text symbol")
		}
		putelfsym(ctxt, s, elf.STT_FUNC, elfbind)
	}

	// Text symbols.
	for _, s := range ctxt.Textp {
		putelfsym(ctxt, s, elf.STT_FUNC, elfbind)
	}

	// runtime.etext marker symbol.
	s = ldr.Lookup("runtime.etext", 0)
	if ldr.SymType(s).IsText() {
		putelfsym(ctxt, s, elf.STT_FUNC, elfbind)
	}

	shouldBeInSymbolTable := func(s loader.Sym) bool {
		if ldr.AttrNotInSymbolTable(s) {
			return false
		}
		// FIXME: avoid having to do name inspections here.
		// NB: the restrictions below on file local symbols are a bit
		// arbitrary -- if it turns out we need nameless static
		// symbols they could be relaxed/removed.
		sn := ldr.SymName(s)
		if (sn == "" || sn[0] == '.') && ldr.IsFileLocal(s) {
			panic(fmt.Sprintf("unexpected file local symbol %d %s<%d>\n",
				s, sn, ldr.SymVersion(s)))
		}
		if (sn == "" || sn[0] == '.') && !ldr.IsFileLocal(s) {
			return false
		}
		return true
	}

	// Data symbols.
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if !ldr.AttrReachable(s) {
			continue
		}
		st := ldr.SymType(s)
		if st >= sym.SELFRXSECT && st < sym.SXREF {
			typ := elf.STT_OBJECT
			if st == sym.STLSBSS {
				if ctxt.IsInternal() {
					continue
				}
				typ = elf.STT_TLS
			}
			if !shouldBeInSymbolTable(s) {
				continue
			}
			putelfsym(ctxt, s, typ, elfbind)
			continue
		}
		if st == sym.SHOSTOBJ || st == sym.SDYNIMPORT || st == sym.SUNDEFEXT {
			putelfsym(ctxt, s, ldr.SymElfType(s), elfbind)
		}
	}
}

func asmElfSym(ctxt *Link) {

	// the first symbol entry is reserved
	putelfsyment(ctxt.Out, 0, 0, 0, elf.ST_INFO(elf.STB_LOCAL, elf.STT_NOTYPE), 0, 0)

	dwarfaddelfsectionsyms(ctxt)

	// Some linkers will add a FILE sym if one is not present.
	// Avoid having the working directory inserted into the symbol table.
	// It is added with a name to avoid problems with external linking
	// encountered on some versions of Solaris. See issue #14957.
	putelfsyment(ctxt.Out, putelfstr("go.go"), 0, 0, elf.ST_INFO(elf.STB_LOCAL, elf.STT_FILE), elf.SHN_ABS, 0)
	ctxt.numelfsym++

	bindings := []elf.SymBind{elf.STB_LOCAL, elf.STB_GLOBAL}
	for _, elfbind := range bindings {
		if elfbind == elf.STB_GLOBAL {
			elfglobalsymndx = ctxt.numelfsym
		}
		genelfsym(ctxt, elfbind)
	}
}

func putplan9sym(ctxt *Link, ldr *loader.Loader, s loader.Sym, char SymbolType) {
	t := int(char)
	if ldr.IsFileLocal(s) {
		t += 'a' - 'A'
	}
	l := 4
	addr := ldr.SymValue(s)
	if ctxt.IsAMD64() && !flag8 {
		ctxt.Out.Write32b(uint32(addr >> 32))
		l = 8
	}

	ctxt.Out.Write32b(uint32(addr))
	ctxt.Out.Write8(uint8(t + 0x80)) /* 0x80 is variable length */

	name := ldr.SymName(s)
	name = mangleABIName(ctxt, ldr, s, name)
	ctxt.Out.WriteString(name)
	ctxt.Out.Write8(0)

	symSize += int32(l) + 1 + int32(len(name)) + 1
}

func asmbPlan9Sym(ctxt *Link) {
	ldr := ctxt.loader

	// Add special runtime.text and runtime.etext symbols.
	s := ldr.Lookup("runtime.text", 0)
	if ldr.SymType(s).IsText() {
		putplan9sym(ctxt, ldr, s, TextSym)
	}
	s = ldr.Lookup("runtime.etext", 0)
	if ldr.SymType(s).IsText() {
		putplan9sym(ctxt, ldr, s, TextSym)
	}

	// Add text symbols.
	for _, s := range ctxt.Textp {
		putplan9sym(ctxt, ldr, s, TextSym)
	}

	shouldBeInSymbolTable := func(s loader.Sym) bool {
		if ldr.AttrNotInSymbolTable(s) {
			return false
		}
		name := ldr.SymName(s) // TODO: try not to read the name
		if name == "" || name[0] == '.' {
			return false
		}
		return true
	}

	// Add data symbols and external references.
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if !ldr.AttrReachable(s) {
			continue
		}
		t := ldr.SymType(s)
		if t >= sym.SELFRXSECT && t < sym.SXREF { // data sections handled in dodata
			if t == sym.STLSBSS {
				continue
			}
			if !shouldBeInSymbolTable(s) {
				continue
			}
			char := DataSym
			if t == sym.SBSS || t == sym.SNOPTRBSS {
				char = BSSSym
			}
			putplan9sym(ctxt, ldr, s, char)
		}
	}
}

// Create a table with information on the text sections.
// Return the symbol of the table, and number of sections.
func textsectionmap(ctxt *Link) (loader.Sym, uint32) {
	ldr := ctxt.loader
	t := ldr.CreateSymForUpdate("runtime.textsectionmap", 0)
	t.SetType(sym.SRODATA)
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
		// The fields written should match runtime/symtab.go:textsect.
		// They are designed to minimize runtime calculations.
		vaddr := sect.Vaddr - textbase
		off = t.SetUint(ctxt.Arch, off, vaddr) // field vaddr
		end := vaddr + sect.Length
		off = t.SetUint(ctxt.Arch, off, end) // field end
		name := "runtime.text"
		if n != 0 {
			name = fmt.Sprintf("runtime.text.%d", n)
		}
		s := ldr.Lookup(name, 0)
		if s == 0 {
			ctxt.Errorf(s, "Unable to find symbol %s\n", name)
		}
		off = t.SetAddr(ctxt.Arch, off, s) // field baseaddr
		n++
	}
	return t.Sym(), uint32(n)
}

func (ctxt *Link) symtab(pcln *pclntab) []sym.SymKind {
	ldr := ctxt.loader

	if !ctxt.IsAIX() && !ctxt.IsWasm() {
		switch ctxt.BuildMode {
		case BuildModeCArchive, BuildModeCShared:
			s := ldr.Lookup(*flagEntrySymbol, sym.SymVerABI0)
			if s != 0 {
				addinitarrdata(ctxt, ldr, s)
			}
		}
	}

	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	ctxt.xdefine("runtime.rodata", sym.SRODATA, 0)
	ctxt.xdefine("runtime.erodata", sym.SRODATAEND, 0)
	ctxt.xdefine("runtime.types", sym.SRODATA, 0)
	ctxt.xdefine("runtime.etypes", sym.SRODATA, 0)
	ctxt.xdefine("runtime.noptrdata", sym.SNOPTRDATA, 0)
	ctxt.xdefine("runtime.enoptrdata", sym.SNOPTRDATAEND, 0)
	ctxt.xdefine("runtime.data", sym.SDATA, 0)
	ctxt.xdefine("runtime.edata", sym.SDATAEND, 0)
	ctxt.xdefine("runtime.bss", sym.SBSS, 0)
	ctxt.xdefine("runtime.ebss", sym.SBSS, 0)
	ctxt.xdefine("runtime.noptrbss", sym.SNOPTRBSS, 0)
	ctxt.xdefine("runtime.enoptrbss", sym.SNOPTRBSS, 0)
	ctxt.xdefine("runtime.covctrs", sym.SNOPTRBSS, 0)
	ctxt.xdefine("runtime.ecovctrs", sym.SNOPTRBSS, 0)
	ctxt.xdefine("runtime.end", sym.SBSS, 0)
	ctxt.xdefine("runtime.epclntab", sym.SRODATA, 0)
	ctxt.xdefine("runtime.esymtab", sym.SRODATA, 0)

	// garbage collection symbols
	s := ldr.CreateSymForUpdate("runtime.gcdata", 0)
	s.SetType(sym.SRODATA)
	s.SetSize(0)
	ctxt.xdefine("runtime.egcdata", sym.SRODATA, 0)

	s = ldr.CreateSymForUpdate("runtime.gcbss", 0)
	s.SetType(sym.SRODATA)
	s.SetSize(0)
	ctxt.xdefine("runtime.egcbss", sym.SRODATA, 0)

	// pseudo-symbols to mark locations of type, string, and go string data.
	var symtype, symtyperel loader.Sym
	if !ctxt.DynlinkingGo() {
		if ctxt.UseRelro() && (ctxt.BuildMode == BuildModeCArchive || ctxt.BuildMode == BuildModeCShared || ctxt.BuildMode == BuildModePIE) {
			s = ldr.CreateSymForUpdate("type:*", 0)
			s.SetType(sym.STYPE)
			s.SetSize(0)
			s.SetAlign(int32(ctxt.Arch.PtrSize))
			symtype = s.Sym()

			s = ldr.CreateSymForUpdate("typerel.*", 0)
			s.SetType(sym.STYPERELRO)
			s.SetSize(0)
			s.SetAlign(int32(ctxt.Arch.PtrSize))
			symtyperel = s.Sym()
		} else {
			s = ldr.CreateSymForUpdate("type:*", 0)
			s.SetType(sym.STYPE)
			s.SetSize(0)
			s.SetAlign(int32(ctxt.Arch.PtrSize))
			symtype = s.Sym()
			symtyperel = s.Sym()
		}
		setCarrierSym(sym.STYPE, symtype)
		setCarrierSym(sym.STYPERELRO, symtyperel)
	}

	groupSym := func(name string, t sym.SymKind) loader.Sym {
		s := ldr.CreateSymForUpdate(name, 0)
		s.SetType(t)
		s.SetSize(0)
		s.SetAlign(int32(ctxt.Arch.PtrSize))
		s.SetLocal(true)
		setCarrierSym(t, s.Sym())
		return s.Sym()
	}
	var (
		symgostring = groupSym("go:string.*", sym.SGOSTRING)
		symgofunc   = groupSym("go:func.*", sym.SGOFUNC)
		symgcbits   = groupSym("runtime.gcbits.*", sym.SGCBITS)
	)

	symgofuncrel := symgofunc
	if ctxt.UseRelro() {
		symgofuncrel = groupSym("go:funcrel.*", sym.SGOFUNCRELRO)
	}

	symt := ldr.CreateSymForUpdate("runtime.symtab", 0)
	symt.SetType(sym.SSYMTAB)
	symt.SetSize(0)
	symt.SetLocal(true)

	// assign specific types so that they sort together.
	// within a type they sort by size, so the .* symbols
	// just defined above will be first.
	// hide the specific symbols.
	// Some of these symbol section conditions are duplicated
	// in cmd/internal/obj.contentHashSection.
	nsym := loader.Sym(ldr.NSym())
	symGroupType := make([]sym.SymKind, nsym)
	for s := loader.Sym(1); s < nsym; s++ {
		if (!ctxt.IsExternal() && ldr.IsFileLocal(s) && !ldr.IsFromAssembly(s) && ldr.SymPkg(s) != "") || (ctxt.LinkMode == LinkInternal && ldr.SymType(s) == sym.SCOVERAGE_COUNTER) {
			ldr.SetAttrNotInSymbolTable(s, true)
		}
		if !ldr.AttrReachable(s) || ldr.AttrSpecial(s) || (ldr.SymType(s) != sym.SRODATA && ldr.SymType(s) != sym.SGOFUNC) {
			continue
		}

		name := ldr.SymName(s)
		switch {
		case strings.HasPrefix(name, "go:string."):
			symGroupType[s] = sym.SGOSTRING
			ldr.SetAttrNotInSymbolTable(s, true)
			ldr.SetCarrierSym(s, symgostring)

		case strings.HasPrefix(name, "runtime.gcbits."),
			strings.HasPrefix(name, "type:.gcprog."):
			symGroupType[s] = sym.SGCBITS
			ldr.SetAttrNotInSymbolTable(s, true)
			ldr.SetCarrierSym(s, symgcbits)

		case strings.HasSuffix(name, "·f"):
			if !ctxt.DynlinkingGo() {
				ldr.SetAttrNotInSymbolTable(s, true)
			}
			if ctxt.UseRelro() {
				symGroupType[s] = sym.SGOFUNCRELRO
				if !ctxt.DynlinkingGo() {
					ldr.SetCarrierSym(s, symgofuncrel)
				}
			} else {
				symGroupType[s] = sym.SGOFUNC
				ldr.SetCarrierSym(s, symgofunc)
			}

		case strings.HasPrefix(name, "gcargs."),
			strings.HasPrefix(name, "gclocals."),
			strings.HasPrefix(name, "gclocals·"),
			ldr.SymType(s) == sym.SGOFUNC && s != symgofunc, // inltree, see pcln.go
			strings.HasSuffix(name, ".opendefer"),
			strings.HasSuffix(name, ".arginfo0"),
			strings.HasSuffix(name, ".arginfo1"),
			strings.HasSuffix(name, ".argliveinfo"),
			strings.HasSuffix(name, ".wrapinfo"),
			strings.HasSuffix(name, ".args_stackmap"),
			strings.HasSuffix(name, ".stkobj"):
			ldr.SetAttrNotInSymbolTable(s, true)
			symGroupType[s] = sym.SGOFUNC
			ldr.SetCarrierSym(s, symgofunc)
			if ctxt.Debugvlog != 0 {
				align := ldr.SymAlign(s)
				liveness += (ldr.SymSize(s) + int64(align) - 1) &^ (int64(align) - 1)
			}

		// Note: Check for "type:" prefix after checking for .arginfo1 suffix.
		// That way symbols like "type:.eq.[2]interface {}.arginfo1" that belong
		// in go:func.* end up there.
		case strings.HasPrefix(name, "type:"):
			if !ctxt.DynlinkingGo() {
				ldr.SetAttrNotInSymbolTable(s, true)
			}
			if ctxt.UseRelro() {
				symGroupType[s] = sym.STYPERELRO
				if symtyperel != 0 {
					ldr.SetCarrierSym(s, symtyperel)
				}
			} else {
				symGroupType[s] = sym.STYPE
				if symtyperel != 0 {
					ldr.SetCarrierSym(s, symtype)
				}
			}
		}
	}

	if ctxt.BuildMode == BuildModeShared {
		abihashgostr := ldr.CreateSymForUpdate("go:link.abihash."+filepath.Base(*flagOutfile), 0)
		abihashgostr.SetType(sym.SRODATA)
		hashsym := ldr.LookupOrCreateSym("go:link.abihashbytes", 0)
		abihashgostr.AddAddr(ctxt.Arch, hashsym)
		abihashgostr.AddUint(ctxt.Arch, uint64(ldr.SymSize(hashsym)))
	}
	if ctxt.BuildMode == BuildModePlugin || ctxt.CanUsePlugins() {
		for _, l := range ctxt.Library {
			s := ldr.CreateSymForUpdate("go:link.pkghashbytes."+l.Pkg, 0)
			s.SetType(sym.SRODATA)
			s.SetSize(int64(len(l.Fingerprint)))
			s.SetData(l.Fingerprint[:])
			str := ldr.CreateSymForUpdate("go:link.pkghash."+l.Pkg, 0)
			str.SetType(sym.SRODATA)
			str.AddAddr(ctxt.Arch, s.Sym())
			str.AddUint(ctxt.Arch, uint64(len(l.Fingerprint)))
		}
	}

	textsectionmapSym, nsections := textsectionmap(ctxt)

	// Information about the layout of the executable image for the
	// runtime to use. Any changes here must be matched by changes to
	// the definition of moduledata in runtime/symtab.go.
	// This code uses several global variables that are set by pcln.go:pclntab.
	moduledata := ldr.MakeSymbolUpdater(ctxt.Moduledata)

	slice := func(sym loader.Sym, len uint64) {
		moduledata.AddAddr(ctxt.Arch, sym)
		moduledata.AddUint(ctxt.Arch, len)
		moduledata.AddUint(ctxt.Arch, len)
	}

	sliceSym := func(sym loader.Sym) {
		slice(sym, uint64(ldr.SymSize(sym)))
	}

	nilSlice := func() {
		moduledata.AddUint(ctxt.Arch, 0)
		moduledata.AddUint(ctxt.Arch, 0)
		moduledata.AddUint(ctxt.Arch, 0)
	}

	// The pcHeader
	moduledata.AddAddr(ctxt.Arch, pcln.pcheader)

	// The function name slice
	sliceSym(pcln.funcnametab)

	// The cutab slice
	sliceSym(pcln.cutab)

	// The filetab slice
	sliceSym(pcln.filetab)

	// The pctab slice
	sliceSym(pcln.pctab)

	// The pclntab slice
	slice(pcln.pclntab, uint64(ldr.SymSize(pcln.pclntab)))

	// The ftab slice
	slice(pcln.pclntab, uint64(pcln.nfunc+1))

	// findfunctab
	moduledata.AddAddr(ctxt.Arch, pcln.findfunctab)
	// minpc, maxpc
	moduledata.AddAddr(ctxt.Arch, pcln.firstFunc)
	moduledata.AddAddrPlus(ctxt.Arch, pcln.lastFunc, ldr.SymSize(pcln.lastFunc))
	// pointers to specific parts of the module
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.text", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.etext", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.noptrdata", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.enoptrdata", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.data", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.edata", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.bss", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.ebss", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.noptrbss", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.enoptrbss", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.covctrs", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.ecovctrs", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.end", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.gcdata", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.gcbss", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.types", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.etypes", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("runtime.rodata", 0))
	moduledata.AddAddr(ctxt.Arch, ldr.Lookup("go:func.*", 0))

	if ctxt.IsAIX() && ctxt.IsExternal() {
		// Add R_XCOFFREF relocation to prevent ld's garbage collection of
		// the following symbols. They might not be referenced in the program.
		addRef := func(name string) {
			s := ldr.Lookup(name, 0)
			if s == 0 {
				return
			}
			r, _ := moduledata.AddRel(objabi.R_XCOFFREF)
			r.SetSym(s)
			r.SetSiz(uint8(ctxt.Arch.PtrSize))
		}
		addRef("runtime.rodata")
		addRef("runtime.erodata")
		addRef("runtime.epclntab")
		// As we use relative addressing for text symbols in functab, it is
		// important that the offsets we computed stay unchanged by the external
		// linker, i.e. all symbols in Textp should not be removed.
		// Most of them are actually referenced (our deadcode pass ensures that),
		// except go:buildid which is generated late and not used by the program.
		addRef("go:buildid")
	}
	if ctxt.IsAIX() {
		// On AIX, an R_ADDR relocation from an RODATA symbol to a DATA symbol
		// does not work. See data.go:relocsym, case R_ADDR.
		// Here we record the unrelocated address in aixStaticDataBase (it is
		// unrelocated as it is in RODATA) so we can compute the delta at
		// run time.
		sb := ldr.CreateSymForUpdate("runtime.aixStaticDataBase", 0)
		sb.SetSize(0)
		sb.AddAddr(ctxt.Arch, ldr.Lookup("runtime.data", 0))
		sb.SetType(sym.SRODATA)
	}

	// text section information
	slice(textsectionmapSym, uint64(nsections))

	// The typelinks slice
	typelinkSym := ldr.Lookup("runtime.typelink", 0)
	ntypelinks := uint64(ldr.SymSize(typelinkSym)) / 4
	slice(typelinkSym, ntypelinks)

	// The itablinks slice
	itablinkSym := ldr.Lookup("runtime.itablink", 0)
	nitablinks := uint64(ldr.SymSize(itablinkSym)) / uint64(ctxt.Arch.PtrSize)
	slice(itablinkSym, nitablinks)

	// The ptab slice
	if ptab := ldr.Lookup("go:plugin.tabs", 0); ptab != 0 && ldr.AttrReachable(ptab) {
		ldr.SetAttrLocal(ptab, true)
		if ldr.SymType(ptab) != sym.SRODATA {
			panic(fmt.Sprintf("go:plugin.tabs is %v, not SRODATA", ldr.SymType(ptab)))
		}
		nentries := uint64(len(ldr.Data(ptab)) / 8) // sizeof(nameOff) + sizeof(typeOff)
		slice(ptab, nentries)
	} else {
		nilSlice()
	}

	if ctxt.BuildMode == BuildModePlugin {
		addgostring(ctxt, ldr, moduledata, "go:link.thispluginpath", objabi.PathToPrefix(*flagPluginPath))

		pkghashes := ldr.CreateSymForUpdate("go:link.pkghashes", 0)
		pkghashes.SetLocal(true)
		pkghashes.SetType(sym.SRODATA)

		for i, l := range ctxt.Library {
			// pkghashes[i].name
			addgostring(ctxt, ldr, pkghashes, fmt.Sprintf("go:link.pkgname.%d", i), l.Pkg)
			// pkghashes[i].linktimehash
			addgostring(ctxt, ldr, pkghashes, fmt.Sprintf("go:link.pkglinkhash.%d", i), string(l.Fingerprint[:]))
			// pkghashes[i].runtimehash
			hash := ldr.Lookup("go:link.pkghash."+l.Pkg, 0)
			pkghashes.AddAddr(ctxt.Arch, hash)
		}
		slice(pkghashes.Sym(), uint64(len(ctxt.Library)))
	} else {
		moduledata.AddUint(ctxt.Arch, 0) // pluginpath
		moduledata.AddUint(ctxt.Arch, 0)
		nilSlice() // pkghashes slice
	}
	// Add inittasks slice
	t := ctxt.mainInittasks
	if t != 0 {
		moduledata.AddAddr(ctxt.Arch, t)
		moduledata.AddUint(ctxt.Arch, uint64(ldr.SymSize(t)/int64(ctxt.Arch.PtrSize)))
		moduledata.AddUint(ctxt.Arch, uint64(ldr.SymSize(t)/int64(ctxt.Arch.PtrSize)))
	} else {
		// Some build modes have no inittasks, like a shared library.
		// Its inittask list will be constructed by a higher-level
		// linking step.
		// This branch can also happen if there are no init tasks at all.
		moduledata.AddUint(ctxt.Arch, 0)
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
		addgostring(ctxt, ldr, moduledata, "go:link.thismodulename", thismodulename)

		modulehashes := ldr.CreateSymForUpdate("go:link.abihashes", 0)
		modulehashes.SetLocal(true)
		modulehashes.SetType(sym.SRODATA)

		for i, shlib := range ctxt.Shlibs {
			// modulehashes[i].modulename
			modulename := filepath.Base(shlib.Path)
			addgostring(ctxt, ldr, modulehashes, fmt.Sprintf("go:link.libname.%d", i), modulename)

			// modulehashes[i].linktimehash
			addgostring(ctxt, ldr, modulehashes, fmt.Sprintf("go:link.linkhash.%d", i), string(shlib.Hash))

			// modulehashes[i].runtimehash
			abihash := ldr.LookupOrCreateSym("go:link.abihash."+modulename, 0)
			ldr.SetAttrReachable(abihash, true)
			modulehashes.AddAddr(ctxt.Arch, abihash)
		}

		slice(modulehashes.Sym(), uint64(len(ctxt.Shlibs)))
	} else {
		moduledata.AddUint(ctxt.Arch, 0) // modulename
		moduledata.AddUint(ctxt.Arch, 0)
		nilSlice() // moduleshashes slice
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
	moduledatatype := ldr.Lookup("type:runtime.moduledata", 0)
	moduledata.SetSize(decodetypeSize(ctxt.Arch, ldr.Data(moduledatatype)))
	moduledata.Grow(moduledata.Size())

	lastmoduledatap := ldr.CreateSymForUpdate("runtime.lastmoduledatap", 0)
	if lastmoduledatap.Type() != sym.SDYNIMPORT {
		lastmoduledatap.SetType(sym.SNOPTRDATA)
		lastmoduledatap.SetSize(0) // overwrite existing value
		lastmoduledatap.SetData(nil)
		lastmoduledatap.AddAddr(ctxt.Arch, moduledata.Sym())
	}
	return symGroupType
}

// CarrierSymByType tracks carrier symbols and their sizes.
var CarrierSymByType [sym.SXREF]struct {
	Sym  loader.Sym
	Size int64
}

func setCarrierSym(typ sym.SymKind, s loader.Sym) {
	if CarrierSymByType[typ].Sym != 0 {
		panic(fmt.Sprintf("carrier symbol for type %v already set", typ))
	}
	CarrierSymByType[typ].Sym = s
}

func setCarrierSize(typ sym.SymKind, sz int64) {
	if typ == sym.Sxxx {
		panic("setCarrierSize(Sxxx)")
	}
	if CarrierSymByType[typ].Size != 0 {
		panic(fmt.Sprintf("carrier symbol size for type %v already set", typ))
	}
	CarrierSymByType[typ].Size = sz
}

func isStaticTmp(name string) bool {
	return strings.Contains(name, "."+obj.StaticNamePrefix)
}

// Mangle function name with ABI information.
func mangleABIName(ctxt *Link, ldr *loader.Loader, x loader.Sym, name string) string {
	// For functions with ABI wrappers, we have to make sure that we
	// don't wind up with two symbol table entries with the same
	// name (since this will generated an error from the external
	// linker). If we have wrappers, keep the ABIInternal name
	// unmangled since we want cross-load-module calls to target
	// ABIInternal, and rename other symbols.
	//
	// TODO: avoid the ldr.Lookup calls below by instead using an aux
	// sym or marker relocation to associate the wrapper with the
	// wrapped function.
	if !buildcfg.Experiment.RegabiWrappers {
		return name
	}

	if ldr.SymType(x).IsText() && ldr.SymVersion(x) != sym.SymVerABIInternal && ldr.SymVersion(x) < sym.SymVerStatic {
		if s2 := ldr.Lookup(name, sym.SymVerABIInternal); s2 != 0 && ldr.SymType(s2).IsText() {
			name = fmt.Sprintf("%s.abi%d", name, ldr.SymVersion(x))
		}
	}

	// When loading a shared library, if a symbol has only one ABI,
	// and the name is not mangled, we don't know what ABI it is.
	// So we always mangle ABIInternal function name in shared linkage,
	// except symbols that are exported to C. Type symbols are always
	// ABIInternal so they are not mangled.
	if ctxt.IsShared() {
		if ldr.SymType(x).IsText() && ldr.SymVersion(x) == sym.SymVerABIInternal && !ldr.AttrCgoExport(x) && !strings.HasPrefix(name, "type:") {
			name = fmt.Sprintf("%s.abiinternal", name)
		}
	}

	return name
}
