// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"strings"
)

// Temporary dumping around for sym.Symbol version of helper
// functions in symtab.go, still being used for some archs/oses.

func Asmelfsym2(ctxt *Link) {

	// the first symbol entry is reserved
	putelfsyment(ctxt.Out, 0, 0, 0, STB_LOCAL<<4|STT_NOTYPE, 0, 0)

	dwarfaddelfsectionsyms2(ctxt)

	// Some linkers will add a FILE sym if one is not present.
	// Avoid having the working directory inserted into the symbol table.
	// It is added with a name to avoid problems with external linking
	// encountered on some versions of Solaris. See issue #14957.
	putelfsyment(ctxt.Out, putelfstr("go.go"), 0, 0, STB_LOCAL<<4|STT_FILE, SHN_ABS, 0)
	ctxt.numelfsym++

	ctxt.elfbind = STB_LOCAL
	genasmsym(ctxt, putelfsym2)

	ctxt.elfbind = STB_GLOBAL
	elfglobalsymndx = ctxt.numelfsym
	genasmsym(ctxt, putelfsym2)
}

func putelfsectionsym2(ctxt *Link, out *OutBuf, s *sym.Symbol, shndx int) {
	putelfsyment(out, 0, 0, 0, STB_LOCAL<<4|STT_SECTION, shndx, 0)
	ctxt.loader.SetSymElfSym(loader.Sym(s.SymIdx), int32(ctxt.numelfsym))
	ctxt.numelfsym++
}

func putelfsym2(ctxt *Link, x *sym.Symbol, s string, t SymbolType, addr int64) {
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
	if xo.Outer != nil {
		xo = xo.Outer
	}

	var elfshnum int
	if xo.Type == sym.SDYNIMPORT || xo.Type == sym.SHOSTOBJ || xo.Type == sym.SUNDEFEXT {
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

	if s == x.Name {
		// We should use Extname for ELF symbol table.
		// TODO: maybe genasmsym should have done this. That function is too
		// overloaded and I would rather not change it for now.
		s = x.Extname()
	}

	// When dynamically linking, we create Symbols by reading the names from
	// the symbol tables of the shared libraries and so the names need to
	// match exactly. Tools like DTrace will have to wait for now.
	if !ctxt.DynlinkingGo() {
		// Rewrite · to . for ASCII-only tools like DTrace (sigh)
		s = strings.Replace(s, "·", ".", -1)
	}

	if ctxt.DynlinkingGo() && bind == STB_GLOBAL && ctxt.elfbind == STB_LOCAL && x.Type == sym.STEXT {
		// When dynamically linking, we want references to functions defined
		// in this module to always be to the function object, not to the
		// PLT. We force this by writing an additional local symbol for every
		// global function symbol and making all relocations against the
		// global symbol refer to this local symbol instead (see
		// (*sym.Symbol).ElfsymForReloc). This is approximately equivalent to the
		// ELF linker -Bsymbolic-functions option, but that is buggy on
		// several platforms.
		putelfsyment(ctxt.Out, putelfstr("local."+s), addr, size, STB_LOCAL<<4|typ&0xf, elfshnum, other)
		ctxt.loader.SetSymLocalElfSym(loader.Sym(x.SymIdx), int32(ctxt.numelfsym))
		ctxt.numelfsym++
		return
	} else if bind != ctxt.elfbind {
		return
	}

	putelfsyment(ctxt.Out, putelfstr(s), addr, size, bind<<4|typ&0xf, elfshnum, other)
	ctxt.loader.SetSymElfSym(loader.Sym(x.SymIdx), int32(ctxt.numelfsym))
	ctxt.numelfsym++
}
