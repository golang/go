// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/sym"
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
