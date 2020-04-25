// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/sys"
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

func elfWriteDynEntSym(arch *sys.Arch, s *sym.Symbol, tag int, t *sym.Symbol) {
	Elfwritedynentsymplus(arch, s, tag, t, 0)
}

func elfdynhash(ctxt *Link) {
	if !ctxt.IsELF {
		return
	}

	nsym := Nelfsym
	s := ctxt.Syms.Lookup(".hash", 0)
	s.Type = sym.SELFROSECT
	s.Attr |= sym.AttrReachable

	i := nsym
	nbucket := 1
	for i > 0 {
		nbucket++
		i >>= 1
	}

	var needlib *Elflib
	need := make([]*Elfaux, nsym)
	chain := make([]uint32, nsym)
	buckets := make([]uint32, nbucket)

	for _, sy := range ctxt.loader.Syms {
		if sy == nil {
			continue
		}
		if sy.Dynid <= 0 {
			continue
		}

		if sy.Dynimpvers() != "" {
			need[sy.Dynid] = addelflib(&needlib, sy.Dynimplib(), sy.Dynimpvers())
		}

		name := sy.Extname()
		hc := elfhash(name)

		b := hc % uint32(nbucket)
		chain[sy.Dynid] = buckets[b]
		buckets[b] = uint32(sy.Dynid)
	}

	// s390x (ELF64) hash table entries are 8 bytes
	if ctxt.Arch.Family == sys.S390X {
		s.AddUint64(ctxt.Arch, uint64(nbucket))
		s.AddUint64(ctxt.Arch, uint64(nsym))
		for i := 0; i < nbucket; i++ {
			s.AddUint64(ctxt.Arch, uint64(buckets[i]))
		}
		for i := 0; i < nsym; i++ {
			s.AddUint64(ctxt.Arch, uint64(chain[i]))
		}
	} else {
		s.AddUint32(ctxt.Arch, uint32(nbucket))
		s.AddUint32(ctxt.Arch, uint32(nsym))
		for i := 0; i < nbucket; i++ {
			s.AddUint32(ctxt.Arch, buckets[i])
		}
		for i := 0; i < nsym; i++ {
			s.AddUint32(ctxt.Arch, chain[i])
		}
	}

	// version symbols
	dynstr := ctxt.Syms.Lookup(".dynstr", 0)

	s = ctxt.Syms.Lookup(".gnu.version_r", 0)
	i = 2
	nfile := 0
	for l := needlib; l != nil; l = l.next {
		nfile++

		// header
		s.AddUint16(ctxt.Arch, 1) // table version
		j := 0
		for x := l.aux; x != nil; x = x.next {
			j++
		}
		s.AddUint16(ctxt.Arch, uint16(j))                         // aux count
		s.AddUint32(ctxt.Arch, uint32(Addstring(dynstr, l.file))) // file string offset
		s.AddUint32(ctxt.Arch, 16)                                // offset from header to first aux
		if l.next != nil {
			s.AddUint32(ctxt.Arch, 16+uint32(j)*16) // offset from this header to next
		} else {
			s.AddUint32(ctxt.Arch, 0)
		}

		for x := l.aux; x != nil; x = x.next {
			x.num = i
			i++

			// aux struct
			s.AddUint32(ctxt.Arch, elfhash(x.vers))                   // hash
			s.AddUint16(ctxt.Arch, 0)                                 // flags
			s.AddUint16(ctxt.Arch, uint16(x.num))                     // other - index we refer to this by
			s.AddUint32(ctxt.Arch, uint32(Addstring(dynstr, x.vers))) // version string offset
			if x.next != nil {
				s.AddUint32(ctxt.Arch, 16) // offset from this aux to next
			} else {
				s.AddUint32(ctxt.Arch, 0)
			}
		}
	}

	// version references
	s = ctxt.Syms.Lookup(".gnu.version", 0)

	for i := 0; i < nsym; i++ {
		if i == 0 {
			s.AddUint16(ctxt.Arch, 0) // first entry - no symbol
		} else if need[i] == nil {
			s.AddUint16(ctxt.Arch, 1) // global
		} else {
			s.AddUint16(ctxt.Arch, uint16(need[i].num))
		}
	}

	s = ctxt.Syms.Lookup(".dynamic", 0)
	elfverneed = nfile
	if elfverneed != 0 {
		elfWriteDynEntSym(ctxt.Arch, s, DT_VERNEED, ctxt.Syms.Lookup(".gnu.version_r", 0))
		elfWriteDynEnt(ctxt.Arch, s, DT_VERNEEDNUM, uint64(nfile))
		elfWriteDynEntSym(ctxt.Arch, s, DT_VERSYM, ctxt.Syms.Lookup(".gnu.version", 0))
	}

	sy := ctxt.Syms.Lookup(elfRelType+".plt", 0)
	if sy.Size > 0 {
		if elfRelType == ".rela" {
			elfWriteDynEnt(ctxt.Arch, s, DT_PLTREL, DT_RELA)
		} else {
			elfWriteDynEnt(ctxt.Arch, s, DT_PLTREL, DT_REL)
		}
		elfWriteDynEntSymSize(ctxt.Arch, s, DT_PLTRELSZ, sy)
		elfWriteDynEntSym(ctxt.Arch, s, DT_JMPREL, sy)
	}

	elfWriteDynEnt(ctxt.Arch, s, DT_NULL, 0)
}
