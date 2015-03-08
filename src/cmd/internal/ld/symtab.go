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

import "strings"

// Symbol table.

func putelfstr(s string) int {
	if len(Elfstrdat) == 0 && s != "" {
		// first entry must be empty string
		putelfstr("")
	}

	// Rewrite · to . for ASCII-only tools like DTrace (sigh)
	s = strings.Replace(s, "·", ".", -1)

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
	case '6',
		'7',
		'9':
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
	}

	xo := x
	for xo.Outer != nil {
		xo = xo.Outer
	}
	if xo.Sect == nil {
		Ctxt.Cursym = x
		Diag("missing section in putelfsym")
		return
	}

	if (xo.Sect.(*Section)).Elfsect == nil {
		Ctxt.Cursym = x
		Diag("missing ELF section in putelfsym")
		return
	}

	// One pass for each binding: STB_LOCAL, STB_GLOBAL,
	// maybe one day STB_WEAK.
	bind := STB_GLOBAL

	if ver != 0 || (x.Type&SHIDDEN != 0) {
		bind = STB_LOCAL
	}

	// In external linking mode, we have to invoke gcc with -rdynamic
	// to get the exported symbols put into the dynamic symbol table.
	// To avoid filling the dynamic table with lots of unnecessary symbols,
	// mark all Go symbols local (not global) in the final executable.
	if Linkmode == LinkExternal && x.Cgoexport&CgoExportStatic == 0 {
		bind = STB_LOCAL
	}

	if bind != elfbind {
		return
	}

	off := putelfstr(s)
	if Linkmode == LinkExternal {
		addr -= int64((xo.Sect.(*Section)).Vaddr)
	}
	other := STV_DEFAULT
	if x.Type&SHIDDEN != 0 {
		other = STV_HIDDEN
	}
	putelfsyment(off, addr, size, bind<<4|type_&0xf, ((xo.Sect.(*Section)).Elfsect.(*ElfShdr)).shnum, other)
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
	switch Thearch.Thechar {
	case '6':
		Cseek(sympos + 6)

	default:
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

	if Linkmode == LinkExternal && HEADTYPE != Hopenbsd {
		s := Linklookup(Ctxt, "runtime.tlsg", 0)
		if s.Sect == nil {
			Ctxt.Cursym = nil
			Diag("missing section for %s", s.Name)
			Errorexit()
		}

		if goos == "android" {
			// Android emulates runtime.tlsg as a regular variable.
			putelfsyment(putelfstr(s.Name), 0, s.Size, STB_LOCAL<<4|STT_OBJECT, ((s.Sect.(*Section)).Elfsect.(*ElfShdr)).shnum, 0)
		} else {
			putelfsyment(putelfstr(s.Name), 0, s.Size, STB_LOCAL<<4|STT_TLS, ((s.Sect.(*Section)).Elfsect.(*ElfShdr)).shnum, 0)
		}

		s.Elfsym = int32(numelfsym)
		numelfsym++
	}

	elfbind = STB_GLOBAL
	elfglobalsymndx = numelfsym
	genasmsym(putelfsym)

	var name string
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if s.Type != SHOSTOBJ && (s.Type != SDYNIMPORT || !s.Reachable) {
			continue
		}
		if s.Type == SDYNIMPORT {
			name = s.Extname
		} else {
			name = s.Name
		}
		putelfsyment(putelfstr(name), 0, 0, STB_GLOBAL<<4|STT_NOTYPE, 0, 0)
		s.Elfsym = int32(numelfsym)
		numelfsym++
	}
}

func putplan9sym(x *LSym, s string, t int, addr int64, size int64, ver int, go_ *LSym) {
	switch t {
	case 'T',
		'L',
		'D',
		'B':
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
		if HEADTYPE == Hplan9 && Thearch.Thechar == '6' && Debug['8'] == 0 {
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

func symtab() {
	dosymtype()

	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	xdefine("runtime.text", STEXT, 0)

	xdefine("runtime.etext", STEXT, 0)
	xdefine("runtime.typelink", SRODATA, 0)
	xdefine("runtime.etypelink", SRODATA, 0)
	xdefine("runtime.rodata", SRODATA, 0)
	xdefine("runtime.erodata", SRODATA, 0)
	xdefine("runtime.noptrdata", SNOPTRDATA, 0)
	xdefine("runtime.enoptrdata", SNOPTRDATA, 0)
	xdefine("runtime.data", SDATA, 0)
	xdefine("runtime.edata", SDATA, 0)
	xdefine("runtime.bss", SBSS, 0)
	xdefine("runtime.ebss", SBSS, 0)
	xdefine("runtime.noptrbss", SNOPTRBSS, 0)
	xdefine("runtime.enoptrbss", SNOPTRBSS, 0)
	xdefine("runtime.end", SBSS, 0)
	xdefine("runtime.epclntab", SRODATA, 0)
	xdefine("runtime.esymtab", SRODATA, 0)

	// garbage collection symbols
	s := Linklookup(Ctxt, "runtime.gcdata", 0)

	s.Type = SRODATA
	s.Size = 0
	s.Reachable = true
	xdefine("runtime.egcdata", SRODATA, 0)

	s = Linklookup(Ctxt, "runtime.gcbss", 0)
	s.Type = SRODATA
	s.Size = 0
	s.Reachable = true
	xdefine("runtime.egcbss", SRODATA, 0)

	// pseudo-symbols to mark locations of type, string, and go string data.
	s = Linklookup(Ctxt, "type.*", 0)

	s.Type = STYPE
	s.Size = 0
	s.Reachable = true
	symtype := s

	s = Linklookup(Ctxt, "go.string.*", 0)
	s.Type = SGOSTRING
	s.Size = 0
	s.Reachable = true
	symgostring := s

	s = Linklookup(Ctxt, "go.func.*", 0)
	s.Type = SGOFUNC
	s.Size = 0
	s.Reachable = true
	symgofunc := s

	symtypelink := Linklookup(Ctxt, "runtime.typelink", 0)

	symt = Linklookup(Ctxt, "runtime.symtab", 0)
	symt.Type = SSYMTAB
	symt.Size = 0
	symt.Reachable = true

	// assign specific types so that they sort together.
	// within a type they sort by size, so the .* symbols
	// just defined above will be first.
	// hide the specific symbols.
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if !s.Reachable || s.Special != 0 || s.Type != SRODATA {
			continue
		}
		if strings.HasPrefix(s.Name, "type.") {
			s.Type = STYPE
			s.Hide = 1
			s.Outer = symtype
		}

		if strings.HasPrefix(s.Name, "go.typelink.") {
			s.Type = STYPELINK
			s.Hide = 1
			s.Outer = symtypelink
		}

		if strings.HasPrefix(s.Name, "go.string.") {
			s.Type = SGOSTRING
			s.Hide = 1
			s.Outer = symgostring
		}

		if strings.HasPrefix(s.Name, "go.func.") {
			s.Type = SGOFUNC
			s.Hide = 1
			s.Outer = symgofunc
		}

		if strings.HasPrefix(s.Name, "gcargs.") || strings.HasPrefix(s.Name, "gclocals.") || strings.HasPrefix(s.Name, "gclocals·") {
			s.Type = SGOFUNC
			s.Hide = 1
			s.Outer = symgofunc
			s.Align = 4
			liveness += (s.Size + int64(s.Align) - 1) &^ (int64(s.Align) - 1)
		}
	}
}
