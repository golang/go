// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
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
	"log"
	"strings"
)

func Symgrow(ctxt *Link, s *LSym, siz int64) {
	if int64(int(siz)) != siz {
		log.Fatalf("symgrow size %d too long", siz)
	}
	if int64(len(s.P)) >= siz {
		return
	}
	for cap(s.P) < int(siz) {
		s.P = append(s.P[:len(s.P)], 0)
	}
	s.P = s.P[:siz]
}

func Addrel(s *LSym) *Reloc {
	s.R = append(s.R, Reloc{})
	return &s.R[len(s.R)-1]
}

func setuintxx(ctxt *Link, s *LSym, off int64, v uint64, wid int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Reachable = true
	if s.Size < off+wid {
		s.Size = off + wid
		Symgrow(ctxt, s, s.Size)
	}

	switch wid {
	case 1:
		s.P[off] = uint8(v)
	case 2:
		ctxt.Arch.ByteOrder.PutUint16(s.P[off:], uint16(v))
	case 4:
		ctxt.Arch.ByteOrder.PutUint32(s.P[off:], uint32(v))
	case 8:
		ctxt.Arch.ByteOrder.PutUint64(s.P[off:], uint64(v))
	}

	return off + wid
}

func adduintxx(ctxt *Link, s *LSym, v uint64, wid int) int64 {
	off := s.Size
	setuintxx(ctxt, s, off, v, int64(wid))
	return off
}

func Adduint8(ctxt *Link, s *LSym, v uint8) int64 {
	return adduintxx(ctxt, s, uint64(v), 1)
}

func Adduint16(ctxt *Link, s *LSym, v uint16) int64 {
	return adduintxx(ctxt, s, uint64(v), 2)
}

func Adduint32(ctxt *Link, s *LSym, v uint32) int64 {
	return adduintxx(ctxt, s, uint64(v), 4)
}

func Adduint64(ctxt *Link, s *LSym, v uint64) int64 {
	return adduintxx(ctxt, s, v, 8)
}

func adduint(ctxt *Link, s *LSym, v uint64) int64 {
	return adduintxx(ctxt, s, v, Thearch.Intsize)
}

func setuint8(ctxt *Link, s *LSym, r int64, v uint8) int64 {
	return setuintxx(ctxt, s, r, uint64(v), 1)
}

func setuint32(ctxt *Link, s *LSym, r int64, v uint32) int64 {
	return setuintxx(ctxt, s, r, uint64(v), 4)
}

func Addaddrplus(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Reachable = true
	i := s.Size
	s.Size += int64(ctxt.Arch.Ptrsize)
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.Ptrsize)
	r.Type = R_ADDR
	r.Add = add
	return i + int64(r.Siz)
}

func Addpcrelplus(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Reachable = true
	i := s.Size
	s.Size += 4
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Add = add
	r.Type = R_PCREL
	r.Siz = 4
	return i + int64(r.Siz)
}

func Addaddr(ctxt *Link, s *LSym, t *LSym) int64 {
	return Addaddrplus(ctxt, s, t, 0)
}

func setaddrplus(ctxt *Link, s *LSym, off int64, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Reachable = true
	if off+int64(ctxt.Arch.Ptrsize) > s.Size {
		s.Size = off + int64(ctxt.Arch.Ptrsize)
		Symgrow(ctxt, s, s.Size)
	}

	r := Addrel(s)
	r.Sym = t
	r.Off = int32(off)
	r.Siz = uint8(ctxt.Arch.Ptrsize)
	r.Type = R_ADDR
	r.Add = add
	return off + int64(r.Siz)
}

func setaddr(ctxt *Link, s *LSym, off int64, t *LSym) int64 {
	return setaddrplus(ctxt, s, off, t, 0)
}

func addsize(ctxt *Link, s *LSym, t *LSym) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Reachable = true
	i := s.Size
	s.Size += int64(ctxt.Arch.Ptrsize)
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.Ptrsize)
	r.Type = R_SIZE
	return i + int64(r.Siz)
}

func addaddrplus4(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Reachable = true
	i := s.Size
	s.Size += 4
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = 4
	r.Type = R_ADDR
	r.Add = add
	return i + int64(r.Siz)
}

/*
 * divide-and-conquer list-link
 * sort of LSym* structures.
 * Used for the data block.
 */
func datcmp(s1 *LSym, s2 *LSym) int {
	if s1.Type != s2.Type {
		return int(s1.Type) - int(s2.Type)
	}

	// For ppc64, we want to interleave the .got and .toc sections
	// from input files.  Both are type SELFGOT, so in that case
	// fall through to the name comparison (conveniently, .got
	// sorts before .toc).
	if s1.Type != SELFGOT && s1.Size != s2.Size {
		if s1.Size < s2.Size {
			return -1
		}
		return +1
	}

	return stringsCompare(s1.Name, s2.Name)
}

func listnextp(s *LSym) **LSym {
	return &s.Next
}

func listsubp(s *LSym) **LSym {
	return &s.Sub
}

func listsort(l *LSym, cmp func(*LSym, *LSym) int, nextp func(*LSym) **LSym) *LSym {
	if l == nil || *nextp(l) == nil {
		return l
	}

	l1 := l
	l2 := l
	for {
		l2 = *nextp(l2)
		if l2 == nil {
			break
		}
		l2 = *nextp(l2)
		if l2 == nil {
			break
		}
		l1 = *nextp(l1)
	}

	l2 = *nextp(l1)
	*nextp(l1) = nil
	l1 = listsort(l, cmp, nextp)
	l2 = listsort(l2, cmp, nextp)

	/* set up lead element */
	if cmp(l1, l2) < 0 {
		l = l1
		l1 = *nextp(l1)
	} else {
		l = l2
		l2 = *nextp(l2)
	}

	le := l

	for {
		if l1 == nil {
			for l2 != nil {
				*nextp(le) = l2
				le = l2
				l2 = *nextp(l2)
			}

			*nextp(le) = nil
			break
		}

		if l2 == nil {
			for l1 != nil {
				*nextp(le) = l1
				le = l1
				l1 = *nextp(l1)
			}

			break
		}

		if cmp(l1, l2) < 0 {
			*nextp(le) = l1
			le = l1
			l1 = *nextp(l1)
		} else {
			*nextp(le) = l2
			le = l2
			l2 = *nextp(l2)
		}
	}

	*nextp(le) = nil
	return l
}

func relocsym(s *LSym) {
	var r *Reloc
	var rs *LSym
	var i16 int16
	var off int32
	var siz int32
	var fl int32
	var o int64

	Ctxt.Cursym = s
	for ri := int32(0); ri < int32(len(s.R)); ri++ {
		r = &s.R[ri]
		r.Done = 1
		off = r.Off
		siz = int32(r.Siz)
		if off < 0 || off+siz > int32(len(s.P)) {
			Diag("%s: invalid relocation %d+%d not in [%d,%d)", s.Name, off, siz, 0, len(s.P))
			continue
		}

		if r.Sym != nil && (r.Sym.Type&(SMASK|SHIDDEN) == 0 || r.Sym.Type&SMASK == SXREF) {
			// When putting the runtime but not main into a shared library
			// these symbols are undefined and that's OK.
			if Buildmode == BuildmodeShared && (r.Sym.Name == "main.main" || r.Sym.Name == "main.init") {
				r.Sym.Type = SDYNIMPORT
			} else {
				Diag("%s: not defined", r.Sym.Name)
				continue
			}
		}

		if r.Type >= 256 {
			continue
		}
		if r.Siz == 0 { // informational relocation - no work to do
			continue
		}

		// We need to be able to reference dynimport symbols when linking against
		// shared libraries, and Solaris needs it always
		if HEADTYPE != Hsolaris && r.Sym != nil && r.Sym.Type == SDYNIMPORT && !DynlinkingGo() {
			Diag("unhandled relocation for %s (type %d rtype %d)", r.Sym.Name, r.Sym.Type, r.Type)
		}
		if r.Sym != nil && r.Sym.Type != STLSBSS && !r.Sym.Reachable {
			Diag("unreachable sym in relocation: %s %s", s.Name, r.Sym.Name)
		}

		// Android emulates runtime.tlsg as a regular variable.
		if r.Type == R_TLS && goos == "android" {
			r.Type = R_ADDR
		}

		switch r.Type {
		default:
			o = 0
			if Thearch.Archreloc(r, s, &o) < 0 {
				Diag("unknown reloc %d", r.Type)
			}

		case R_TLS:
			if Linkmode == LinkInternal && Iself && Thearch.Thechar == '5' {
				// On ELF ARM, the thread pointer is 8 bytes before
				// the start of the thread-local data block, so add 8
				// to the actual TLS offset (r->sym->value).
				// This 8 seems to be a fundamental constant of
				// ELF on ARM (or maybe Glibc on ARM); it is not
				// related to the fact that our own TLS storage happens
				// to take up 8 bytes.
				o = 8 + r.Sym.Value

				break
			}

			r.Done = 0
			o = 0
			if Thearch.Thechar != '6' {
				o = r.Add
			}

		case R_TLS_LE:
			if Linkmode == LinkExternal && Iself && HEADTYPE != Hopenbsd {
				r.Done = 0
				r.Sym = Ctxt.Tlsg
				r.Xsym = Ctxt.Tlsg
				r.Xadd = r.Add
				o = 0
				if Thearch.Thechar != '6' {
					o = r.Add
				}
				break
			}

			o = int64(Ctxt.Tlsoffset) + r.Add

		case R_TLS_IE:
			if Linkmode == LinkExternal && Iself && HEADTYPE != Hopenbsd {
				r.Done = 0
				r.Sym = Ctxt.Tlsg
				r.Xsym = Ctxt.Tlsg
				r.Xadd = r.Add
				o = 0
				if Thearch.Thechar != '6' {
					o = r.Add
				}
				break
			}

			if Iself || Ctxt.Headtype == Hplan9 {
				o = int64(Ctxt.Tlsoffset) + r.Add
			} else if Ctxt.Headtype == Hwindows {
				o = r.Add
			} else {
				log.Fatalf("unexpected R_TLS_IE relocation for %s", Headstr(Ctxt.Headtype))
			}

		case R_ADDR:
			if Linkmode == LinkExternal && r.Sym.Type != SCONST {
				r.Done = 0

				// set up addend for eventual relocation via outer symbol.
				rs = r.Sym

				r.Xadd = r.Add
				for rs.Outer != nil {
					r.Xadd += Symaddr(rs) - Symaddr(rs.Outer)
					rs = rs.Outer
				}

				if rs.Type != SHOSTOBJ && rs.Type != SDYNIMPORT && rs.Sect == nil {
					Diag("missing section for %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if Iself {
					if Thearch.Thechar == '6' {
						o = 0
					}
				} else if HEADTYPE == Hdarwin {
					// ld64 for arm64 has a bug where if the address pointed to by o exists in the
					// symbol table (dynid >= 0), or is inside a symbol that exists in the symbol
					// table, then it will add o twice into the relocated value.
					// The workaround is that on arm64 don't ever add symaddr to o and always use
					// extern relocation by requiring rs->dynid >= 0.
					if rs.Type != SHOSTOBJ {
						if Thearch.Thechar == '7' && rs.Dynid < 0 {
							Diag("R_ADDR reloc to %s+%d is not supported on darwin/arm64", rs.Name, o)
						}
						if Thearch.Thechar != '7' {
							o += Symaddr(rs)
						}
					}
				} else if HEADTYPE == Hwindows {
					// nothing to do
				} else {
					Diag("unhandled pcrel relocation for %s", headstring)
				}

				break
			}

			o = Symaddr(r.Sym) + r.Add

			// On amd64, 4-byte offsets will be sign-extended, so it is impossible to
			// access more than 2GB of static data; fail at link time is better than
			// fail at runtime. See http://golang.org/issue/7980.
			// Instead of special casing only amd64, we treat this as an error on all
			// 64-bit architectures so as to be future-proof.
			if int32(o) < 0 && Thearch.Ptrsize > 4 && siz == 4 {
				Diag("non-pc-relative relocation address is too big: %#x (%#x + %#x)", uint64(o), Symaddr(r.Sym), r.Add)
				Errorexit()
			}

			// r->sym can be null when CALL $(constant) is transformed from absolute PC to relative PC call.
		case R_CALL, R_GOTPCREL, R_PCREL:
			if Linkmode == LinkExternal && r.Sym != nil && r.Sym.Type != SCONST && (r.Sym.Sect != Ctxt.Cursym.Sect || r.Type == R_GOTPCREL) {
				r.Done = 0

				// set up addend for eventual relocation via outer symbol.
				rs = r.Sym

				r.Xadd = r.Add
				for rs.Outer != nil {
					r.Xadd += Symaddr(rs) - Symaddr(rs.Outer)
					rs = rs.Outer
				}

				r.Xadd -= int64(r.Siz) // relative to address after the relocated chunk
				if rs.Type != SHOSTOBJ && rs.Type != SDYNIMPORT && rs.Sect == nil {
					Diag("missing section for %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if Iself {
					if Thearch.Thechar == '6' {
						o = 0
					}
				} else if HEADTYPE == Hdarwin {
					if r.Type == R_CALL {
						if rs.Type != SHOSTOBJ {
							o += int64(uint64(Symaddr(rs)) - (rs.Sect.(*Section)).Vaddr)
						}
						o -= int64(r.Off) // relative to section offset, not symbol
					} else {
						o += int64(r.Siz)
					}
				} else if HEADTYPE == Hwindows && Thearch.Thechar == '6' { // only amd64 needs PCREL
					// PE/COFF's PC32 relocation uses the address after the relocated
					// bytes as the base. Compensate by skewing the addend.
					o += int64(r.Siz)
					// GNU ld always add VirtualAddress of the .text section to the
					// relocated address, compensate that.
					o -= int64(s.Sect.(*Section).Vaddr - PEBASE)
				} else {
					Diag("unhandled pcrel relocation for %s", headstring)
				}

				break
			}

			o = 0
			if r.Sym != nil {
				o += Symaddr(r.Sym)
			}

			// NOTE: The (int32) cast on the next line works around a bug in Plan 9's 8c
			// compiler. The expression s->value + r->off + r->siz is int32 + int32 +
			// uchar, and Plan 9 8c incorrectly treats the expression as type uint32
			// instead of int32, causing incorrect values when sign extended for adding
			// to o. The bug only occurs on Plan 9, because this C program is compiled by
			// the standard host compiler (gcc on most other systems).
			o += r.Add - (s.Value + int64(r.Off) + int64(int32(r.Siz)))

		case R_SIZE:
			o = r.Sym.Size + r.Add
		}

		if r.Variant != RV_NONE {
			o = Thearch.Archrelocvariant(r, s, o)
		}

		if false {
			nam := "<nil>"
			if r.Sym != nil {
				nam = r.Sym.Name
			}
			fmt.Printf("relocate %s %#x (%#x+%#x, size %d) => %s %#x +%#x [type %d/%d, %x]\n", s.Name, s.Value+int64(off), s.Value, r.Off, r.Siz, nam, Symaddr(r.Sym), r.Add, r.Type, r.Variant, o)
		}
		switch siz {
		default:
			Ctxt.Cursym = s
			Diag("bad reloc size %#x for %s", uint32(siz), r.Sym.Name)
			fallthrough

			// TODO(rsc): Remove.
		case 1:
			s.P[off] = byte(int8(o))

		case 2:
			if o != int64(int16(o)) {
				Diag("relocation address is too big: %#x", o)
			}
			i16 = int16(o)
			Ctxt.Arch.ByteOrder.PutUint16(s.P[off:], uint16(i16))

		case 4:
			if r.Type == R_PCREL || r.Type == R_CALL {
				if o != int64(int32(o)) {
					Diag("pc-relative relocation address is too big: %#x", o)
				}
			} else {
				if o != int64(int32(o)) && o != int64(uint32(o)) {
					Diag("non-pc-relative relocation address is too big: %#x", uint64(o))
				}
			}

			fl = int32(o)
			Ctxt.Arch.ByteOrder.PutUint32(s.P[off:], uint32(fl))

		case 8:
			Ctxt.Arch.ByteOrder.PutUint64(s.P[off:], uint64(o))
		}
	}
}

func reloc() {
	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f reloc\n", obj.Cputime())
	}
	Bflush(&Bso)

	for s := Ctxt.Textp; s != nil; s = s.Next {
		relocsym(s)
	}
	for s := datap; s != nil; s = s.Next {
		relocsym(s)
	}
}

func dynrelocsym(s *LSym) {
	if HEADTYPE == Hwindows && Linkmode != LinkExternal {
		rel := Linklookup(Ctxt, ".rel", 0)
		if s == rel {
			return
		}
		var r *Reloc
		var targ *LSym
		for ri := 0; ri < len(s.R); ri++ {
			r = &s.R[ri]
			targ = r.Sym
			if targ == nil {
				continue
			}
			if !targ.Reachable {
				Diag("internal inconsistency: dynamic symbol %s is not reachable.", targ.Name)
			}
			if r.Sym.Plt == -2 && r.Sym.Got != -2 { // make dynimport JMP table for PE object files.
				targ.Plt = int32(rel.Size)
				r.Sym = rel
				r.Add = int64(targ.Plt)

				// jmp *addr
				if Thearch.Thechar == '8' {
					Adduint8(Ctxt, rel, 0xff)
					Adduint8(Ctxt, rel, 0x25)
					Addaddr(Ctxt, rel, targ)
					Adduint8(Ctxt, rel, 0x90)
					Adduint8(Ctxt, rel, 0x90)
				} else {
					Adduint8(Ctxt, rel, 0xff)
					Adduint8(Ctxt, rel, 0x24)
					Adduint8(Ctxt, rel, 0x25)
					addaddrplus4(Ctxt, rel, targ, 0)
					Adduint8(Ctxt, rel, 0x90)
				}
			} else if r.Sym.Plt >= 0 {
				r.Sym = rel
				r.Add = int64(targ.Plt)
			}
		}

		return
	}

	var r *Reloc
	for ri := 0; ri < len(s.R); ri++ {
		r = &s.R[ri]
		if r.Sym != nil && r.Sym.Type == SDYNIMPORT || r.Type >= 256 {
			if r.Sym != nil && !r.Sym.Reachable {
				Diag("internal inconsistency: dynamic symbol %s is not reachable.", r.Sym.Name)
			}
			Thearch.Adddynrel(s, r)
		}
	}
}

func dynreloc() {
	// -d suppresses dynamic loader format, so we may as well not
	// compute these sections or mark their symbols as reachable.
	if Debug['d'] != 0 && HEADTYPE != Hwindows {
		return
	}
	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f reloc\n", obj.Cputime())
	}
	Bflush(&Bso)

	for s := Ctxt.Textp; s != nil; s = s.Next {
		dynrelocsym(s)
	}
	for s := datap; s != nil; s = s.Next {
		dynrelocsym(s)
	}
	if Iself {
		elfdynhash()
	}
}

func blk(start *LSym, addr int64, size int64) {
	var sym *LSym

	for sym = start; sym != nil; sym = sym.Next {
		if sym.Type&SSUB == 0 && sym.Value >= addr {
			break
		}
	}

	eaddr := addr + size
	var ep []byte
	var p []byte
	for ; sym != nil; sym = sym.Next {
		if sym.Type&SSUB != 0 {
			continue
		}
		if sym.Value >= eaddr {
			break
		}
		Ctxt.Cursym = sym
		if sym.Value < addr {
			Diag("phase error: addr=%#x but sym=%#x type=%d", int64(addr), int64(sym.Value), sym.Type)
			Errorexit()
		}

		for ; addr < sym.Value; addr++ {
			Cput(0)
		}
		p = sym.P
		ep = p[len(sym.P):]
		for -cap(p) < -cap(ep) {
			Cput(uint8(p[0]))
			p = p[1:]
		}
		addr += int64(len(sym.P))
		for ; addr < sym.Value+sym.Size; addr++ {
			Cput(0)
		}
		if addr != sym.Value+sym.Size {
			Diag("phase error: addr=%#x value+size=%#x", int64(addr), int64(sym.Value)+sym.Size)
			Errorexit()
		}

		if sym.Value+sym.Size >= eaddr {
			break
		}
	}

	for ; addr < eaddr; addr++ {
		Cput(0)
	}
	Cflush()
}

func Codeblk(addr int64, size int64) {
	if Debug['a'] != 0 {
		fmt.Fprintf(&Bso, "codeblk [%#x,%#x) at offset %#x\n", addr, addr+size, Cpos())
	}

	blk(Ctxt.Textp, addr, size)

	/* again for printing */
	if Debug['a'] == 0 {
		return
	}

	var sym *LSym
	for sym = Ctxt.Textp; sym != nil; sym = sym.Next {
		if !sym.Reachable {
			continue
		}
		if sym.Value >= addr {
			break
		}
	}

	eaddr := addr + size
	var n int64
	var q []byte
	for ; sym != nil; sym = sym.Next {
		if !sym.Reachable {
			continue
		}
		if sym.Value >= eaddr {
			break
		}

		if addr < sym.Value {
			fmt.Fprintf(&Bso, "%-20s %.8x|", "_", uint64(int64(addr)))
			for ; addr < sym.Value; addr++ {
				fmt.Fprintf(&Bso, " %.2x", 0)
			}
			fmt.Fprintf(&Bso, "\n")
		}

		fmt.Fprintf(&Bso, "%.6x\t%-20s\n", uint64(int64(addr)), sym.Name)
		n = sym.Size
		q = sym.P

		for n >= 16 {
			fmt.Fprintf(&Bso, "%.6x\t%%-20.16I\n", uint64(addr), q)
			addr += 16
			q = q[16:]
			n -= 16
		}

		if n > 0 {
			fmt.Fprintf(&Bso, "%.6x\t%%-20.*I\n", uint64(addr), int(n), q)
		}
		addr += n
	}

	if addr < eaddr {
		fmt.Fprintf(&Bso, "%-20s %.8x|", "_", uint64(int64(addr)))
		for ; addr < eaddr; addr++ {
			fmt.Fprintf(&Bso, " %.2x", 0)
		}
	}

	Bflush(&Bso)
}

func Datblk(addr int64, size int64) {
	if Debug['a'] != 0 {
		fmt.Fprintf(&Bso, "datblk [%#x,%#x) at offset %#x\n", addr, addr+size, Cpos())
	}

	blk(datap, addr, size)

	/* again for printing */
	if Debug['a'] == 0 {
		return
	}

	var sym *LSym
	for sym = datap; sym != nil; sym = sym.Next {
		if sym.Value >= addr {
			break
		}
	}

	eaddr := addr + size
	var ep []byte
	var i int64
	var p []byte
	var r *Reloc
	var rsname string
	var typ string
	for ; sym != nil; sym = sym.Next {
		if sym.Value >= eaddr {
			break
		}
		if addr < sym.Value {
			fmt.Fprintf(&Bso, "\t%.8x| 00 ...\n", uint64(addr))
			addr = sym.Value
		}

		fmt.Fprintf(&Bso, "%s\n\t%.8x|", sym.Name, uint(addr))
		p = sym.P
		ep = p[len(sym.P):]
		for -cap(p) < -cap(ep) {
			if -cap(p) > -cap(sym.P) && int(-cap(p)+cap(sym.P))%16 == 0 {
				fmt.Fprintf(&Bso, "\n\t%.8x|", uint(addr+int64(-cap(p)+cap(sym.P))))
			}
			fmt.Fprintf(&Bso, " %.2x", p[0])
			p = p[1:]
		}

		addr += int64(len(sym.P))
		for ; addr < sym.Value+sym.Size; addr++ {
			fmt.Fprintf(&Bso, " %.2x", 0)
		}
		fmt.Fprintf(&Bso, "\n")

		if Linkmode == LinkExternal {
			for i = 0; i < int64(len(sym.R)); i++ {
				r = &sym.R[i]
				rsname = ""
				if r.Sym != nil {
					rsname = r.Sym.Name
				}
				typ = "?"
				switch r.Type {
				case R_ADDR:
					typ = "addr"

				case R_PCREL:
					typ = "pcrel"

				case R_CALL:
					typ = "call"
				}

				fmt.Fprintf(&Bso, "\treloc %.8x/%d %s %s+%#x [%#x]\n", uint(sym.Value+int64(r.Off)), r.Siz, typ, rsname, int64(r.Add), int64(r.Sym.Value+r.Add))
			}
		}
	}

	if addr < eaddr {
		fmt.Fprintf(&Bso, "\t%.8x| 00 ...\n", uint(addr))
	}
	fmt.Fprintf(&Bso, "\t%.8x|\n", uint(eaddr))
}

func strnput(s string, n int) {
	for ; n > 0 && s != ""; s = s[1:] {
		Cput(uint8(s[0]))
		n--
	}

	for n > 0 {
		Cput(0)
		n--
	}
}

var addstrdata_name string

func addstrdata1(arg string) {
	if strings.HasPrefix(arg, "VALUE:") {
		addstrdata(addstrdata_name, arg[6:])
	} else {
		addstrdata_name = arg
	}
}

func addstrdata(name string, value string) {
	p := fmt.Sprintf("%s.str", name)
	sp := Linklookup(Ctxt, p, 0)

	Addstring(sp, value)
	sp.Type = SRODATA

	s := Linklookup(Ctxt, name, 0)
	s.Size = 0
	s.Dupok = 1
	reachable := s.Reachable
	Addaddr(Ctxt, s, sp)
	adduintxx(Ctxt, s, uint64(len(value)), Thearch.Ptrsize)

	// addstring, addaddr, etc., mark the symbols as reachable.
	// In this case that is not necessarily true, so stick to what
	// we know before entering this function.
	s.Reachable = reachable

	sp.Reachable = reachable
}

func Addstring(s *LSym, str string) int64 {
	if s.Type == 0 {
		s.Type = SNOPTRDATA
	}
	s.Reachable = true
	r := int32(s.Size)
	n := len(str) + 1
	if s.Name == ".shstrtab" {
		elfsetstring(str, int(r))
	}
	Symgrow(Ctxt, s, int64(r)+int64(n))
	copy(s.P[r:], str)
	s.P[int(r)+len(str)] = 0
	s.Size += int64(n)
	return int64(r)
}

func addinitarrdata(s *LSym) {
	p := s.Name + ".ptr"
	sp := Linklookup(Ctxt, p, 0)
	sp.Type = SINITARR
	sp.Size = 0
	sp.Dupok = 1
	Addaddr(Ctxt, sp, s)
}

func dosymtype() {
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if len(s.P) > 0 {
			if s.Type == SBSS {
				s.Type = SDATA
			}
			if s.Type == SNOPTRBSS {
				s.Type = SNOPTRDATA
			}
		}
		// Create a new entry in the .init_array section that points to the
		// library initializer function.
		switch Buildmode {
		case BuildmodeCArchive, BuildmodeCShared:
			if s.Name == INITENTRY {
				addinitarrdata(s)
			}
		}
	}
}

func symalign(s *LSym) int32 {
	if s.Align != 0 {
		return s.Align
	}

	align := int32(Thearch.Maxalign)
	for int64(align) > s.Size && align > 1 {
		align >>= 1
	}
	if align < s.Align {
		align = s.Align
	}
	return align
}

func aligndatsize(datsize int64, s *LSym) int64 {
	return Rnd(datsize, int64(symalign(s)))
}

// maxalign returns the maximum required alignment for
// the list of symbols s; the list stops when s->type exceeds type.
func maxalign(s *LSym, type_ int) int32 {
	var align int32

	max := int32(0)
	for ; s != nil && int(s.Type) <= type_; s = s.Next {
		align = symalign(s)
		if max < align {
			max = align
		}
	}

	return max
}

// Helper object for building GC type programs.
type ProgGen struct {
	s        *LSym
	datasize int32
	data     [256 / obj.PointersPerByte]uint8
	pos      int64
}

func proggeninit(g *ProgGen, s *LSym) {
	g.s = s
	g.datasize = 0
	g.pos = 0
	g.data = [256 / obj.PointersPerByte]uint8{}
}

func proggenemit(g *ProgGen, v uint8) {
	Adduint8(Ctxt, g.s, v)
}

// Writes insData block from g->data.
func proggendataflush(g *ProgGen) {
	if g.datasize == 0 {
		return
	}
	proggenemit(g, obj.InsData)
	proggenemit(g, uint8(g.datasize))
	s := (g.datasize + obj.PointersPerByte - 1) / obj.PointersPerByte
	for i := int32(0); i < s; i++ {
		proggenemit(g, g.data[i])
	}
	g.datasize = 0
	g.data = [256 / obj.PointersPerByte]uint8{}
}

func proggendata(g *ProgGen, d uint8) {
	g.data[g.datasize/obj.PointersPerByte] |= d << uint((g.datasize%obj.PointersPerByte)*obj.BitsPerPointer)
	g.datasize++
	if g.datasize == 255 {
		proggendataflush(g)
	}
}

// Skip v bytes due to alignment, etc.
func proggenskip(g *ProgGen, off int64, v int64) {
	for i := off; i < off+v; i++ {
		if (i % int64(Thearch.Ptrsize)) == 0 {
			proggendata(g, obj.BitsScalar)
		}
	}
}

// Emit insArray instruction.
func proggenarray(g *ProgGen, length int64) {
	var i int32

	proggendataflush(g)
	proggenemit(g, obj.InsArray)
	for i = 0; i < int32(Thearch.Ptrsize); i, length = i+1, length>>8 {
		proggenemit(g, uint8(length))
	}
}

func proggenarrayend(g *ProgGen) {
	proggendataflush(g)
	proggenemit(g, obj.InsArrayEnd)
}

func proggenfini(g *ProgGen, size int64) {
	proggenskip(g, g.pos, size-g.pos)
	proggendataflush(g)
	proggenemit(g, obj.InsEnd)
}

// This function generates GC pointer info for global variables.
func proggenaddsym(g *ProgGen, s *LSym) {
	if s.Size == 0 {
		return
	}

	// Skip alignment hole from the previous symbol.
	proggenskip(g, g.pos, s.Value-g.pos)

	g.pos += s.Value - g.pos

	// The test for names beginning with . here is meant
	// to keep .dynamic and .dynsym from turning up as
	// conservative symbols. They should be marked SELFSECT
	// and not SDATA, but sometimes that doesn't happen.
	// Leave debugging the SDATA issue for the Go rewrite.

	if s.Gotype == nil && s.Size >= int64(Thearch.Ptrsize) && s.Name[0] != '.' {
		// conservative scan
		Diag("missing Go type information for global symbol: %s size %d", s.Name, int(s.Size))

		if (s.Size%int64(Thearch.Ptrsize) != 0) || (g.pos%int64(Thearch.Ptrsize) != 0) {
			Diag("proggenaddsym: unaligned conservative symbol %s: size=%d pos=%d", s.Name, s.Size, g.pos)
		}
		size := (s.Size + int64(Thearch.Ptrsize) - 1) / int64(Thearch.Ptrsize) * int64(Thearch.Ptrsize)
		if size < int64(32*Thearch.Ptrsize) {
			// Emit small symbols as data.
			for i := int64(0); i < size/int64(Thearch.Ptrsize); i++ {
				proggendata(g, obj.BitsPointer)
			}
		} else {
			// Emit large symbols as array.
			proggenarray(g, size/int64(Thearch.Ptrsize))

			proggendata(g, obj.BitsPointer)
			proggenarrayend(g)
		}

		g.pos = s.Value + size
	} else if s.Gotype == nil || decodetype_noptr(s.Gotype) != 0 || s.Size < int64(Thearch.Ptrsize) || s.Name[0] == '.' {
		// no scan
		if s.Size < int64(32*Thearch.Ptrsize) {
			// Emit small symbols as data.
			// This case also handles unaligned and tiny symbols, so tread carefully.
			for i := s.Value; i < s.Value+s.Size; i++ {
				if (i % int64(Thearch.Ptrsize)) == 0 {
					proggendata(g, obj.BitsScalar)
				}
			}
		} else {
			// Emit large symbols as array.
			if (s.Size%int64(Thearch.Ptrsize) != 0) || (g.pos%int64(Thearch.Ptrsize) != 0) {
				Diag("proggenaddsym: unaligned noscan symbol %s: size=%d pos=%d", s.Name, s.Size, g.pos)
			}
			proggenarray(g, s.Size/int64(Thearch.Ptrsize))
			proggendata(g, obj.BitsScalar)
			proggenarrayend(g)
		}

		g.pos = s.Value + s.Size
	} else if decodetype_usegcprog(s.Gotype) != 0 {
		// gc program, copy directly
		proggendataflush(g)

		gcprog := decodetype_gcprog(s.Gotype)
		size := decodetype_size(s.Gotype)
		if (size%int64(Thearch.Ptrsize) != 0) || (g.pos%int64(Thearch.Ptrsize) != 0) {
			Diag("proggenaddsym: unaligned gcprog symbol %s: size=%d pos=%d", s.Name, s.Size, g.pos)
		}
		for i := int64(0); i < int64(len(gcprog.P)-1); i++ {
			proggenemit(g, uint8(gcprog.P[i]))
		}
		g.pos = s.Value + size
	} else {
		// gc mask, it's small so emit as data
		mask := decodetype_gcmask(s.Gotype)

		size := decodetype_size(s.Gotype)
		if (size%int64(Thearch.Ptrsize) != 0) || (g.pos%int64(Thearch.Ptrsize) != 0) {
			Diag("proggenaddsym: unaligned gcmask symbol %s: size=%d pos=%d", s.Name, s.Size, g.pos)
		}
		for i := int64(0); i < size; i += int64(Thearch.Ptrsize) {
			proggendata(g, uint8((mask[i/int64(Thearch.Ptrsize)/2]>>uint64((i/int64(Thearch.Ptrsize)%2)*4+2))&obj.BitsMask))
		}
		g.pos = s.Value + size
	}
}

func growdatsize(datsizep *int64, s *LSym) {
	datsize := *datsizep
	if s.Size < 0 {
		Diag("negative size (datsize = %d, s->size = %d)", datsize, s.Size)
	}
	if datsize+s.Size < datsize {
		Diag("symbol too large (datsize = %d, s->size = %d)", datsize, s.Size)
	}
	*datsizep = datsize + s.Size
}

func dodata() {
	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f dodata\n", obj.Cputime())
	}
	Bflush(&Bso)

	var last *LSym
	datap = nil

	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if !s.Reachable || s.Special != 0 {
			continue
		}
		if STEXT < s.Type && s.Type < SXREF {
			if s.Onlist != 0 {
				log.Fatalf("symbol %s listed multiple times", s.Name)
			}
			s.Onlist = 1
			if last == nil {
				datap = s
			} else {
				last.Next = s
			}
			s.Next = nil
			last = s
		}
	}

	for s := datap; s != nil; s = s.Next {
		if int64(len(s.P)) > s.Size {
			Diag("%s: initialize bounds (%d < %d)", s.Name, int64(s.Size), len(s.P))
		}
	}

	/*
	 * now that we have the datap list, but before we start
	 * to assign addresses, record all the necessary
	 * dynamic relocations.  these will grow the relocation
	 * symbol, which is itself data.
	 *
	 * on darwin, we need the symbol table numbers for dynreloc.
	 */
	if HEADTYPE == Hdarwin {
		machosymorder()
	}
	dynreloc()

	/* some symbols may no longer belong in datap (Mach-O) */
	var l **LSym
	var s *LSym
	for l = &datap; ; {
		s = *l
		if s == nil {
			break
		}

		if s.Type <= STEXT || SXREF <= s.Type {
			*l = s.Next
		} else {
			l = &s.Next
		}
	}

	*l = nil

	datap = listsort(datap, datcmp, listnextp)

	/*
	 * allocate sections.  list is sorted by type,
	 * so we can just walk it for each piece we want to emit.
	 * segdata is processed before segtext, because we need
	 * to see all symbols in the .data and .bss sections in order
	 * to generate garbage collection information.
	 */

	/* begin segdata */

	/* skip symbols belonging to segtext */
	s = datap

	for ; s != nil && s.Type < SELFSECT; s = s.Next {
	}

	/* writable ELF sections */
	datsize := int64(0)

	var sect *Section
	for ; s != nil && s.Type < SELFGOT; s = s.Next {
		sect = addsection(&Segdata, s.Name, 06)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = SDATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* .got (and .toc on ppc64) */
	if s.Type == SELFGOT {
		sect := addsection(&Segdata, ".got", 06)
		sect.Align = maxalign(s, SELFGOT)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		var toc *LSym
		for ; s != nil && s.Type == SELFGOT; s = s.Next {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Type = SDATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)

			// Resolve .TOC. symbol for this object file (ppc64)
			toc = Linkrlookup(Ctxt, ".TOC.", int(s.Version))

			if toc != nil {
				toc.Sect = sect
				toc.Outer = s
				toc.Sub = s.Sub
				s.Sub = toc

				toc.Value = 0x8000
			}

			growdatsize(&datsize, s)
		}

		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* pointer-free data */
	sect = addsection(&Segdata, ".noptrdata", 06)

	sect.Align = maxalign(s, SINITARR-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.noptrdata", 0).Sect = sect
	Linklookup(Ctxt, "runtime.enoptrdata", 0).Sect = sect
	for ; s != nil && s.Type < SINITARR; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SDATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	hasinitarr := Linkshared

	/* shared library initializer */
	switch Buildmode {
	case BuildmodeCArchive, BuildmodeCShared, BuildmodeShared:
		hasinitarr = true
	}

	if hasinitarr {
		sect := addsection(&Segdata, ".init_array", 06)
		sect.Align = maxalign(s, SINITARR)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for ; s != nil && s.Type == SINITARR; s = s.Next {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			growdatsize(&datsize, s)
		}

		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* data */
	sect = addsection(&Segdata, ".data", 06)

	sect.Align = maxalign(s, SBSS-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.data", 0).Sect = sect
	Linklookup(Ctxt, "runtime.edata", 0).Sect = sect
	gcdata := Linklookup(Ctxt, "runtime.gcdata", 0)
	var gen ProgGen
	proggeninit(&gen, gcdata)
	for ; s != nil && s.Type < SBSS; s = s.Next {
		if s.Type == SINITARR {
			Ctxt.Cursym = s
			Diag("unexpected symbol type %d", s.Type)
		}

		s.Sect = sect
		s.Type = SDATA
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		proggenaddsym(&gen, s) // gc
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr
	proggenfini(&gen, int64(sect.Length)) // gc

	/* bss */
	sect = addsection(&Segdata, ".bss", 06)

	sect.Align = maxalign(s, SNOPTRBSS-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.bss", 0).Sect = sect
	Linklookup(Ctxt, "runtime.ebss", 0).Sect = sect
	gcbss := Linklookup(Ctxt, "runtime.gcbss", 0)
	proggeninit(&gen, gcbss)
	for ; s != nil && s.Type < SNOPTRBSS; s = s.Next {
		s.Sect = sect
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		proggenaddsym(&gen, s) // gc
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr
	proggenfini(&gen, int64(sect.Length)) // gc

	/* pointer-free bss */
	sect = addsection(&Segdata, ".noptrbss", 06)

	sect.Align = maxalign(s, SNOPTRBSS)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.noptrbss", 0).Sect = sect
	Linklookup(Ctxt, "runtime.enoptrbss", 0).Sect = sect
	for ; s != nil && s.Type == SNOPTRBSS; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr
	Linklookup(Ctxt, "runtime.end", 0).Sect = sect

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if datsize != int64(uint32(datsize)) {
		Diag("data or bss segment too large")
	}

	if Iself && Linkmode == LinkExternal && s != nil && s.Type == STLSBSS && HEADTYPE != Hopenbsd {
		sect := addsection(&Segdata, ".tbss", 06)
		sect.Align = int32(Thearch.Ptrsize)
		sect.Vaddr = 0
		datsize = 0
		for ; s != nil && s.Type == STLSBSS; s = s.Next {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			growdatsize(&datsize, s)
		}

		sect.Length = uint64(datsize)
	} else {
		// Might be internal linking but still using cgo.
		// In that case, the only possible STLSBSS symbol is runtime.tlsg.
		// Give it offset 0, because it's the only thing here.
		if s != nil && s.Type == STLSBSS && s.Name == "runtime.tlsg" {
			s.Value = 0
			s = s.Next
		}
	}

	if s != nil {
		Ctxt.Cursym = nil
		Diag("unexpected symbol type %d for %s", s.Type, s.Name)
	}

	/*
	 * We finished data, begin read-only data.
	 * Not all systems support a separate read-only non-executable data section.
	 * ELF systems do.
	 * OS X and Plan 9 do not.
	 * Windows PE may, but if so we have not implemented it.
	 * And if we're using external linking mode, the point is moot,
	 * since it's not our decision; that code expects the sections in
	 * segtext.
	 */
	var segro *Segment
	if Iself && Linkmode == LinkInternal {
		segro = &Segrodata
	} else {
		segro = &Segtext
	}

	s = datap

	datsize = 0

	/* read-only executable ELF, Mach-O sections */
	for ; s != nil && s.Type < STYPE; s = s.Next {
		sect = addsection(&Segtext, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* read-only data */
	sect = addsection(segro, ".rodata", 04)

	sect.Align = maxalign(s, STYPELINK-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = 0
	Linklookup(Ctxt, "runtime.rodata", 0).Sect = sect
	Linklookup(Ctxt, "runtime.erodata", 0).Sect = sect
	for ; s != nil && s.Type < STYPELINK; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	/* typelink */
	sect = addsection(segro, ".typelink", 04)

	sect.Align = maxalign(s, STYPELINK)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.typelink", 0).Sect = sect
	Linklookup(Ctxt, "runtime.etypelink", 0).Sect = sect
	for ; s != nil && s.Type == STYPELINK; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	/* gosymtab */
	sect = addsection(segro, ".gosymtab", 04)

	sect.Align = maxalign(s, SPCLNTAB-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.symtab", 0).Sect = sect
	Linklookup(Ctxt, "runtime.esymtab", 0).Sect = sect
	for ; s != nil && s.Type < SPCLNTAB; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	/* gopclntab */
	sect = addsection(segro, ".gopclntab", 04)

	sect.Align = maxalign(s, SELFROSECT-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.pclntab", 0).Sect = sect
	Linklookup(Ctxt, "runtime.epclntab", 0).Sect = sect
	for ; s != nil && s.Type < SELFROSECT; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	/* read-only ELF, Mach-O sections */
	for ; s != nil && s.Type < SELFSECT; s = s.Next {
		sect = addsection(segro, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
		sect.Length = uint64(datsize) - sect.Vaddr
	}

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if datsize != int64(uint32(datsize)) {
		Diag("read-only data segment too large")
	}

	/* number the sections */
	n := int32(1)

	for sect := Segtext.Sect; sect != nil; sect = sect.Next {
		sect.Extnum = int16(n)
		n++
	}
	for sect := Segrodata.Sect; sect != nil; sect = sect.Next {
		sect.Extnum = int16(n)
		n++
	}
	for sect := Segdata.Sect; sect != nil; sect = sect.Next {
		sect.Extnum = int16(n)
		n++
	}
}

// assign addresses to text
func textaddress() {
	var sub *LSym

	addsection(&Segtext, ".text", 05)

	// Assign PCs in text segment.
	// Could parallelize, by assigning to text
	// and then letting threads copy down, but probably not worth it.
	sect := Segtext.Sect

	sect.Align = int32(Funcalign)
	Linklookup(Ctxt, "runtime.text", 0).Sect = sect
	Linklookup(Ctxt, "runtime.etext", 0).Sect = sect
	va := uint64(INITTEXT)
	sect.Vaddr = va
	for sym := Ctxt.Textp; sym != nil; sym = sym.Next {
		sym.Sect = sect
		if sym.Type&SSUB != 0 {
			continue
		}
		if sym.Align != 0 {
			va = uint64(Rnd(int64(va), int64(sym.Align)))
		} else {
			va = uint64(Rnd(int64(va), int64(Funcalign)))
		}
		sym.Value = 0
		for sub = sym; sub != nil; sub = sub.Sub {
			sub.Value += int64(va)
		}
		if sym.Size == 0 && sym.Sub != nil {
			Ctxt.Cursym = sym
		}
		if sym.Size < MINFUNC {
			va += MINFUNC // spacing required for findfunctab
		} else {
			va += uint64(sym.Size)
		}
	}

	sect.Length = va - sect.Vaddr
}

// assign addresses
func address() {
	va := uint64(INITTEXT)
	Segtext.Rwx = 05
	Segtext.Vaddr = va
	Segtext.Fileoff = uint64(HEADR)
	for s := Segtext.Sect; s != nil; s = s.Next {
		va = uint64(Rnd(int64(va), int64(s.Align)))
		s.Vaddr = va
		va += s.Length
	}

	Segtext.Length = va - uint64(INITTEXT)
	Segtext.Filelen = Segtext.Length
	if HEADTYPE == Hnacl {
		va += 32 // room for the "halt sled"
	}

	if Segrodata.Sect != nil {
		// align to page boundary so as not to mix
		// rodata and executable text.
		va = uint64(Rnd(int64(va), int64(INITRND)))

		Segrodata.Rwx = 04
		Segrodata.Vaddr = va
		Segrodata.Fileoff = va - Segtext.Vaddr + Segtext.Fileoff
		Segrodata.Filelen = 0
		for s := Segrodata.Sect; s != nil; s = s.Next {
			va = uint64(Rnd(int64(va), int64(s.Align)))
			s.Vaddr = va
			va += s.Length
		}

		Segrodata.Length = va - Segrodata.Vaddr
		Segrodata.Filelen = Segrodata.Length
	}

	va = uint64(Rnd(int64(va), int64(INITRND)))
	Segdata.Rwx = 06
	Segdata.Vaddr = va
	Segdata.Fileoff = va - Segtext.Vaddr + Segtext.Fileoff
	Segdata.Filelen = 0
	if HEADTYPE == Hwindows {
		Segdata.Fileoff = Segtext.Fileoff + uint64(Rnd(int64(Segtext.Length), PEFILEALIGN))
	}
	if HEADTYPE == Hplan9 {
		Segdata.Fileoff = Segtext.Fileoff + Segtext.Filelen
	}
	var data *Section
	var noptr *Section
	var bss *Section
	var noptrbss *Section
	var vlen int64
	for s := Segdata.Sect; s != nil; s = s.Next {
		vlen = int64(s.Length)
		if s.Next != nil {
			vlen = int64(s.Next.Vaddr - s.Vaddr)
		}
		s.Vaddr = va
		va += uint64(vlen)
		Segdata.Length = va - Segdata.Vaddr
		if s.Name == ".data" {
			data = s
		}
		if s.Name == ".noptrdata" {
			noptr = s
		}
		if s.Name == ".bss" {
			bss = s
		}
		if s.Name == ".noptrbss" {
			noptrbss = s
		}
	}

	Segdata.Filelen = bss.Vaddr - Segdata.Vaddr

	text := Segtext.Sect
	var rodata *Section
	if Segrodata.Sect != nil {
		rodata = Segrodata.Sect
	} else {
		rodata = text.Next
	}
	typelink := rodata.Next
	symtab := typelink.Next
	pclntab := symtab.Next

	var sub *LSym
	for sym := datap; sym != nil; sym = sym.Next {
		Ctxt.Cursym = sym
		if sym.Sect != nil {
			sym.Value += int64((sym.Sect.(*Section)).Vaddr)
		}
		for sub = sym.Sub; sub != nil; sub = sub.Sub {
			sub.Value += sym.Value
		}
	}

	xdefine("runtime.text", STEXT, int64(text.Vaddr))
	xdefine("runtime.etext", STEXT, int64(text.Vaddr+text.Length))
	xdefine("runtime.rodata", SRODATA, int64(rodata.Vaddr))
	xdefine("runtime.erodata", SRODATA, int64(rodata.Vaddr+rodata.Length))
	xdefine("runtime.typelink", SRODATA, int64(typelink.Vaddr))
	xdefine("runtime.etypelink", SRODATA, int64(typelink.Vaddr+typelink.Length))

	sym := Linklookup(Ctxt, "runtime.gcdata", 0)
	sym.Local = true
	xdefine("runtime.egcdata", SRODATA, Symaddr(sym)+sym.Size)
	Linklookup(Ctxt, "runtime.egcdata", 0).Sect = sym.Sect

	sym = Linklookup(Ctxt, "runtime.gcbss", 0)
	sym.Local = true
	xdefine("runtime.egcbss", SRODATA, Symaddr(sym)+sym.Size)
	Linklookup(Ctxt, "runtime.egcbss", 0).Sect = sym.Sect

	xdefine("runtime.symtab", SRODATA, int64(symtab.Vaddr))
	xdefine("runtime.esymtab", SRODATA, int64(symtab.Vaddr+symtab.Length))
	xdefine("runtime.pclntab", SRODATA, int64(pclntab.Vaddr))
	xdefine("runtime.epclntab", SRODATA, int64(pclntab.Vaddr+pclntab.Length))
	xdefine("runtime.noptrdata", SNOPTRDATA, int64(noptr.Vaddr))
	xdefine("runtime.enoptrdata", SNOPTRDATA, int64(noptr.Vaddr+noptr.Length))
	xdefine("runtime.bss", SBSS, int64(bss.Vaddr))
	xdefine("runtime.ebss", SBSS, int64(bss.Vaddr+bss.Length))
	xdefine("runtime.data", SDATA, int64(data.Vaddr))
	xdefine("runtime.edata", SDATA, int64(data.Vaddr+data.Length))
	xdefine("runtime.noptrbss", SNOPTRBSS, int64(noptrbss.Vaddr))
	xdefine("runtime.enoptrbss", SNOPTRBSS, int64(noptrbss.Vaddr+noptrbss.Length))
	xdefine("runtime.end", SBSS, int64(Segdata.Vaddr+Segdata.Length))
}
