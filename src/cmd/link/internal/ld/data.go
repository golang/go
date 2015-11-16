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
	"cmd/internal/gcprog"
	"cmd/internal/obj"
	"fmt"
	"log"
	"os"
	"strconv"
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
		s.Type = obj.SDATA
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
		s.Type = obj.SDATA
	}
	s.Reachable = true
	i := s.Size
	s.Size += int64(ctxt.Arch.Ptrsize)
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.Ptrsize)
	r.Type = obj.R_ADDR
	r.Add = add
	return i + int64(r.Siz)
}

func Addpcrelplus(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = obj.SDATA
	}
	s.Reachable = true
	i := s.Size
	s.Size += 4
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Add = add
	r.Type = obj.R_PCREL
	r.Siz = 4
	return i + int64(r.Siz)
}

func Addaddr(ctxt *Link, s *LSym, t *LSym) int64 {
	return Addaddrplus(ctxt, s, t, 0)
}

func setaddrplus(ctxt *Link, s *LSym, off int64, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = obj.SDATA
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
	r.Type = obj.R_ADDR
	r.Add = add
	return off + int64(r.Siz)
}

func setaddr(ctxt *Link, s *LSym, off int64, t *LSym) int64 {
	return setaddrplus(ctxt, s, off, t, 0)
}

func addsize(ctxt *Link, s *LSym, t *LSym) int64 {
	if s.Type == 0 {
		s.Type = obj.SDATA
	}
	s.Reachable = true
	i := s.Size
	s.Size += int64(ctxt.Arch.Ptrsize)
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.Ptrsize)
	r.Type = obj.R_SIZE
	return i + int64(r.Siz)
}

func addaddrplus4(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = obj.SDATA
	}
	s.Reachable = true
	i := s.Size
	s.Size += 4
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = 4
	r.Type = obj.R_ADDR
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
	if s1.Type != obj.SELFGOT && s1.Size != s2.Size {
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

		if r.Sym != nil && (r.Sym.Type&(obj.SMASK|obj.SHIDDEN) == 0 || r.Sym.Type&obj.SMASK == obj.SXREF) {
			// When putting the runtime but not main into a shared library
			// these symbols are undefined and that's OK.
			if Buildmode == BuildmodeShared && (r.Sym.Name == "main.main" || r.Sym.Name == "main.init") {
				r.Sym.Type = obj.SDYNIMPORT
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
		if HEADTYPE != obj.Hsolaris && r.Sym != nil && r.Sym.Type == obj.SDYNIMPORT && !DynlinkingGo() {
			if !(Thearch.Thechar == '9' && Linkmode == LinkExternal && r.Sym.Name == ".TOC.") {
				Diag("unhandled relocation for %s (type %d rtype %d)", r.Sym.Name, r.Sym.Type, r.Type)
			}
		}
		if r.Sym != nil && r.Sym.Type != obj.STLSBSS && !r.Sym.Reachable {
			Diag("unreachable sym in relocation: %s %s", s.Name, r.Sym.Name)
		}

		switch r.Type {
		default:
			switch siz {
			default:
				Diag("bad reloc size %#x for %s", uint32(siz), r.Sym.Name)
			case 1:
				o = int64(s.P[off])
			case 2:
				o = int64(Ctxt.Arch.ByteOrder.Uint16(s.P[off:]))
			case 4:
				o = int64(Ctxt.Arch.ByteOrder.Uint32(s.P[off:]))
			case 8:
				o = int64(Ctxt.Arch.ByteOrder.Uint64(s.P[off:]))
			}
			if Thearch.Archreloc(r, s, &o) < 0 {
				Diag("unknown reloc %d", r.Type)
			}

		case obj.R_TLS_LE:
			isAndroidX86 := goos == "android" && (Thearch.Thechar == '6' || Thearch.Thechar == '8')

			if Linkmode == LinkExternal && Iself && HEADTYPE != obj.Hopenbsd && !isAndroidX86 {
				r.Done = 0
				if r.Sym == nil {
					r.Sym = Ctxt.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if Thearch.Thechar != '6' {
					o = r.Add
				}
				break
			}

			if Iself && Thearch.Thechar == '5' {
				// On ELF ARM, the thread pointer is 8 bytes before
				// the start of the thread-local data block, so add 8
				// to the actual TLS offset (r->sym->value).
				// This 8 seems to be a fundamental constant of
				// ELF on ARM (or maybe Glibc on ARM); it is not
				// related to the fact that our own TLS storage happens
				// to take up 8 bytes.
				o = 8 + r.Sym.Value
			} else if Iself || Ctxt.Headtype == obj.Hplan9 || Ctxt.Headtype == obj.Hdarwin || isAndroidX86 {
				o = int64(Ctxt.Tlsoffset) + r.Add
			} else if Ctxt.Headtype == obj.Hwindows {
				o = r.Add
			} else {
				log.Fatalf("unexpected R_TLS_LE relocation for %s", Headstr(Ctxt.Headtype))
			}

		case obj.R_TLS_IE:
			isAndroidX86 := goos == "android" && (Thearch.Thechar == '6' || Thearch.Thechar == '8')

			if Linkmode == LinkExternal && Iself && HEADTYPE != obj.Hopenbsd && !isAndroidX86 {
				r.Done = 0
				if r.Sym == nil {
					r.Sym = Ctxt.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if Thearch.Thechar != '6' {
					o = r.Add
				}
				break
			}
			log.Fatalf("cannot handle R_TLS_IE when linking internally")

		case obj.R_ADDR:
			if Linkmode == LinkExternal && r.Sym.Type != obj.SCONST {
				r.Done = 0

				// set up addend for eventual relocation via outer symbol.
				rs = r.Sym

				r.Xadd = r.Add
				for rs.Outer != nil {
					r.Xadd += Symaddr(rs) - Symaddr(rs.Outer)
					rs = rs.Outer
				}

				if rs.Type != obj.SHOSTOBJ && rs.Type != obj.SDYNIMPORT && rs.Sect == nil {
					Diag("missing section for %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if Iself {
					if Thearch.Thechar == '6' {
						o = 0
					}
				} else if HEADTYPE == obj.Hdarwin {
					// ld64 for arm64 has a bug where if the address pointed to by o exists in the
					// symbol table (dynid >= 0), or is inside a symbol that exists in the symbol
					// table, then it will add o twice into the relocated value.
					// The workaround is that on arm64 don't ever add symaddr to o and always use
					// extern relocation by requiring rs->dynid >= 0.
					if rs.Type != obj.SHOSTOBJ {
						if Thearch.Thechar == '7' && rs.Dynid < 0 {
							Diag("R_ADDR reloc to %s+%d is not supported on darwin/arm64", rs.Name, o)
						}
						if Thearch.Thechar != '7' {
							o += Symaddr(rs)
						}
					}
				} else if HEADTYPE == obj.Hwindows {
					// nothing to do
				} else {
					Diag("unhandled pcrel relocation for %s", headstring)
				}

				break
			}

			o = Symaddr(r.Sym) + r.Add

			// On amd64, 4-byte offsets will be sign-extended, so it is impossible to
			// access more than 2GB of static data; fail at link time is better than
			// fail at runtime. See https://golang.org/issue/7980.
			// Instead of special casing only amd64, we treat this as an error on all
			// 64-bit architectures so as to be future-proof.
			if int32(o) < 0 && Thearch.Ptrsize > 4 && siz == 4 {
				Diag("non-pc-relative relocation address is too big: %#x (%#x + %#x)", uint64(o), Symaddr(r.Sym), r.Add)
				errorexit()
			}

			// r->sym can be null when CALL $(constant) is transformed from absolute PC to relative PC call.
		case obj.R_CALL, obj.R_GOTPCREL, obj.R_PCREL:
			if Linkmode == LinkExternal && r.Sym != nil && r.Sym.Type != obj.SCONST && (r.Sym.Sect != Ctxt.Cursym.Sect || r.Type == obj.R_GOTPCREL) {
				r.Done = 0

				// set up addend for eventual relocation via outer symbol.
				rs = r.Sym

				r.Xadd = r.Add
				for rs.Outer != nil {
					r.Xadd += Symaddr(rs) - Symaddr(rs.Outer)
					rs = rs.Outer
				}

				r.Xadd -= int64(r.Siz) // relative to address after the relocated chunk
				if rs.Type != obj.SHOSTOBJ && rs.Type != obj.SDYNIMPORT && rs.Sect == nil {
					Diag("missing section for %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if Iself {
					if Thearch.Thechar == '6' {
						o = 0
					}
				} else if HEADTYPE == obj.Hdarwin {
					if r.Type == obj.R_CALL {
						if rs.Type != obj.SHOSTOBJ {
							o += int64(uint64(Symaddr(rs)) - rs.Sect.Vaddr)
						}
						o -= int64(r.Off) // relative to section offset, not symbol
					} else {
						o += int64(r.Siz)
					}
				} else if HEADTYPE == obj.Hwindows && Thearch.Thechar == '6' { // only amd64 needs PCREL
					// PE/COFF's PC32 relocation uses the address after the relocated
					// bytes as the base. Compensate by skewing the addend.
					o += int64(r.Siz)
					// GNU ld always add VirtualAddress of the .text section to the
					// relocated address, compensate that.
					o -= int64(s.Sect.Vaddr - PEBASE)
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

		case obj.R_SIZE:
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
			if r.Type == obj.R_PCREL || r.Type == obj.R_CALL {
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
	Bso.Flush()

	for s := Ctxt.Textp; s != nil; s = s.Next {
		relocsym(s)
	}
	for s := datap; s != nil; s = s.Next {
		relocsym(s)
	}
}

func dynrelocsym(s *LSym) {
	if HEADTYPE == obj.Hwindows && Linkmode != LinkExternal {
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
		if r.Sym != nil && r.Sym.Type == obj.SDYNIMPORT || r.Type >= 256 {
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
	if Debug['d'] != 0 && HEADTYPE != obj.Hwindows {
		return
	}
	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f reloc\n", obj.Cputime())
	}
	Bso.Flush()

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
		if sym.Type&obj.SSUB == 0 && sym.Value >= addr {
			break
		}
	}

	eaddr := addr + size
	var ep []byte
	var p []byte
	for ; sym != nil; sym = sym.Next {
		if sym.Type&obj.SSUB != 0 {
			continue
		}
		if sym.Value >= eaddr {
			break
		}
		Ctxt.Cursym = sym
		if sym.Value < addr {
			Diag("phase error: addr=%#x but sym=%#x type=%d", int64(addr), int64(sym.Value), sym.Type)
			errorexit()
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
			errorexit()
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
		q = sym.P

		for len(q) >= 16 {
			fmt.Fprintf(&Bso, "%.6x\t% x\n", uint64(addr), q[:16])
			addr += 16
			q = q[16:]
		}

		if len(q) > 0 {
			fmt.Fprintf(&Bso, "%.6x\t% x\n", uint64(addr), q)
			addr += int64(len(q))
		}
	}

	if addr < eaddr {
		fmt.Fprintf(&Bso, "%-20s %.8x|", "_", uint64(int64(addr)))
		for ; addr < eaddr; addr++ {
			fmt.Fprintf(&Bso, " %.2x", 0)
		}
	}

	Bso.Flush()
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
				case obj.R_ADDR:
					typ = "addr"

				case obj.R_PCREL:
					typ = "pcrel"

				case obj.R_CALL:
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

var strdata []*LSym

func addstrdata1(arg string) {
	i := strings.Index(arg, "=")
	if i < 0 {
		Exitf("-X flag requires argument of the form importpath.name=value")
	}
	addstrdata(arg[:i], arg[i+1:])
}

func addstrdata(name string, value string) {
	p := fmt.Sprintf("%s.str", name)
	sp := Linklookup(Ctxt, p, 0)

	Addstring(sp, value)
	sp.Type = obj.SRODATA

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

	strdata = append(strdata, s)

	sp.Reachable = reachable
}

func checkstrdata() {
	for _, s := range strdata {
		if s.Type == obj.STEXT {
			Diag("cannot use -X with text symbol %s", s.Name)
		} else if s.Gotype != nil && s.Gotype.Name != "type.string" {
			Diag("cannot use -X with non-string symbol %s", s.Name)
		}
	}
}

func Addstring(s *LSym, str string) int64 {
	if s.Type == 0 {
		s.Type = obj.SNOPTRDATA
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

// addgostring adds str, as a Go string value, to s. symname is the name of the
// symbol used to define the string data and must be unique per linked object.
func addgostring(s *LSym, symname, str string) {
	sym := Linklookup(Ctxt, symname, 0)
	if sym.Type != obj.Sxxx {
		Diag("duplicate symname in addgostring: %s", symname)
	}
	sym.Reachable = true
	sym.Local = true
	sym.Type = obj.SRODATA
	sym.Size = int64(len(str))
	sym.P = []byte(str)
	Addaddr(Ctxt, s, sym)
	adduint(Ctxt, s, uint64(len(str)))
}

func addinitarrdata(s *LSym) {
	p := s.Name + ".ptr"
	sp := Linklookup(Ctxt, p, 0)
	sp.Type = obj.SINITARR
	sp.Size = 0
	sp.Dupok = 1
	Addaddr(Ctxt, sp, s)
}

func dosymtype() {
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if len(s.P) > 0 {
			if s.Type == obj.SBSS {
				s.Type = obj.SDATA
			}
			if s.Type == obj.SNOPTRBSS {
				s.Type = obj.SNOPTRDATA
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

const debugGCProg = false

type GCProg struct {
	sym *LSym
	w   gcprog.Writer
}

func (p *GCProg) Init(name string) {
	p.sym = Linklookup(Ctxt, name, 0)
	p.w.Init(p.writeByte)
	if debugGCProg {
		fmt.Fprintf(os.Stderr, "ld: start GCProg %s\n", name)
		p.w.Debug(os.Stderr)
	}
}

func (p *GCProg) writeByte(x byte) {
	Adduint8(Ctxt, p.sym, x)
}

func (p *GCProg) End(size int64) {
	p.w.ZeroUntil(size / int64(Thearch.Ptrsize))
	p.w.End()
	if debugGCProg {
		fmt.Fprintf(os.Stderr, "ld: end GCProg\n")
	}
}

func (p *GCProg) AddSym(s *LSym) {
	typ := s.Gotype
	// Things without pointers should be in SNOPTRDATA or SNOPTRBSS;
	// everything we see should have pointers and should therefore have a type.
	if typ == nil {
		Diag("missing Go type information for global symbol: %s size %d", s.Name, int(s.Size))
		return
	}

	ptrsize := int64(Thearch.Ptrsize)
	nptr := decodetype_ptrdata(typ) / ptrsize

	if debugGCProg {
		fmt.Fprintf(os.Stderr, "gcprog sym: %s at %d (ptr=%d+%d)\n", s.Name, s.Value, s.Value/ptrsize, nptr)
	}

	if decodetype_usegcprog(typ) == 0 {
		// Copy pointers from mask into program.
		mask := decodetype_gcmask(typ)
		for i := int64(0); i < nptr; i++ {
			if (mask[i/8]>>uint(i%8))&1 != 0 {
				p.w.Ptr(s.Value/ptrsize + i)
			}
		}
		return
	}

	// Copy program.
	prog := decodetype_gcprog(typ)
	p.w.ZeroUntil(s.Value / ptrsize)
	p.w.Append(prog[4:], nptr)
}

func growdatsize(datsizep *int64, s *LSym) {
	datsize := *datsizep
	const cutoff int64 = 2e9 // 2 GB (or so; looks better in errors than 2^31)
	switch {
	case s.Size < 0:
		Diag("%s: negative size (%d bytes)", s.Name, s.Size)
	case s.Size > cutoff:
		Diag("%s: symbol too large (%d bytes)", s.Name, s.Size)
	case datsize <= cutoff && datsize+s.Size > cutoff:
		Diag("%s: too much data (over %d bytes)", s.Name, cutoff)
	}
	*datsizep = datsize + s.Size
}

func dodata() {
	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f dodata\n", obj.Cputime())
	}
	Bso.Flush()

	var last *LSym
	datap = nil

	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if !s.Reachable || s.Special != 0 {
			continue
		}
		if obj.STEXT < s.Type && s.Type < obj.SXREF {
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
	if HEADTYPE == obj.Hdarwin {
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

		if s.Type <= obj.STEXT || obj.SXREF <= s.Type {
			*l = s.Next
		} else {
			l = &s.Next
		}
	}

	*l = nil

	if UseRelro() {
		// "read only" data with relocations needs to go in its own section
		// when building a shared library. We do this by boosting objects of
		// type SXXX with relocations to type SXXXRELRO.
		for s := datap; s != nil; s = s.Next {
			if (s.Type >= obj.STYPE && s.Type <= obj.SFUNCTAB && len(s.R) > 0) || s.Type == obj.SGOSTRING {
				s.Type += (obj.STYPERELRO - obj.STYPE)
				if s.Outer != nil {
					s.Outer.Type = s.Type
				}
			}
		}
		// Check that we haven't made two symbols with the same .Outer into
		// different types (because references two symbols with non-nil Outer
		// become references to the outer symbol + offset it's vital that the
		// symbol and the outer end up in the same section).
		for s := datap; s != nil; s = s.Next {
			if s.Outer != nil && s.Outer.Type != s.Type {
				Diag("inconsistent types for %s and its Outer %s (%d != %d)",
					s.Name, s.Outer.Name, s.Type, s.Outer.Type)
			}
		}

	}

	datap = listsort(datap, datcmp, listnextp)

	if Iself {
		// Make .rela and .rela.plt contiguous, the ELF ABI requires this
		// and Solaris actually cares.
		var relplt *LSym
		for l = &datap; *l != nil; l = &(*l).Next {
			if (*l).Name == ".rel.plt" || (*l).Name == ".rela.plt" {
				relplt = (*l)
				*l = (*l).Next
				break
			}
		}
		if relplt != nil {
			for s = datap; s != nil; s = s.Next {
				if s.Name == ".rel" || s.Name == ".rela" {
					relplt.Next = s.Next
					s.Next = relplt
				}
			}
		}
	}

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

	for ; s != nil && s.Type < obj.SELFSECT; s = s.Next {
	}

	/* writable ELF sections */
	datsize := int64(0)

	var sect *Section
	for ; s != nil && s.Type < obj.SELFGOT; s = s.Next {
		sect = addsection(&Segdata, s.Name, 06)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = obj.SDATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* .got (and .toc on ppc64) */
	if s.Type == obj.SELFGOT {
		sect := addsection(&Segdata, ".got", 06)
		sect.Align = maxalign(s, obj.SELFGOT)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		var toc *LSym
		for ; s != nil && s.Type == obj.SELFGOT; s = s.Next {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Type = obj.SDATA
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

	sect.Align = maxalign(s, obj.SINITARR-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.noptrdata", 0).Sect = sect
	Linklookup(Ctxt, "runtime.enoptrdata", 0).Sect = sect
	for ; s != nil && s.Type < obj.SINITARR; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SDATA
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
		sect.Align = maxalign(s, obj.SINITARR)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for ; s != nil && s.Type == obj.SINITARR; s = s.Next {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			growdatsize(&datsize, s)
		}

		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* data */
	sect = addsection(&Segdata, ".data", 06)
	sect.Align = maxalign(s, obj.SBSS-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.data", 0).Sect = sect
	Linklookup(Ctxt, "runtime.edata", 0).Sect = sect
	var gc GCProg
	gc.Init("runtime.gcdata")
	for ; s != nil && s.Type < obj.SBSS; s = s.Next {
		if s.Type == obj.SINITARR {
			Ctxt.Cursym = s
			Diag("unexpected symbol type %d", s.Type)
		}

		s.Sect = sect
		s.Type = obj.SDATA
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		gc.AddSym(s)
		growdatsize(&datsize, s)
	}
	sect.Length = uint64(datsize) - sect.Vaddr
	gc.End(int64(sect.Length))

	/* bss */
	sect = addsection(&Segdata, ".bss", 06)
	sect.Align = maxalign(s, obj.SNOPTRBSS-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.bss", 0).Sect = sect
	Linklookup(Ctxt, "runtime.ebss", 0).Sect = sect
	gc = GCProg{}
	gc.Init("runtime.gcbss")
	for ; s != nil && s.Type < obj.SNOPTRBSS; s = s.Next {
		s.Sect = sect
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		gc.AddSym(s)
		growdatsize(&datsize, s)
	}
	sect.Length = uint64(datsize) - sect.Vaddr
	gc.End(int64(sect.Length))

	/* pointer-free bss */
	sect = addsection(&Segdata, ".noptrbss", 06)

	sect.Align = maxalign(s, obj.SNOPTRBSS)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.noptrbss", 0).Sect = sect
	Linklookup(Ctxt, "runtime.enoptrbss", 0).Sect = sect
	for ; s != nil && s.Type == obj.SNOPTRBSS; s = s.Next {
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

	if s != nil && s.Type == obj.STLSBSS {
		if Iself && (Linkmode == LinkExternal || Debug['d'] == 0) && HEADTYPE != obj.Hopenbsd {
			sect = addsection(&Segdata, ".tbss", 06)
			sect.Align = int32(Thearch.Ptrsize)
			sect.Vaddr = 0
		} else {
			sect = nil
		}
		datsize = 0

		for ; s != nil && s.Type == obj.STLSBSS; s = s.Next {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = datsize
			growdatsize(&datsize, s)
		}

		if sect != nil {
			sect.Length = uint64(datsize)
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
	for ; s != nil && s.Type < obj.STYPE; s = s.Next {
		sect = addsection(&Segtext, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* read-only data */
	sect = addsection(segro, ".rodata", 04)

	sect.Align = maxalign(s, obj.STYPERELRO-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = 0
	Linklookup(Ctxt, "runtime.rodata", 0).Sect = sect
	Linklookup(Ctxt, "runtime.erodata", 0).Sect = sect
	for ; s != nil && s.Type < obj.STYPERELRO; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	// There is some data that are conceptually read-only but are written to by
	// relocations. On GNU systems, we can arrange for the dynamic linker to
	// mprotect sections after relocations are applied by giving them write
	// permissions in the object file and calling them ".data.rel.ro.FOO". We
	// divide the .rodata section between actual .rodata and .data.rel.ro.rodata,
	// but for the other sections that this applies to, we just write a read-only
	// .FOO section or a read-write .data.rel.ro.FOO section depending on the
	// situation.
	// TODO(mwhudson): It would make sense to do this more widely, but it makes
	// the system linker segfault on darwin.
	relro_perms := 04
	relro_prefix := ""

	if UseRelro() {
		relro_perms = 06
		relro_prefix = ".data.rel.ro"
		/* data only written by relocations */
		sect = addsection(segro, ".data.rel.ro", 06)

		sect.Align = maxalign(s, obj.STYPELINK-1)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = 0
		for ; s != nil && s.Type < obj.STYPELINK; s = s.Next {
			datsize = aligndatsize(datsize, s)
			if s.Outer != nil && s.Outer.Sect != nil && s.Outer.Sect != sect {
				Diag("s.Outer (%s) in different section from s (%s)", s.Outer.Name, s.Name)
			}
			s.Sect = sect
			s.Type = obj.SRODATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			growdatsize(&datsize, s)
		}

		sect.Length = uint64(datsize) - sect.Vaddr

	}

	/* typelink */
	sect = addsection(segro, relro_prefix+".typelink", relro_perms)

	sect.Align = maxalign(s, obj.STYPELINK)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.typelink", 0).Sect = sect
	Linklookup(Ctxt, "runtime.etypelink", 0).Sect = sect
	for ; s != nil && s.Type == obj.STYPELINK; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	/* gosymtab */
	sect = addsection(segro, relro_prefix+".gosymtab", relro_perms)

	sect.Align = maxalign(s, obj.SPCLNTAB-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.symtab", 0).Sect = sect
	Linklookup(Ctxt, "runtime.esymtab", 0).Sect = sect
	for ; s != nil && s.Type < obj.SPCLNTAB; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	/* gopclntab */
	sect = addsection(segro, relro_prefix+".gopclntab", relro_perms)

	sect.Align = maxalign(s, obj.SELFROSECT-1)
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.pclntab", 0).Sect = sect
	Linklookup(Ctxt, "runtime.epclntab", 0).Sect = sect
	for ; s != nil && s.Type < obj.SELFROSECT; s = s.Next {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		growdatsize(&datsize, s)
	}

	sect.Length = uint64(datsize) - sect.Vaddr

	/* read-only ELF, Mach-O sections */
	for ; s != nil && s.Type < obj.SELFSECT; s = s.Next {
		sect = addsection(segro, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = obj.SRODATA
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

// Add buildid to beginning of text segment, on non-ELF systems.
// Non-ELF binary formats are not always flexible enough to
// give us a place to put the Go build ID. On those systems, we put it
// at the very beginning of the text segment.
// This ``header'' is read by cmd/go.
func textbuildid() {
	if Iself || buildid == "" {
		return
	}

	sym := Linklookup(Ctxt, "go.buildid", 0)
	sym.Reachable = true
	// The \xff is invalid UTF-8, meant to make it less likely
	// to find one of these accidentally.
	data := "\xff Go build ID: " + strconv.Quote(buildid) + "\n \xff"
	sym.Type = obj.STEXT
	sym.P = []byte(data)
	sym.Size = int64(len(sym.P))

	sym.Next = Ctxt.Textp
	Ctxt.Textp = sym
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
		if sym.Type&obj.SSUB != 0 {
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
	if HEADTYPE == obj.Hnacl {
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
	if HEADTYPE == obj.Hwindows {
		Segdata.Fileoff = Segtext.Fileoff + uint64(Rnd(int64(Segtext.Length), PEFILEALIGN))
	}
	if HEADTYPE == obj.Hplan9 {
		Segdata.Fileoff = Segtext.Fileoff + Segtext.Filelen
	}
	var data *Section
	var noptr *Section
	var bss *Section
	var noptrbss *Section
	var vlen int64
	for s := Segdata.Sect; s != nil; s = s.Next {
		if Iself && s.Name == ".tbss" {
			continue
		}
		vlen = int64(s.Length)
		if s.Next != nil && !(Iself && s.Next.Name == ".tbss") {
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
	if UseRelro() {
		// There is another section (.data.rel.ro) when building a shared
		// object on elf systems.
		typelink = typelink.Next
	}
	symtab := typelink.Next
	pclntab := symtab.Next

	var sub *LSym
	for sym := datap; sym != nil; sym = sym.Next {
		Ctxt.Cursym = sym
		if sym.Sect != nil {
			sym.Value += int64(sym.Sect.Vaddr)
		}
		for sub = sym.Sub; sub != nil; sub = sub.Sub {
			sub.Value += sym.Value
		}
	}

	if Buildmode == BuildmodeShared {
		s := Linklookup(Ctxt, "go.link.abihashbytes", 0)
		sectSym := Linklookup(Ctxt, ".note.go.abihash", 0)
		s.Sect = sectSym.Sect
		s.Value = int64(sectSym.Sect.Vaddr + 16)
	}

	xdefine("runtime.text", obj.STEXT, int64(text.Vaddr))
	xdefine("runtime.etext", obj.STEXT, int64(text.Vaddr+text.Length))
	xdefine("runtime.rodata", obj.SRODATA, int64(rodata.Vaddr))
	xdefine("runtime.erodata", obj.SRODATA, int64(rodata.Vaddr+rodata.Length))
	xdefine("runtime.typelink", obj.SRODATA, int64(typelink.Vaddr))
	xdefine("runtime.etypelink", obj.SRODATA, int64(typelink.Vaddr+typelink.Length))

	sym := Linklookup(Ctxt, "runtime.gcdata", 0)
	sym.Local = true
	xdefine("runtime.egcdata", obj.SRODATA, Symaddr(sym)+sym.Size)
	Linklookup(Ctxt, "runtime.egcdata", 0).Sect = sym.Sect

	sym = Linklookup(Ctxt, "runtime.gcbss", 0)
	sym.Local = true
	xdefine("runtime.egcbss", obj.SRODATA, Symaddr(sym)+sym.Size)
	Linklookup(Ctxt, "runtime.egcbss", 0).Sect = sym.Sect

	xdefine("runtime.symtab", obj.SRODATA, int64(symtab.Vaddr))
	xdefine("runtime.esymtab", obj.SRODATA, int64(symtab.Vaddr+symtab.Length))
	xdefine("runtime.pclntab", obj.SRODATA, int64(pclntab.Vaddr))
	xdefine("runtime.epclntab", obj.SRODATA, int64(pclntab.Vaddr+pclntab.Length))
	xdefine("runtime.noptrdata", obj.SNOPTRDATA, int64(noptr.Vaddr))
	xdefine("runtime.enoptrdata", obj.SNOPTRDATA, int64(noptr.Vaddr+noptr.Length))
	xdefine("runtime.bss", obj.SBSS, int64(bss.Vaddr))
	xdefine("runtime.ebss", obj.SBSS, int64(bss.Vaddr+bss.Length))
	xdefine("runtime.data", obj.SDATA, int64(data.Vaddr))
	xdefine("runtime.edata", obj.SDATA, int64(data.Vaddr+data.Length))
	xdefine("runtime.noptrbss", obj.SNOPTRBSS, int64(noptrbss.Vaddr))
	xdefine("runtime.enoptrbss", obj.SNOPTRBSS, int64(noptrbss.Vaddr+noptrbss.Length))
	xdefine("runtime.end", obj.SBSS, int64(Segdata.Vaddr+Segdata.Length))
}
