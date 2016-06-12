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
	"cmd/internal/gcprog"
	"cmd/internal/obj"
	"cmd/internal/sys"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
)

func Symgrow(ctxt *Link, s *LSym, siz int64) {
	if int64(int(siz)) != siz {
		log.Fatalf("symgrow size %d too long", siz)
	}
	if int64(len(s.P)) >= siz {
		return
	}
	if cap(s.P) < int(siz) {
		p := make([]byte, 2*(siz+1))
		s.P = append(p[:0], s.P...)
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
	s.Attr |= AttrReachable
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
		ctxt.Arch.ByteOrder.PutUint64(s.P[off:], v)
	}

	return off + wid
}

func Addbytes(ctxt *Link, s *LSym, bytes []byte) int64 {
	if s.Type == 0 {
		s.Type = obj.SDATA
	}
	s.Attr |= AttrReachable
	s.P = append(s.P, bytes...)
	s.Size = int64(len(s.P))

	return s.Size
}

func adduintxx(ctxt *Link, s *LSym, v uint64, wid int) int64 {
	off := s.Size
	setuintxx(ctxt, s, off, v, int64(wid))
	return off
}

func Adduint8(ctxt *Link, s *LSym, v uint8) int64 {
	off := s.Size
	if s.Type == 0 {
		s.Type = obj.SDATA
	}
	s.Attr |= AttrReachable
	s.Size++
	s.P = append(s.P, v)

	return off
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
	return adduintxx(ctxt, s, v, SysArch.IntSize)
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
	s.Attr |= AttrReachable
	i := s.Size
	s.Size += int64(ctxt.Arch.PtrSize)
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.PtrSize)
	r.Type = obj.R_ADDR
	r.Add = add
	return i + int64(r.Siz)
}

func Addpcrelplus(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = obj.SDATA
	}
	s.Attr |= AttrReachable
	i := s.Size
	s.Size += 4
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Add = add
	r.Type = obj.R_PCREL
	r.Siz = 4
	if SysArch.Family == sys.S390X {
		r.Variant = RV_390_DBL
	}
	return i + int64(r.Siz)
}

func Addaddr(ctxt *Link, s *LSym, t *LSym) int64 {
	return Addaddrplus(ctxt, s, t, 0)
}

func setaddrplus(ctxt *Link, s *LSym, off int64, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = obj.SDATA
	}
	s.Attr |= AttrReachable
	if off+int64(ctxt.Arch.PtrSize) > s.Size {
		s.Size = off + int64(ctxt.Arch.PtrSize)
		Symgrow(ctxt, s, s.Size)
	}

	r := Addrel(s)
	r.Sym = t
	r.Off = int32(off)
	r.Siz = uint8(ctxt.Arch.PtrSize)
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
	s.Attr |= AttrReachable
	i := s.Size
	s.Size += int64(ctxt.Arch.PtrSize)
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.PtrSize)
	r.Type = obj.R_SIZE
	return i + int64(r.Siz)
}

func addaddrplus4(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = obj.SDATA
	}
	s.Attr |= AttrReachable
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
			if !(SysArch.Family == sys.PPC64 && Linkmode == LinkExternal && r.Sym.Name == ".TOC.") {
				Diag("unhandled relocation for %s (type %d rtype %d)", r.Sym.Name, r.Sym.Type, r.Type)
			}
		}
		if r.Sym != nil && r.Sym.Type != obj.STLSBSS && !r.Sym.Attr.Reachable() {
			Diag("unreachable sym in relocation: %s %s", s.Name, r.Sym.Name)
		}

		// TODO(mundaym): remove this special case - see issue 14218.
		if SysArch.Family == sys.S390X {
			switch r.Type {
			case obj.R_PCRELDBL:
				r.Type = obj.R_PCREL
				r.Variant = RV_390_DBL
			case obj.R_CALL:
				r.Variant = RV_390_DBL
			}
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
			isAndroidX86 := goos == "android" && (SysArch.InFamily(sys.AMD64, sys.I386))

			if Linkmode == LinkExternal && Iself && HEADTYPE != obj.Hopenbsd && !isAndroidX86 {
				r.Done = 0
				if r.Sym == nil {
					r.Sym = Ctxt.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if SysArch.Family != sys.AMD64 {
					o = r.Add
				}
				break
			}

			if Iself && SysArch.Family == sys.ARM {
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
			isAndroidX86 := goos == "android" && (SysArch.InFamily(sys.AMD64, sys.I386))

			if Linkmode == LinkExternal && Iself && HEADTYPE != obj.Hopenbsd && !isAndroidX86 {
				r.Done = 0
				if r.Sym == nil {
					r.Sym = Ctxt.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if SysArch.Family != sys.AMD64 {
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
					if SysArch.Family == sys.AMD64 {
						o = 0
					}
				} else if HEADTYPE == obj.Hdarwin {
					// ld64 for arm64 has a bug where if the address pointed to by o exists in the
					// symbol table (dynid >= 0), or is inside a symbol that exists in the symbol
					// table, then it will add o twice into the relocated value.
					// The workaround is that on arm64 don't ever add symaddr to o and always use
					// extern relocation by requiring rs->dynid >= 0.
					if rs.Type != obj.SHOSTOBJ {
						if SysArch.Family == sys.ARM64 && rs.Dynid < 0 {
							Diag("R_ADDR reloc to %s+%d is not supported on darwin/arm64", rs.Name, o)
						}
						if SysArch.Family != sys.ARM64 {
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
			if int32(o) < 0 && SysArch.PtrSize > 4 && siz == 4 {
				Diag("non-pc-relative relocation address is too big: %#x (%#x + %#x)", uint64(o), Symaddr(r.Sym), r.Add)
				errorexit()
			}

		case obj.R_DWARFREF:
			if r.Sym.Sect == nil {
				Diag("missing DWARF section: %s from %s", r.Sym.Name, s.Name)
			}
			if Linkmode == LinkExternal {
				r.Done = 0
				r.Type = obj.R_ADDR

				r.Xsym = Linkrlookup(Ctxt, r.Sym.Sect.Name, 0)
				r.Xadd = r.Add + Symaddr(r.Sym) - int64(r.Sym.Sect.Vaddr)
				o = r.Xadd
				rs = r.Xsym
				if Iself && SysArch.Family == sys.AMD64 {
					o = 0
				}
				break
			}
			o = Symaddr(r.Sym) + r.Add - int64(r.Sym.Sect.Vaddr)

		case obj.R_ADDROFF:
			o = Symaddr(r.Sym) - int64(r.Sym.Sect.Vaddr) + r.Add

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
					if SysArch.Family == sys.AMD64 {
						o = 0
					}
				} else if HEADTYPE == obj.Hdarwin {
					if r.Type == obj.R_CALL {
						if rs.Type != obj.SHOSTOBJ {
							o += int64(uint64(Symaddr(rs)) - rs.Sect.Vaddr)
						}
						o -= int64(r.Off) // relative to section offset, not symbol
					} else if SysArch.Family == sys.ARM {
						// see ../arm/asm.go:/machoreloc1
						o += Symaddr(rs) - int64(Ctxt.Cursym.Value) - int64(r.Off)
					} else {
						o += int64(r.Siz)
					}
				} else if HEADTYPE == obj.Hwindows && SysArch.Family == sys.AMD64 { // only amd64 needs PCREL
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
		fmt.Fprintf(Bso, "%5.2f reloc\n", obj.Cputime())
	}
	Bso.Flush()

	for _, s := range Ctxt.Textp {
		relocsym(s)
	}
	for _, sym := range datap {
		relocsym(sym)
	}
	for s := dwarfp; s != nil; s = s.Next {
		relocsym(s)
	}
}

func dynrelocsym(s *LSym) {
	if HEADTYPE == obj.Hwindows && Linkmode != LinkExternal {
		rel := Linklookup(Ctxt, ".rel", 0)
		if s == rel {
			return
		}
		for ri := 0; ri < len(s.R); ri++ {
			r := &s.R[ri]
			targ := r.Sym
			if targ == nil {
				continue
			}
			if !targ.Attr.Reachable() {
				Diag("internal inconsistency: dynamic symbol %s is not reachable.", targ.Name)
			}
			if r.Sym.Plt == -2 && r.Sym.Got != -2 { // make dynimport JMP table for PE object files.
				targ.Plt = int32(rel.Size)
				r.Sym = rel
				r.Add = int64(targ.Plt)

				// jmp *addr
				if SysArch.Family == sys.I386 {
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

	for ri := 0; ri < len(s.R); ri++ {
		r := &s.R[ri]
		if r.Sym != nil && r.Sym.Type == obj.SDYNIMPORT || r.Type >= 256 {
			if r.Sym != nil && !r.Sym.Attr.Reachable() {
				Diag("internal inconsistency: dynamic symbol %s is not reachable.", r.Sym.Name)
			}
			Thearch.Adddynrel(s, r)
		}
	}
}

func dynreloc(data *[obj.SXREF][]*LSym) {
	// -d suppresses dynamic loader format, so we may as well not
	// compute these sections or mark their symbols as reachable.
	if Debug['d'] != 0 && HEADTYPE != obj.Hwindows {
		return
	}
	if Debug['v'] != 0 {
		fmt.Fprintf(Bso, "%5.2f reloc\n", obj.Cputime())
	}
	Bso.Flush()

	for _, s := range Ctxt.Textp {
		dynrelocsym(s)
	}
	for _, syms := range data {
		for _, sym := range syms {
			dynrelocsym(sym)
		}
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
	for ; sym != nil; sym = sym.Next {
		if sym.Type&obj.SSUB != 0 {
			continue
		}
		if sym.Value >= eaddr {
			break
		}
		Ctxt.Cursym = sym
		if sym.Value < addr {
			Diag("phase error: addr=%#x but sym=%#x type=%d", addr, sym.Value, sym.Type)
			errorexit()
		}

		if addr < sym.Value {
			strnput("", int(sym.Value-addr))
			addr = sym.Value
		}
		Cwrite(sym.P)
		addr += int64(len(sym.P))
		if addr < sym.Value+sym.Size {
			strnput("", int(sym.Value+sym.Size-addr))
			addr = sym.Value + sym.Size
		}
		if addr != sym.Value+sym.Size {
			Diag("phase error: addr=%#x value+size=%#x", addr, sym.Value+sym.Size)
			errorexit()
		}

		if sym.Value+sym.Size >= eaddr {
			break
		}
	}

	if addr < eaddr {
		strnput("", int(eaddr-addr))
	}
	Cflush()
}

func Codeblk(addr int64, size int64) {
	CodeblkPad(addr, size, zeros[:])
}
func CodeblkPad(addr int64, size int64, pad []byte) {
	if Debug['a'] != 0 {
		fmt.Fprintf(Bso, "codeblk [%#x,%#x) at offset %#x\n", addr, addr+size, Cpos())
	}

	blkSlice(Ctxt.Textp, addr, size, pad)

	/* again for printing */
	if Debug['a'] == 0 {
		return
	}

	syms := Ctxt.Textp
	for i, sym := range syms {
		if !sym.Attr.Reachable() {
			continue
		}
		if sym.Value >= addr {
			syms = syms[i:]
			break
		}
	}

	eaddr := addr + size
	var q []byte
	for _, sym := range syms {
		if !sym.Attr.Reachable() {
			continue
		}
		if sym.Value >= eaddr {
			break
		}

		if addr < sym.Value {
			fmt.Fprintf(Bso, "%-20s %.8x|", "_", uint64(addr))
			for ; addr < sym.Value; addr++ {
				fmt.Fprintf(Bso, " %.2x", 0)
			}
			fmt.Fprintf(Bso, "\n")
		}

		fmt.Fprintf(Bso, "%.6x\t%-20s\n", uint64(addr), sym.Name)
		q = sym.P

		for len(q) >= 16 {
			fmt.Fprintf(Bso, "%.6x\t% x\n", uint64(addr), q[:16])
			addr += 16
			q = q[16:]
		}

		if len(q) > 0 {
			fmt.Fprintf(Bso, "%.6x\t% x\n", uint64(addr), q)
			addr += int64(len(q))
		}
	}

	if addr < eaddr {
		fmt.Fprintf(Bso, "%-20s %.8x|", "_", uint64(addr))
		for ; addr < eaddr; addr++ {
			fmt.Fprintf(Bso, " %.2x", 0)
		}
	}

	Bso.Flush()
}

// blkSlice is a variant of blk that processes slices.
// After text symbols are converted from a linked list to a slice,
// delete blk and give this function its name.
func blkSlice(syms []*LSym, addr, size int64, pad []byte) {
	for i, s := range syms {
		if s.Type&obj.SSUB == 0 && s.Value >= addr {
			syms = syms[i:]
			break
		}
	}

	eaddr := addr + size
	for _, s := range syms {
		if s.Type&obj.SSUB != 0 {
			continue
		}
		if s.Value >= eaddr {
			break
		}
		Ctxt.Cursym = s
		if s.Value < addr {
			Diag("phase error: addr=%#x but sym=%#x type=%d", addr, s.Value, s.Type)
			errorexit()
		}
		if addr < s.Value {
			strnputPad("", int(s.Value-addr), pad)
			addr = s.Value
		}
		Cwrite(s.P)
		addr += int64(len(s.P))
		if addr < s.Value+s.Size {
			strnputPad("", int(s.Value+s.Size-addr), pad)
			addr = s.Value + s.Size
		}
		if addr != s.Value+s.Size {
			Diag("phase error: addr=%#x value+size=%#x", addr, s.Value+s.Size)
			errorexit()
		}
		if s.Value+s.Size >= eaddr {
			break
		}
	}

	if addr < eaddr {
		strnputPad("", int(eaddr-addr), pad)
	}
	Cflush()
}

func Datblk(addr int64, size int64) {
	if Debug['a'] != 0 {
		fmt.Fprintf(Bso, "datblk [%#x,%#x) at offset %#x\n", addr, addr+size, Cpos())
	}

	blkSlice(datap, addr, size, zeros[:])

	/* again for printing */
	if Debug['a'] == 0 {
		return
	}

	syms := datap
	for i, sym := range syms {
		if sym.Value >= addr {
			syms = syms[i:]
			break
		}
	}

	eaddr := addr + size
	for _, sym := range syms {
		if sym.Value >= eaddr {
			break
		}
		if addr < sym.Value {
			fmt.Fprintf(Bso, "\t%.8x| 00 ...\n", uint64(addr))
			addr = sym.Value
		}

		fmt.Fprintf(Bso, "%s\n\t%.8x|", sym.Name, uint64(addr))
		for i, b := range sym.P {
			if i > 0 && i%16 == 0 {
				fmt.Fprintf(Bso, "\n\t%.8x|", uint64(addr)+uint64(i))
			}
			fmt.Fprintf(Bso, " %.2x", b)
		}

		addr += int64(len(sym.P))
		for ; addr < sym.Value+sym.Size; addr++ {
			fmt.Fprintf(Bso, " %.2x", 0)
		}
		fmt.Fprintf(Bso, "\n")

		if Linkmode != LinkExternal {
			continue
		}
		for _, r := range sym.R {
			rsname := ""
			if r.Sym != nil {
				rsname = r.Sym.Name
			}
			typ := "?"
			switch r.Type {
			case obj.R_ADDR:
				typ = "addr"
			case obj.R_PCREL:
				typ = "pcrel"
			case obj.R_CALL:
				typ = "call"
			}
			fmt.Fprintf(Bso, "\treloc %.8x/%d %s %s+%#x [%#x]\n", uint(sym.Value+int64(r.Off)), r.Siz, typ, rsname, r.Add, r.Sym.Value+r.Add)
		}
	}

	if addr < eaddr {
		fmt.Fprintf(Bso, "\t%.8x| 00 ...\n", uint(addr))
	}
	fmt.Fprintf(Bso, "\t%.8x|\n", uint(eaddr))
}

func Dwarfblk(addr int64, size int64) {
	if Debug['a'] != 0 {
		fmt.Fprintf(Bso, "dwarfblk [%#x,%#x) at offset %#x\n", addr, addr+size, Cpos())
	}

	blk(dwarfp, addr, size)
}

var zeros [512]byte

// strnput writes the first n bytes of s.
// If n is larger than len(s),
// it is padded with NUL bytes.
func strnput(s string, n int) {
	strnputPad(s, n, zeros[:])
}

// strnput writes the first n bytes of s.
// If n is larger than len(s),
// it is padded with the bytes in pad (repeated as needed).
func strnputPad(s string, n int, pad []byte) {
	if len(s) >= n {
		Cwritestring(s[:n])
	} else {
		Cwritestring(s)
		n -= len(s)
		for n > len(pad) {
			Cwrite(pad)
			n -= len(pad)

		}
		Cwrite(pad[:n])
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
	s.Attr |= AttrDuplicateOK
	reachable := s.Attr.Reachable()
	Addaddr(Ctxt, s, sp)
	adduintxx(Ctxt, s, uint64(len(value)), SysArch.PtrSize)

	// addstring, addaddr, etc., mark the symbols as reachable.
	// In this case that is not necessarily true, so stick to what
	// we know before entering this function.
	s.Attr.Set(AttrReachable, reachable)

	strdata = append(strdata, s)

	sp.Attr.Set(AttrReachable, reachable)
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
	s.Attr |= AttrReachable
	r := s.Size
	if s.Name == ".shstrtab" {
		elfsetstring(str, int(r))
	}
	s.P = append(s.P, str...)
	s.P = append(s.P, 0)
	s.Size = int64(len(s.P))
	return r
}

// addgostring adds str, as a Go string value, to s. symname is the name of the
// symbol used to define the string data and must be unique per linked object.
func addgostring(s *LSym, symname, str string) {
	sym := Linklookup(Ctxt, symname, 0)
	if sym.Type != obj.Sxxx {
		Diag("duplicate symname in addgostring: %s", symname)
	}
	sym.Attr |= AttrReachable
	sym.Attr |= AttrLocal
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
	sp.Attr |= AttrDuplicateOK
	Addaddr(Ctxt, sp, s)
}

func dosymtype() {
	for _, s := range Ctxt.Allsym {
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

// symalign returns the required alignment for the given symbol s.
func symalign(s *LSym) int32 {
	min := int32(Thearch.Minalign)
	if s.Align >= min {
		return s.Align
	} else if s.Align != 0 {
		return min
	}
	if (strings.HasPrefix(s.Name, "go.string.") && !strings.HasPrefix(s.Name, "go.string.hdr.")) || strings.HasPrefix(s.Name, "type..namedata.") {
		// String data is just bytes.
		// If we align it, we waste a lot of space to padding.
		return min
	}
	align := int32(Thearch.Maxalign)
	for int64(align) > s.Size && align > min {
		align >>= 1
	}
	return align
}

func aligndatsize(datsize int64, s *LSym) int64 {
	return Rnd(datsize, int64(symalign(s)))
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
	p.w.ZeroUntil(size / int64(SysArch.PtrSize))
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

	ptrsize := int64(SysArch.PtrSize)
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

// dataSortKey is used to sort a slice of data symbol *LSym pointers.
// The sort keys are kept inline to improve cache behaviour while sorting.
type dataSortKey struct {
	size int64
	name string
	lsym *LSym
}

type bySizeAndName []dataSortKey

func (d bySizeAndName) Len() int      { return len(d) }
func (d bySizeAndName) Swap(i, j int) { d[i], d[j] = d[j], d[i] }
func (d bySizeAndName) Less(i, j int) bool {
	s1, s2 := d[i], d[j]
	if s1.size != s2.size {
		return s1.size < s2.size
	}
	return s1.name < s2.name
}

const cutoff int64 = 2e9 // 2 GB (or so; looks better in errors than 2^31)

func checkdatsize(datsize int64, symn int) {
	if datsize > cutoff {
		Diag("too much data in section %v (over %d bytes)", symn, cutoff)
	}
}

func list2slice(s *LSym) []*LSym {
	var syms []*LSym
	for ; s != nil; s = s.Next {
		syms = append(syms, s)
	}
	return syms
}

// datap is a collection of reachable data symbols in address order.
// Generated by dodata.
var datap []*LSym

func dodata() {
	if Debug['v'] != 0 {
		fmt.Fprintf(Bso, "%5.2f dodata\n", obj.Cputime())
	}
	Bso.Flush()

	// Collect data symbols by type into data.
	var data [obj.SXREF][]*LSym
	for _, s := range Ctxt.Allsym {
		if !s.Attr.Reachable() || s.Attr.Special() {
			continue
		}
		if s.Type <= obj.STEXT || s.Type >= obj.SXREF {
			continue
		}
		data[s.Type] = append(data[s.Type], s)
	}

	// Now that we have the data symbols, but before we start
	// to assign addresses, record all the necessary
	// dynamic relocations. These will grow the relocation
	// symbol, which is itself data.
	//
	// On darwin, we need the symbol table numbers for dynreloc.
	if HEADTYPE == obj.Hdarwin {
		machosymorder()
	}
	dynreloc(&data)

	if UseRelro() {
		// "read only" data with relocations needs to go in its own section
		// when building a shared library. We do this by boosting objects of
		// type SXXX with relocations to type SXXXRELRO.
		for symnro := int16(obj.STYPE); symnro < obj.STYPERELRO; symnro++ {
			symnrelro := symnro + obj.STYPERELRO - obj.STYPE

			ro := []*LSym{}
			relro := data[symnrelro]

			for _, s := range data[symnro] {
				isRelro := len(s.R) > 0
				switch s.Type {
				case obj.STYPE, obj.SGOSTRINGHDR, obj.STYPERELRO, obj.SGOSTRINGHDRRELRO:
					// Symbols are not sorted yet, so it is possible
					// that an Outer symbol has been changed to a
					// relro Type before it reaches here.
					isRelro = true
				}
				if isRelro {
					s.Type = symnrelro
					if s.Outer != nil {
						s.Outer.Type = s.Type
					}
					relro = append(relro, s)
				} else {
					ro = append(ro, s)
				}
			}

			// Check that we haven't made two symbols with the same .Outer into
			// different types (because references two symbols with non-nil Outer
			// become references to the outer symbol + offset it's vital that the
			// symbol and the outer end up in the same section).
			for _, s := range relro {
				if s.Outer != nil && s.Outer.Type != s.Type {
					Diag("inconsistent types for %s and its Outer %s (%d != %d)",
						s.Name, s.Outer.Name, s.Type, s.Outer.Type)
				}
			}

			data[symnro] = ro
			data[symnrelro] = relro
		}
	}

	// Sort symbols.
	var dataMaxAlign [obj.SXREF]int32
	var wg sync.WaitGroup
	for symn := range data {
		symn := symn
		wg.Add(1)
		go func() {
			data[symn], dataMaxAlign[symn] = dodataSect(symn, data[symn])
			wg.Done()
		}()
	}
	wg.Wait()

	// Allocate sections.
	// Data is processed before segtext, because we need
	// to see all symbols in the .data and .bss sections in order
	// to generate garbage collection information.
	datsize := int64(0)

	// Writable sections.
	writableSects := []int{
		obj.SELFSECT,
		obj.SMACHO,
		obj.SMACHOGOT,
		obj.SWINDOWS,
	}
	for _, symn := range writableSects {
		for _, s := range data[symn] {
			sect := addsection(&Segdata, s.Name, 06)
			sect.Align = symalign(s)
			datsize = Rnd(datsize, int64(sect.Align))
			sect.Vaddr = uint64(datsize)
			s.Sect = sect
			s.Type = obj.SDATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
			sect.Length = uint64(datsize) - sect.Vaddr
		}
		checkdatsize(datsize, symn)
	}

	// .got (and .toc on ppc64)
	if len(data[obj.SELFGOT]) > 0 {
		sect := addsection(&Segdata, ".got", 06)
		sect.Align = dataMaxAlign[obj.SELFGOT]
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		var toc *LSym
		for _, s := range data[obj.SELFGOT] {
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

			datsize += s.Size
		}
		checkdatsize(datsize, obj.SELFGOT)
		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* pointer-free data */
	sect := addsection(&Segdata, ".noptrdata", 06)
	sect.Align = dataMaxAlign[obj.SNOPTRDATA]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.noptrdata", 0).Sect = sect
	Linklookup(Ctxt, "runtime.enoptrdata", 0).Sect = sect
	for _, s := range data[obj.SNOPTRDATA] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SDATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(datsize, obj.SNOPTRDATA)
	sect.Length = uint64(datsize) - sect.Vaddr

	hasinitarr := Linkshared

	/* shared library initializer */
	switch Buildmode {
	case BuildmodeCArchive, BuildmodeCShared, BuildmodeShared:
		hasinitarr = true
	}
	if hasinitarr {
		sect := addsection(&Segdata, ".init_array", 06)
		sect.Align = dataMaxAlign[obj.SINITARR]
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for _, s := range data[obj.SINITARR] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
		}
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(datsize, obj.SINITARR)
	}

	/* data */
	sect = addsection(&Segdata, ".data", 06)
	sect.Align = dataMaxAlign[obj.SDATA]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.data", 0).Sect = sect
	Linklookup(Ctxt, "runtime.edata", 0).Sect = sect
	var gc GCProg
	gc.Init("runtime.gcdata")
	for _, s := range data[obj.SDATA] {
		s.Sect = sect
		s.Type = obj.SDATA
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		gc.AddSym(s)
		datsize += s.Size
	}
	checkdatsize(datsize, obj.SDATA)
	sect.Length = uint64(datsize) - sect.Vaddr
	gc.End(int64(sect.Length))

	/* bss */
	sect = addsection(&Segdata, ".bss", 06)
	sect.Align = dataMaxAlign[obj.SBSS]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.bss", 0).Sect = sect
	Linklookup(Ctxt, "runtime.ebss", 0).Sect = sect
	gc = GCProg{}
	gc.Init("runtime.gcbss")
	for _, s := range data[obj.SBSS] {
		s.Sect = sect
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		gc.AddSym(s)
		datsize += s.Size
	}
	checkdatsize(datsize, obj.SBSS)
	sect.Length = uint64(datsize) - sect.Vaddr
	gc.End(int64(sect.Length))

	/* pointer-free bss */
	sect = addsection(&Segdata, ".noptrbss", 06)
	sect.Align = dataMaxAlign[obj.SNOPTRBSS]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.noptrbss", 0).Sect = sect
	Linklookup(Ctxt, "runtime.enoptrbss", 0).Sect = sect
	for _, s := range data[obj.SNOPTRBSS] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}

	sect.Length = uint64(datsize) - sect.Vaddr
	Linklookup(Ctxt, "runtime.end", 0).Sect = sect
	checkdatsize(datsize, obj.SNOPTRBSS)

	if len(data[obj.STLSBSS]) > 0 {
		var sect *Section
		if Iself && (Linkmode == LinkExternal || Debug['d'] == 0) && HEADTYPE != obj.Hopenbsd {
			sect = addsection(&Segdata, ".tbss", 06)
			sect.Align = int32(SysArch.PtrSize)
			sect.Vaddr = 0
		}
		datsize = 0

		for _, s := range data[obj.STLSBSS] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = datsize
			datsize += s.Size
		}
		checkdatsize(datsize, obj.STLSBSS)

		if sect != nil {
			sect.Length = uint64(datsize)
		}
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

	datsize = 0

	/* read-only executable ELF, Mach-O sections */
	if len(data[obj.STEXT]) != 0 {
		Diag("dodata found an STEXT symbol: %s", data[obj.STEXT][0].Name)
	}
	for _, s := range data[obj.SELFRXSECT] {
		sect := addsection(&Segtext, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(datsize, obj.SELFRXSECT)
	}

	/* read-only data */
	sect = addsection(segro, ".rodata", 04)

	sect.Vaddr = 0
	Linklookup(Ctxt, "runtime.rodata", 0).Sect = sect
	Linklookup(Ctxt, "runtime.erodata", 0).Sect = sect
	if !UseRelro() {
		Linklookup(Ctxt, "runtime.types", 0).Sect = sect
		Linklookup(Ctxt, "runtime.etypes", 0).Sect = sect
	}
	roSects := []int{
		obj.STYPE,
		obj.SSTRING,
		obj.SGOSTRING,
		obj.SGOSTRINGHDR,
		obj.SGOFUNC,
		obj.SGCBITS,
		obj.SRODATA,
		obj.SFUNCTAB,
	}
	for _, symn := range roSects {
		align := dataMaxAlign[symn]
		if sect.Align < align {
			sect.Align = align
		}
	}
	datsize = Rnd(datsize, int64(sect.Align))
	for _, symn := range roSects {
		for _, s := range data[symn] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Type = obj.SRODATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
		}
		checkdatsize(datsize, symn)
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

		sect.Vaddr = 0
		Linklookup(Ctxt, "runtime.types", 0).Sect = sect
		Linklookup(Ctxt, "runtime.etypes", 0).Sect = sect
		relroSects := []int{
			obj.STYPERELRO,
			obj.SSTRINGRELRO,
			obj.SGOSTRINGRELRO,
			obj.SGOSTRINGHDRRELRO,
			obj.SGOFUNCRELRO,
			obj.SGCBITSRELRO,
			obj.SRODATARELRO,
			obj.SFUNCTABRELRO,
		}
		for _, symn := range relroSects {
			align := dataMaxAlign[symn]
			if sect.Align < align {
				sect.Align = align
			}
		}
		datsize = Rnd(datsize, int64(sect.Align))
		for _, symn := range relroSects {
			for _, s := range data[symn] {
				datsize = aligndatsize(datsize, s)
				if s.Outer != nil && s.Outer.Sect != nil && s.Outer.Sect != sect {
					Diag("s.Outer (%s) in different section from s (%s)", s.Outer.Name, s.Name)
				}
				s.Sect = sect
				s.Type = obj.SRODATA
				s.Value = int64(uint64(datsize) - sect.Vaddr)
				datsize += s.Size
			}
			checkdatsize(datsize, symn)
		}

		sect.Length = uint64(datsize) - sect.Vaddr

	}

	/* typelink */
	sect = addsection(segro, relro_prefix+".typelink", relro_perms)
	sect.Align = dataMaxAlign[obj.STYPELINK]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.typelink", 0).Sect = sect
	Linklookup(Ctxt, "runtime.etypelink", 0).Sect = sect
	for _, s := range data[obj.STYPELINK] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(datsize, obj.STYPELINK)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* itablink */
	sect = addsection(segro, relro_prefix+".itablink", relro_perms)
	sect.Align = dataMaxAlign[obj.SITABLINK]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.itablink", 0).Sect = sect
	Linklookup(Ctxt, "runtime.eitablink", 0).Sect = sect
	for _, s := range data[obj.SITABLINK] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(datsize, obj.SITABLINK)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* gosymtab */
	sect = addsection(segro, relro_prefix+".gosymtab", relro_perms)
	sect.Align = dataMaxAlign[obj.SSYMTAB]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.symtab", 0).Sect = sect
	Linklookup(Ctxt, "runtime.esymtab", 0).Sect = sect
	for _, s := range data[obj.SSYMTAB] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(datsize, obj.SSYMTAB)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* gopclntab */
	sect = addsection(segro, relro_prefix+".gopclntab", relro_perms)
	sect.Align = dataMaxAlign[obj.SPCLNTAB]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	Linklookup(Ctxt, "runtime.pclntab", 0).Sect = sect
	Linklookup(Ctxt, "runtime.epclntab", 0).Sect = sect
	for _, s := range data[obj.SPCLNTAB] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(datsize, obj.SRODATA)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* read-only ELF, Mach-O sections */
	for _, s := range data[obj.SELFROSECT] {
		sect = addsection(segro, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(datsize, obj.SELFROSECT)

	for _, s := range data[obj.SMACHOPLT] {
		sect = addsection(segro, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(datsize, obj.SMACHOPLT)

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if datsize != int64(uint32(datsize)) {
		Diag("read-only data segment too large")
	}

	for symn := obj.SELFRXSECT; symn < obj.SXREF; symn++ {
		datap = append(datap, data[symn]...)
	}

	dwarfgeneratedebugsyms()

	var s *LSym
	for s = dwarfp; s != nil && s.Type == obj.SDWARFSECT; s = s.Next {
		sect = addsection(&Segdwarf, s.Name, 04)
		sect.Align = 1
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = obj.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(datsize, obj.SDWARFSECT)

	if s != nil {
		sect = addsection(&Segdwarf, ".debug_info", 04)
		sect.Align = 1
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for ; s != nil && s.Type == obj.SDWARFINFO; s = s.Next {
			s.Sect = sect
			s.Type = obj.SRODATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			s.Attr |= AttrLocal
			datsize += s.Size
		}
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(datsize, obj.SDWARFINFO)
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
	for sect := Segdwarf.Sect; sect != nil; sect = sect.Next {
		sect.Extnum = int16(n)
		n++
	}
}

func dodataSect(symn int, syms []*LSym) (result []*LSym, maxAlign int32) {
	if HEADTYPE == obj.Hdarwin {
		// Some symbols may no longer belong in syms
		// due to movement in machosymorder.
		newSyms := make([]*LSym, 0, len(syms))
		for _, s := range syms {
			if int(s.Type) == symn {
				newSyms = append(newSyms, s)
			}
		}
		syms = newSyms
	}

	symsSort := make([]dataSortKey, len(syms))
	for i, s := range syms {
		if s.Attr.OnList() {
			log.Fatalf("symbol %s listed multiple times", s.Name)
		}
		s.Attr |= AttrOnList
		switch {
		case s.Size < int64(len(s.P)):
			Diag("%s: initialize bounds (%d < %d)", s.Name, s.Size, len(s.P))
		case s.Size < 0:
			Diag("%s: negative size (%d bytes)", s.Name, s.Size)
		case s.Size > cutoff:
			Diag("%s: symbol too large (%d bytes)", s.Name, s.Size)
		}

		symsSort[i] = dataSortKey{
			size: s.Size,
			name: s.Name,
			lsym: s,
		}

		switch s.Type {
		case obj.SELFGOT:
			// For ppc64, we want to interleave the .got and .toc sections
			// from input files. Both are type SELFGOT, so in that case
			// we skip size comparison and fall through to the name
			// comparison (conveniently, .got sorts before .toc).
			symsSort[i].size = 0
		case obj.STYPELINK:
			// Sort typelinks by the rtype.string field so the reflect
			// package can binary search type links.
			symsSort[i].name = string(decodetype_str(s.R[0].Sym))
		}
	}

	sort.Sort(bySizeAndName(symsSort))

	for i, symSort := range symsSort {
		syms[i] = symSort.lsym
		align := symalign(symSort.lsym)
		if maxAlign < align {
			maxAlign = align
		}
	}

	if Iself && symn == obj.SELFROSECT {
		// Make .rela and .rela.plt contiguous, the ELF ABI requires this
		// and Solaris actually cares.
		reli, plti := -1, -1
		for i, s := range syms {
			switch s.Name {
			case ".rel.plt", ".rela.plt":
				plti = i
			case ".rel", ".rela":
				reli = i
			}
		}
		if reli >= 0 && plti >= 0 && plti != reli+1 {
			var first, second int
			if plti > reli {
				first, second = reli, plti
			} else {
				first, second = plti, reli
			}
			rel, plt := syms[reli], syms[plti]
			copy(syms[first+2:], syms[first+1:second])
			syms[first+0] = rel
			syms[first+1] = plt
		}
	}

	return syms, maxAlign
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
	sym.Attr |= AttrReachable
	// The \xff is invalid UTF-8, meant to make it less likely
	// to find one of these accidentally.
	data := "\xff Go build ID: " + strconv.Quote(buildid) + "\n \xff"
	sym.Type = obj.STEXT
	sym.P = []byte(data)
	sym.Size = int64(len(sym.P))

	Ctxt.Textp = append(Ctxt.Textp, nil)
	copy(Ctxt.Textp[1:], Ctxt.Textp)
	Ctxt.Textp[0] = sym
}

// assign addresses to text
func textaddress() {
	addsection(&Segtext, ".text", 05)

	// Assign PCs in text segment.
	// Could parallelize, by assigning to text
	// and then letting threads copy down, but probably not worth it.
	sect := Segtext.Sect

	sect.Align = int32(Funcalign)
	Linklookup(Ctxt, "runtime.text", 0).Sect = sect
	Linklookup(Ctxt, "runtime.etext", 0).Sect = sect
	if HEADTYPE == obj.Hwindows {
		Linklookup(Ctxt, ".text", 0).Sect = sect
	}
	va := uint64(INITTEXT)
	sect.Vaddr = va
	for _, sym := range Ctxt.Textp {
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
		for sub := sym; sub != nil; sub = sub.Sub {
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

	va = uint64(Rnd(int64(va), int64(INITRND)))
	Segdwarf.Rwx = 06
	Segdwarf.Vaddr = va
	Segdwarf.Fileoff = Segdata.Fileoff + uint64(Rnd(int64(Segdata.Filelen), int64(INITRND)))
	Segdwarf.Filelen = 0
	if HEADTYPE == obj.Hwindows {
		Segdwarf.Fileoff = Segdata.Fileoff + uint64(Rnd(int64(Segdata.Filelen), int64(PEFILEALIGN)))
	}
	for s := Segdwarf.Sect; s != nil; s = s.Next {
		vlen = int64(s.Length)
		if s.Next != nil {
			vlen = int64(s.Next.Vaddr - s.Vaddr)
		}
		s.Vaddr = va
		va += uint64(vlen)
		if HEADTYPE == obj.Hwindows {
			va = uint64(Rnd(int64(va), PEFILEALIGN))
		}
		Segdwarf.Length = va - Segdwarf.Vaddr
	}

	Segdwarf.Filelen = va - Segdwarf.Vaddr

	text := Segtext.Sect
	var rodata *Section
	if Segrodata.Sect != nil {
		rodata = Segrodata.Sect
	} else {
		rodata = text.Next
	}
	var relrodata *Section
	typelink := rodata.Next
	if UseRelro() {
		// There is another section (.data.rel.ro) when building a shared
		// object on elf systems.
		relrodata = typelink
		typelink = typelink.Next
	}
	itablink := typelink.Next
	symtab := itablink.Next
	pclntab := symtab.Next

	for _, s := range datap {
		Ctxt.Cursym = s
		if s.Sect != nil {
			s.Value += int64(s.Sect.Vaddr)
		}
		for sub := s.Sub; sub != nil; sub = sub.Sub {
			sub.Value += s.Value
		}
	}

	for sym := dwarfp; sym != nil; sym = sym.Next {
		Ctxt.Cursym = sym
		if sym.Sect != nil {
			sym.Value += int64(sym.Sect.Vaddr)
		}
		for sub := sym.Sub; sub != nil; sub = sub.Sub {
			sub.Value += sym.Value
		}
	}

	if Buildmode == BuildmodeShared {
		s := Linklookup(Ctxt, "go.link.abihashbytes", 0)
		sectSym := Linklookup(Ctxt, ".note.go.abihash", 0)
		s.Sect = sectSym.Sect
		s.Value = int64(sectSym.Sect.Vaddr + 16)
	}

	types := relrodata
	if types == nil {
		types = rodata
	}

	xdefine("runtime.text", obj.STEXT, int64(text.Vaddr))
	xdefine("runtime.etext", obj.STEXT, int64(text.Vaddr+text.Length))
	if HEADTYPE == obj.Hwindows {
		xdefine(".text", obj.STEXT, int64(text.Vaddr))
	}
	xdefine("runtime.rodata", obj.SRODATA, int64(rodata.Vaddr))
	xdefine("runtime.erodata", obj.SRODATA, int64(rodata.Vaddr+rodata.Length))
	xdefine("runtime.types", obj.SRODATA, int64(types.Vaddr))
	xdefine("runtime.etypes", obj.SRODATA, int64(types.Vaddr+types.Length))
	xdefine("runtime.typelink", obj.SRODATA, int64(typelink.Vaddr))
	xdefine("runtime.etypelink", obj.SRODATA, int64(typelink.Vaddr+typelink.Length))
	xdefine("runtime.itablink", obj.SRODATA, int64(itablink.Vaddr))
	xdefine("runtime.eitablink", obj.SRODATA, int64(itablink.Vaddr+itablink.Length))

	sym := Linklookup(Ctxt, "runtime.gcdata", 0)
	sym.Attr |= AttrLocal
	xdefine("runtime.egcdata", obj.SRODATA, Symaddr(sym)+sym.Size)
	Linklookup(Ctxt, "runtime.egcdata", 0).Sect = sym.Sect

	sym = Linklookup(Ctxt, "runtime.gcbss", 0)
	sym.Attr |= AttrLocal
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
