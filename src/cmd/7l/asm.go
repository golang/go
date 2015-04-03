// Inferno utils/5l/asm.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/asm.c
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

package main

import (
	"cmd/internal/ld"
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"log"
)

func gentext() {}

func needlib(name string) int {
	if name[0] == '\x00' {
		return 0
	}

	/* reuse hash code in symbol table */
	p := fmt.Sprintf(".dynlib.%s", name)

	s := ld.Linklookup(ld.Ctxt, p, 0)

	if s.Type == 0 {
		s.Type = 100 // avoid SDATA, etc.
		return 1
	}

	return 0
}

func adddynrela(rel *ld.LSym, s *ld.LSym, r *ld.Reloc) {
	log.Fatalf("adddynrela not implemented")
}

func adddynrel(s *ld.LSym, r *ld.Reloc) {
	log.Fatalf("adddynrel not implemented")
}

func elfreloc1(r *ld.Reloc, sectoff int64) int {
	ld.Thearch.Vput(uint64(sectoff))

	elfsym := r.Xsym.Elfsym
	switch r.Type {
	default:
		return -1

	case ld.R_ADDR:
		switch r.Siz {
		case 4:
			ld.Thearch.Vput(ld.R_AARCH64_ABS32 | uint64(elfsym)<<32)
		case 8:
			ld.Thearch.Vput(ld.R_AARCH64_ABS64 | uint64(elfsym)<<32)
		default:
			return -1
		}

	case ld.R_ADDRARM64:
		// two relocations: R_AARCH64_ADR_PREL_PG_HI21 and R_AARCH64_ADD_ABS_LO12_NC
		ld.Thearch.Vput(ld.R_AARCH64_ADR_PREL_PG_HI21 | uint64(elfsym)<<32)
		ld.Thearch.Vput(uint64(r.Xadd))
		ld.Thearch.Vput(uint64(sectoff + 4))
		ld.Thearch.Vput(ld.R_AARCH64_ADD_ABS_LO12_NC | uint64(elfsym)<<32)

	case ld.R_CALLARM64:
		if r.Siz != 4 {
			return -1
		}
		ld.Thearch.Vput(ld.R_AARCH64_CALL26 | uint64(elfsym)<<32)

	}
	ld.Thearch.Vput(uint64(r.Xadd))

	return 0
}

func elfsetupplt() {
	// TODO(aram)
	return
}

func machoreloc1(r *ld.Reloc, sectoff int64) int {
	return -1
}

func archreloc(r *ld.Reloc, s *ld.LSym, val *int64) int {
	if ld.Linkmode == ld.LinkExternal {
		switch r.Type {
		default:
			return -1

		case ld.R_ADDRARM64:
			r.Done = 0

			// the first instruction is always at the lower address, this is endian neutral;
			// but note that o0 and o1 should still use the target endian.
			o0 := ld.Thelinkarch.ByteOrder.Uint32(s.P[r.Off : r.Off+4])
			o1 := ld.Thelinkarch.ByteOrder.Uint32(s.P[r.Off+4 : r.Off+8])

			// when laid out, the instruction order must always be o1, o2.
			if ld.Ctxt.Arch.ByteOrder == binary.BigEndian {
				*val = int64(o0)<<32 | int64(o1)
			} else {
				*val = int64(o1)<<32 | int64(o0)
			}

			// set up addend for eventual relocation via outer symbol.
			rs := r.Sym
			r.Xadd = r.Add
			for rs.Outer != nil {
				r.Xadd += ld.Symaddr(rs) - ld.Symaddr(rs.Outer)
				rs = rs.Outer
			}

			if rs.Type != ld.SHOSTOBJ && rs.Sect == nil {
				ld.Diag("missing section for %s", rs.Name)
			}
			r.Xsym = rs

			return 0

		case ld.R_CALLARM64:
			r.Done = 0
			r.Xsym = r.Sym
			*val = int64(0xfc000000 & uint32(r.Add))
			r.Xadd = int64((uint32(r.Add) &^ 0xfc000000) * 4)
			r.Add = 0
			return 0
		}
	}

	switch r.Type {
	case ld.R_CONST:
		*val = r.Add
		return 0

	case ld.R_GOTOFF:
		*val = ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ld.Linklookup(ld.Ctxt, ".got", 0))
		return 0

	case ld.R_ADDRARM64:
		t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
		if t >= 1<<32 || t < -1<<32 {
			ld.Diag("program too large, address relocation distance = %d", t)
		}

		// the first instruction is always at the lower address, this is endian neutral;
		// but note that o0 and o1 should still use the target endian.
		o0 := ld.Thelinkarch.ByteOrder.Uint32(s.P[r.Off : r.Off+4])
		o1 := ld.Thelinkarch.ByteOrder.Uint32(s.P[r.Off+4 : r.Off+8])

		o0 |= (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
		o1 |= uint32(t&0xfff) << 10

		// when laid out, the instruction order must always be o1, o2.
		if ld.Ctxt.Arch.ByteOrder == binary.BigEndian {
			*val = int64(o0)<<32 | int64(o1)
		} else {
			*val = int64(o1)<<32 | int64(o0)
		}
		return 0

	case ld.R_CALLARM64:
		*val = int64((0xfc000000 & uint32(r.Add)) | uint32((ld.Symaddr(r.Sym)+r.Add*4-(s.Value+int64(r.Off)))/4))
		return 0
	}

	return -1
}

func archrelocvariant(r *ld.Reloc, s *ld.LSym, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return -1
}

func adddynsym(ctxt *ld.Link, s *ld.LSym) {
	// TODO(minux): implement when needed.
}

func adddynlib(lib string) {
	if needlib(lib) == 0 {
		return
	}

	if ld.Iself {
		s := ld.Linklookup(ld.Ctxt, ".dynstr", 0)
		if s.Size == 0 {
			ld.Addstring(s, "")
		}
		ld.Elfwritedynent(ld.Linklookup(ld.Ctxt, ".dynamic", 0), ld.DT_NEEDED, uint64(ld.Addstring(s, lib)))
	} else {
		ld.Diag("adddynlib: unsupported binary format")
	}
}

func asmb() {
	if ld.Debug['v'] != 0 {
		fmt.Fprintf(&ld.Bso, "%5.2f asmb\n", obj.Cputime())
	}
	ld.Bflush(&ld.Bso)

	if ld.Iself {
		ld.Asmbelfsetup()
	}

	sect := ld.Segtext.Sect
	ld.Cseek(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
	ld.Codeblk(int64(sect.Vaddr), int64(sect.Length))
	for sect = sect.Next; sect != nil; sect = sect.Next {
		ld.Cseek(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
		ld.Datblk(int64(sect.Vaddr), int64(sect.Length))
	}

	if ld.Segrodata.Filelen > 0 {
		if ld.Debug['v'] != 0 {
			fmt.Fprintf(&ld.Bso, "%5.2f rodatblk\n", obj.Cputime())
		}
		ld.Bflush(&ld.Bso)

		ld.Cseek(int64(ld.Segrodata.Fileoff))
		ld.Datblk(int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}

	if ld.Debug['v'] != 0 {
		fmt.Fprintf(&ld.Bso, "%5.2f datblk\n", obj.Cputime())
	}
	ld.Bflush(&ld.Bso)

	ld.Cseek(int64(ld.Segdata.Fileoff))
	ld.Datblk(int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	/* output symbol table */
	ld.Symsize = 0

	ld.Lcsize = 0
	symo := uint32(0)
	if ld.Debug['s'] == 0 {
		// TODO: rationalize
		if ld.Debug['v'] != 0 {
			fmt.Fprintf(&ld.Bso, "%5.2f sym\n", obj.Cputime())
		}
		ld.Bflush(&ld.Bso)
		switch ld.HEADTYPE {
		default:
			if ld.Iself {
				symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(ld.INITRND)))
			}

		case ld.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)
		}

		ld.Cseek(int64(symo))
		switch ld.HEADTYPE {
		default:
			if ld.Iself {
				if ld.Debug['v'] != 0 {
					fmt.Fprintf(&ld.Bso, "%5.2f elfsym\n", obj.Cputime())
				}
				ld.Asmelfsym()
				ld.Cflush()
				ld.Cwrite(ld.Elfstrdat)

				if ld.Debug['v'] != 0 {
					fmt.Fprintf(&ld.Bso, "%5.2f dwarf\n", obj.Cputime())
				}
				ld.Dwarfemitdebugsections()

				if ld.Linkmode == ld.LinkExternal {
					ld.Elfemitreloc()
				}
			}

		case ld.Hplan9:
			ld.Asmplan9sym()
			ld.Cflush()

			sym := ld.Linklookup(ld.Ctxt, "pclntab", 0)
			if sym != nil {
				ld.Lcsize = int32(len(sym.P))
				for i := 0; int32(i) < ld.Lcsize; i++ {
					ld.Cput(uint8(sym.P[i]))
				}

				ld.Cflush()
			}
		}
	}

	ld.Ctxt.Cursym = nil
	if ld.Debug['v'] != 0 {
		fmt.Fprintf(&ld.Bso, "%5.2f header\n", obj.Cputime())
	}
	ld.Bflush(&ld.Bso)
	ld.Cseek(0)
	switch ld.HEADTYPE {
	default:
	case ld.Hplan9: /* plan 9 */
		ld.Thearch.Lput(0x647)                      /* magic */
		ld.Thearch.Lput(uint32(ld.Segtext.Filelen)) /* sizes */
		ld.Thearch.Lput(uint32(ld.Segdata.Filelen))
		ld.Thearch.Lput(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ld.Thearch.Lput(uint32(ld.Symsize))      /* nsyms */
		ld.Thearch.Lput(uint32(ld.Entryvalue())) /* va of entry */
		ld.Thearch.Lput(0)
		ld.Thearch.Lput(uint32(ld.Lcsize))

	case ld.Hlinux,
		ld.Hfreebsd,
		ld.Hnetbsd,
		ld.Hopenbsd,
		ld.Hnacl:
		ld.Asmbelf(int64(symo))
	}

	ld.Cflush()
	if ld.Debug['c'] != 0 {
		fmt.Printf("textsize=%d\n", ld.Segtext.Filelen)
		fmt.Printf("datsize=%d\n", ld.Segdata.Filelen)
		fmt.Printf("bsssize=%d\n", ld.Segdata.Length-ld.Segdata.Filelen)
		fmt.Printf("symsize=%d\n", ld.Symsize)
		fmt.Printf("lcsize=%d\n", ld.Lcsize)
		fmt.Printf("total=%d\n", ld.Segtext.Filelen+ld.Segdata.Length+uint64(ld.Symsize)+uint64(ld.Lcsize))
	}
}
