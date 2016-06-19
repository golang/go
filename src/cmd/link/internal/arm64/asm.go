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

package arm64

import (
	"cmd/internal/obj"
	"cmd/link/internal/ld"
	"encoding/binary"
	"fmt"
	"log"
)

func gentext() {
	if !ld.DynlinkingGo() {
		return
	}
	addmoduledata := ld.Linklookup(ld.Ctxt, "runtime.addmoduledata", 0)
	if addmoduledata.Type == obj.STEXT {
		// we're linking a module containing the runtime -> no need for
		// an init function
		return
	}
	addmoduledata.Attr |= ld.AttrReachable
	initfunc := ld.Linklookup(ld.Ctxt, "go.link.addmoduledata", 0)
	initfunc.Type = obj.STEXT
	initfunc.Attr |= ld.AttrLocal
	initfunc.Attr |= ld.AttrReachable
	o := func(op uint32) {
		ld.Adduint32(ld.Ctxt, initfunc, op)
	}
	// 0000000000000000 <local.dso_init>:
	// 0:	90000000 	adrp	x0, 0 <runtime.firstmoduledata>
	// 	0: R_AARCH64_ADR_PREL_PG_HI21	local.moduledata
	// 4:	91000000 	add	x0, x0, #0x0
	// 	4: R_AARCH64_ADD_ABS_LO12_NC	local.moduledata
	o(0x90000000)
	o(0x91000000)
	rel := ld.Addrel(initfunc)
	rel.Off = 0
	rel.Siz = 8
	rel.Sym = ld.Ctxt.Moduledata
	rel.Type = obj.R_ADDRARM64

	// 8:	14000000 	bl	0 <runtime.addmoduledata>
	// 	8: R_AARCH64_CALL26	runtime.addmoduledata
	o(0x14000000)
	rel = ld.Addrel(initfunc)
	rel.Off = 8
	rel.Siz = 4
	rel.Sym = ld.Linklookup(ld.Ctxt, "runtime.addmoduledata", 0)
	rel.Type = obj.R_CALLARM64 // Really should be R_AARCH64_JUMP26 but doesn't seem to make any difference

	ld.Ctxt.Textp = append(ld.Ctxt.Textp, initfunc)
	initarray_entry := ld.Linklookup(ld.Ctxt, "go.link.addmoduledatainit", 0)
	initarray_entry.Attr |= ld.AttrReachable
	initarray_entry.Attr |= ld.AttrLocal
	initarray_entry.Type = obj.SINITARR
	ld.Addaddr(ld.Ctxt, initarray_entry, initfunc)
}

func adddynrel(s *ld.LSym, r *ld.Reloc) {
	log.Fatalf("adddynrel not implemented")
}

func elfreloc1(r *ld.Reloc, sectoff int64) int {
	ld.Thearch.Vput(uint64(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	switch r.Type {
	default:
		return -1

	case obj.R_ADDR:
		switch r.Siz {
		case 4:
			ld.Thearch.Vput(ld.R_AARCH64_ABS32 | uint64(elfsym)<<32)
		case 8:
			ld.Thearch.Vput(ld.R_AARCH64_ABS64 | uint64(elfsym)<<32)
		default:
			return -1
		}

	case obj.R_ADDRARM64:
		// two relocations: R_AARCH64_ADR_PREL_PG_HI21 and R_AARCH64_ADD_ABS_LO12_NC
		ld.Thearch.Vput(ld.R_AARCH64_ADR_PREL_PG_HI21 | uint64(elfsym)<<32)
		ld.Thearch.Vput(uint64(r.Xadd))
		ld.Thearch.Vput(uint64(sectoff + 4))
		ld.Thearch.Vput(ld.R_AARCH64_ADD_ABS_LO12_NC | uint64(elfsym)<<32)

	case obj.R_ARM64_TLS_LE:
		ld.Thearch.Vput(ld.R_AARCH64_TLSLE_MOVW_TPREL_G0 | uint64(elfsym)<<32)

	case obj.R_ARM64_TLS_IE:
		ld.Thearch.Vput(ld.R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 | uint64(elfsym)<<32)
		ld.Thearch.Vput(uint64(r.Xadd))
		ld.Thearch.Vput(uint64(sectoff + 4))
		ld.Thearch.Vput(ld.R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC | uint64(elfsym)<<32)

	case obj.R_ARM64_GOTPCREL:
		ld.Thearch.Vput(ld.R_AARCH64_ADR_GOT_PAGE | uint64(elfsym)<<32)
		ld.Thearch.Vput(uint64(r.Xadd))
		ld.Thearch.Vput(uint64(sectoff + 4))
		ld.Thearch.Vput(ld.R_AARCH64_LD64_GOT_LO12_NC | uint64(elfsym)<<32)

	case obj.R_CALLARM64:
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
	var v uint32

	rs := r.Xsym

	// ld64 has a bug handling MACHO_ARM64_RELOC_UNSIGNED with !extern relocation.
	// see cmd/internal/ld/data.go for details. The workaround is that don't use !extern
	// UNSIGNED relocation at all.
	if rs.Type == obj.SHOSTOBJ || r.Type == obj.R_CALLARM64 || r.Type == obj.R_ADDRARM64 || r.Type == obj.R_ADDR {
		if rs.Dynid < 0 {
			ld.Diag("reloc %d to non-macho symbol %s type=%d", r.Type, rs.Name, rs.Type)
			return -1
		}

		v = uint32(rs.Dynid)
		v |= 1 << 27 // external relocation
	} else {
		v = uint32(rs.Sect.Extnum)
		if v == 0 {
			ld.Diag("reloc %d to symbol %s in non-macho section %s type=%d", r.Type, rs.Name, rs.Sect.Name, rs.Type)
			return -1
		}
	}

	switch r.Type {
	default:
		return -1

	case obj.R_ADDR:
		v |= ld.MACHO_ARM64_RELOC_UNSIGNED << 28

	case obj.R_CALLARM64:
		if r.Xadd != 0 {
			ld.Diag("ld64 doesn't allow BR26 reloc with non-zero addend: %s+%d", rs.Name, r.Xadd)
		}

		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_BRANCH26 << 28

	case obj.R_ADDRARM64:
		r.Siz = 4
		// Two relocation entries: MACHO_ARM64_RELOC_PAGEOFF12 MACHO_ARM64_RELOC_PAGE21
		// if r.Xadd is non-zero, add two MACHO_ARM64_RELOC_ADDEND.
		if r.Xadd != 0 {
			ld.Thearch.Lput(uint32(sectoff + 4))
			ld.Thearch.Lput((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(r.Xadd&0xffffff))
		}
		ld.Thearch.Lput(uint32(sectoff + 4))
		ld.Thearch.Lput(v | (ld.MACHO_ARM64_RELOC_PAGEOFF12 << 28) | (2 << 25))
		if r.Xadd != 0 {
			ld.Thearch.Lput(uint32(sectoff))
			ld.Thearch.Lput((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(r.Xadd&0xffffff))
		}
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_PAGE21 << 28
	}

	switch r.Siz {
	default:
		return -1

	case 1:
		v |= 0 << 25

	case 2:
		v |= 1 << 25

	case 4:
		v |= 2 << 25

	case 8:
		v |= 3 << 25
	}

	ld.Thearch.Lput(uint32(sectoff))
	ld.Thearch.Lput(v)
	return 0
}

func archreloc(r *ld.Reloc, s *ld.LSym, val *int64) int {
	if ld.Linkmode == ld.LinkExternal {
		switch r.Type {
		default:
			return -1

		case obj.R_ARM64_GOTPCREL:
			var o1, o2 uint32
			if ld.Ctxt.Arch.ByteOrder == binary.BigEndian {
				o1 = uint32(*val >> 32)
				o2 = uint32(*val)
			} else {
				o1 = uint32(*val)
				o2 = uint32(*val >> 32)
			}
			// Any relocation against a function symbol is redirected to
			// be against a local symbol instead (see putelfsym in
			// symtab.go) but unfortunately the system linker was buggy
			// when confronted with a R_AARCH64_ADR_GOT_PAGE relocation
			// against a local symbol until May 2015
			// (https://sourceware.org/bugzilla/show_bug.cgi?id=18270). So
			// we convert the adrp; ld64 + R_ARM64_GOTPCREL into adrp;
			// add + R_ADDRARM64.
			if !(r.Sym.Version != 0 || (r.Sym.Type&obj.SHIDDEN != 0) || r.Sym.Attr.Local()) && r.Sym.Type == obj.STEXT && ld.DynlinkingGo() {
				if o2&0xffc00000 != 0xf9400000 {
					ld.Ctxt.Diag("R_ARM64_GOTPCREL against unexpected instruction %x", o2)
				}
				o2 = 0x91000000 | (o2 & 0x000003ff)
				r.Type = obj.R_ADDRARM64
			}
			if ld.Ctxt.Arch.ByteOrder == binary.BigEndian {
				*val = int64(o1)<<32 | int64(o2)
			} else {
				*val = int64(o2)<<32 | int64(o1)
			}
			fallthrough

		case obj.R_ADDRARM64:
			r.Done = 0

			// set up addend for eventual relocation via outer symbol.
			rs := r.Sym
			r.Xadd = r.Add
			for rs.Outer != nil {
				r.Xadd += ld.Symaddr(rs) - ld.Symaddr(rs.Outer)
				rs = rs.Outer
			}

			if rs.Type != obj.SHOSTOBJ && rs.Type != obj.SDYNIMPORT && rs.Sect == nil {
				ld.Diag("missing section for %s", rs.Name)
			}
			r.Xsym = rs

			// Note: ld64 currently has a bug that any non-zero addend for BR26 relocation
			// will make the linking fail because it thinks the code is not PIC even though
			// the BR26 relocation should be fully resolved at link time.
			// That is the reason why the next if block is disabled. When the bug in ld64
			// is fixed, we can enable this block and also enable duff's device in cmd/7g.
			if false && ld.HEADTYPE == obj.Hdarwin {
				var o0, o1 uint32

				if ld.Ctxt.Arch.ByteOrder == binary.BigEndian {
					o0 = uint32(*val >> 32)
					o1 = uint32(*val)
				} else {
					o0 = uint32(*val)
					o1 = uint32(*val >> 32)
				}
				// Mach-O wants the addend to be encoded in the instruction
				// Note that although Mach-O supports ARM64_RELOC_ADDEND, it
				// can only encode 24-bit of signed addend, but the instructions
				// supports 33-bit of signed addend, so we always encode the
				// addend in place.
				o0 |= (uint32((r.Xadd>>12)&3) << 29) | (uint32((r.Xadd>>12>>2)&0x7ffff) << 5)
				o1 |= uint32(r.Xadd&0xfff) << 10
				r.Xadd = 0

				// when laid out, the instruction order must always be o1, o2.
				if ld.Ctxt.Arch.ByteOrder == binary.BigEndian {
					*val = int64(o0)<<32 | int64(o1)
				} else {
					*val = int64(o1)<<32 | int64(o0)
				}
			}

			return 0

		case obj.R_CALLARM64,
			obj.R_ARM64_TLS_LE,
			obj.R_ARM64_TLS_IE:
			r.Done = 0
			r.Xsym = r.Sym
			r.Xadd = r.Add
			return 0
		}
	}

	switch r.Type {
	case obj.R_CONST:
		*val = r.Add
		return 0

	case obj.R_GOTOFF:
		*val = ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ld.Linklookup(ld.Ctxt, ".got", 0))
		return 0

	case obj.R_ADDRARM64:
		t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
		if t >= 1<<32 || t < -1<<32 {
			ld.Diag("program too large, address relocation distance = %d", t)
		}

		var o0, o1 uint32

		if ld.Ctxt.Arch.ByteOrder == binary.BigEndian {
			o0 = uint32(*val >> 32)
			o1 = uint32(*val)
		} else {
			o0 = uint32(*val)
			o1 = uint32(*val >> 32)
		}

		o0 |= (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
		o1 |= uint32(t&0xfff) << 10

		// when laid out, the instruction order must always be o1, o2.
		if ld.Ctxt.Arch.ByteOrder == binary.BigEndian {
			*val = int64(o0)<<32 | int64(o1)
		} else {
			*val = int64(o1)<<32 | int64(o0)
		}
		return 0

	case obj.R_ARM64_TLS_LE:
		r.Done = 0
		if ld.HEADTYPE != obj.Hlinux {
			ld.Diag("TLS reloc on unsupported OS %s", ld.Headstr(int(ld.HEADTYPE)))
		}
		// The TCB is two pointers. This is not documented anywhere, but is
		// de facto part of the ABI.
		v := r.Sym.Value + int64(2*ld.SysArch.PtrSize)
		if v < 0 || v >= 32678 {
			ld.Diag("TLS offset out of range %d", v)
		}
		*val |= v << 5
		return 0

	case obj.R_CALLARM64:
		t := (ld.Symaddr(r.Sym) + r.Add) - (s.Value + int64(r.Off))
		if t >= 1<<27 || t < -1<<27 {
			ld.Diag("program too large, call relocation distance = %d", t)
		}
		*val |= (t >> 2) & 0x03ffffff
		return 0
	}

	return -1
}

func archrelocvariant(r *ld.Reloc, s *ld.LSym, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return -1
}

func asmb() {
	if ld.Debug['v'] != 0 {
		fmt.Fprintf(ld.Bso, "%5.2f asmb\n", obj.Cputime())
	}
	ld.Bso.Flush()

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
			fmt.Fprintf(ld.Bso, "%5.2f rodatblk\n", obj.Cputime())
		}
		ld.Bso.Flush()

		ld.Cseek(int64(ld.Segrodata.Fileoff))
		ld.Datblk(int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}

	if ld.Debug['v'] != 0 {
		fmt.Fprintf(ld.Bso, "%5.2f datblk\n", obj.Cputime())
	}
	ld.Bso.Flush()

	ld.Cseek(int64(ld.Segdata.Fileoff))
	ld.Datblk(int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	ld.Cseek(int64(ld.Segdwarf.Fileoff))
	ld.Dwarfblk(int64(ld.Segdwarf.Vaddr), int64(ld.Segdwarf.Filelen))

	machlink := uint32(0)
	if ld.HEADTYPE == obj.Hdarwin {
		machlink = uint32(ld.Domacholink())
	}

	/* output symbol table */
	ld.Symsize = 0

	ld.Lcsize = 0
	symo := uint32(0)
	if ld.Debug['s'] == 0 {
		// TODO: rationalize
		if ld.Debug['v'] != 0 {
			fmt.Fprintf(ld.Bso, "%5.2f sym\n", obj.Cputime())
		}
		ld.Bso.Flush()
		switch ld.HEADTYPE {
		default:
			if ld.Iself {
				symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(ld.INITRND)))
			}

		case obj.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)

		case obj.Hdarwin:
			symo = uint32(ld.Segdwarf.Fileoff + uint64(ld.Rnd(int64(ld.Segdwarf.Filelen), int64(ld.INITRND))) + uint64(machlink))
		}

		ld.Cseek(int64(symo))
		switch ld.HEADTYPE {
		default:
			if ld.Iself {
				if ld.Debug['v'] != 0 {
					fmt.Fprintf(ld.Bso, "%5.2f elfsym\n", obj.Cputime())
				}
				ld.Asmelfsym()
				ld.Cflush()
				ld.Cwrite(ld.Elfstrdat)

				if ld.Linkmode == ld.LinkExternal {
					ld.Elfemitreloc()
				}
			}

		case obj.Hplan9:
			ld.Asmplan9sym()
			ld.Cflush()

			sym := ld.Linklookup(ld.Ctxt, "pclntab", 0)
			if sym != nil {
				ld.Lcsize = int32(len(sym.P))
				for i := 0; int32(i) < ld.Lcsize; i++ {
					ld.Cput(sym.P[i])
				}

				ld.Cflush()
			}

		case obj.Hdarwin:
			if ld.Linkmode == ld.LinkExternal {
				ld.Machoemitreloc()
			}
		}
	}

	ld.Ctxt.Cursym = nil
	if ld.Debug['v'] != 0 {
		fmt.Fprintf(ld.Bso, "%5.2f header\n", obj.Cputime())
	}
	ld.Bso.Flush()
	ld.Cseek(0)
	switch ld.HEADTYPE {
	default:
	case obj.Hplan9: /* plan 9 */
		ld.Thearch.Lput(0x647)                      /* magic */
		ld.Thearch.Lput(uint32(ld.Segtext.Filelen)) /* sizes */
		ld.Thearch.Lput(uint32(ld.Segdata.Filelen))
		ld.Thearch.Lput(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ld.Thearch.Lput(uint32(ld.Symsize))      /* nsyms */
		ld.Thearch.Lput(uint32(ld.Entryvalue())) /* va of entry */
		ld.Thearch.Lput(0)
		ld.Thearch.Lput(uint32(ld.Lcsize))

	case obj.Hlinux,
		obj.Hfreebsd,
		obj.Hnetbsd,
		obj.Hopenbsd,
		obj.Hnacl:
		ld.Asmbelf(int64(symo))

	case obj.Hdarwin:
		ld.Asmbmacho()
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
