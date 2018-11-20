// Inferno utils/5l/asm.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/5l/asm.c
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
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/sym"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"log"
)

func gentext(ctxt *ld.Link) {
	if !ctxt.DynlinkingGo() {
		return
	}
	addmoduledata := ctxt.Syms.Lookup("runtime.addmoduledata", 0)
	if addmoduledata.Type == sym.STEXT && ctxt.BuildMode != ld.BuildModePlugin {
		// we're linking a module containing the runtime -> no need for
		// an init function
		return
	}
	addmoduledata.Attr |= sym.AttrReachable
	initfunc := ctxt.Syms.Lookup("go.link.addmoduledata", 0)
	initfunc.Type = sym.STEXT
	initfunc.Attr |= sym.AttrLocal
	initfunc.Attr |= sym.AttrReachable
	o := func(op uint32) {
		initfunc.AddUint32(ctxt.Arch, op)
	}
	// 0000000000000000 <local.dso_init>:
	// 0:	90000000 	adrp	x0, 0 <runtime.firstmoduledata>
	// 	0: R_AARCH64_ADR_PREL_PG_HI21	local.moduledata
	// 4:	91000000 	add	x0, x0, #0x0
	// 	4: R_AARCH64_ADD_ABS_LO12_NC	local.moduledata
	o(0x90000000)
	o(0x91000000)
	rel := initfunc.AddRel()
	rel.Off = 0
	rel.Siz = 8
	rel.Sym = ctxt.Moduledata
	rel.Type = objabi.R_ADDRARM64

	// 8:	14000000 	b	0 <runtime.addmoduledata>
	// 	8: R_AARCH64_CALL26	runtime.addmoduledata
	o(0x14000000)
	rel = initfunc.AddRel()
	rel.Off = 8
	rel.Siz = 4
	rel.Sym = ctxt.Syms.Lookup("runtime.addmoduledata", 0)
	rel.Type = objabi.R_CALLARM64 // Really should be R_AARCH64_JUMP26 but doesn't seem to make any difference

	if ctxt.BuildMode == ld.BuildModePlugin {
		ctxt.Textp = append(ctxt.Textp, addmoduledata)
	}
	ctxt.Textp = append(ctxt.Textp, initfunc)
	initarray_entry := ctxt.Syms.Lookup("go.link.addmoduledatainit", 0)
	initarray_entry.Attr |= sym.AttrReachable
	initarray_entry.Attr |= sym.AttrLocal
	initarray_entry.Type = sym.SINITARR
	initarray_entry.AddAddr(ctxt.Arch, initfunc)
}

// adddynrel implements just enough to support external linking to
// the system libc functions used by the runtime.
func adddynrel(ctxt *ld.Link, s *sym.Symbol, r *sym.Reloc) bool {
	targ := r.Sym

	switch r.Type {
	case objabi.R_CALL,
		objabi.R_PCREL,
		objabi.R_CALLARM64:
		if targ.Type != sym.SDYNIMPORT {
			// nothing to do, the relocation will be laid out in reloc
			return true
		}
		if ctxt.LinkMode == ld.LinkExternal {
			// External linker will do this relocation.
			return true
		}
	}
	log.Fatalf("adddynrel not implemented for relocation type %d (%s)", r.Type, sym.RelocName(ctxt.Arch, r.Type))
	return false
}

func elfreloc1(ctxt *ld.Link, r *sym.Reloc, sectoff int64) bool {
	ctxt.Out.Write64(uint64(sectoff))

	elfsym := r.Xsym.ElfsymForReloc()
	switch r.Type {
	default:
		return false
	case objabi.R_ADDR:
		switch r.Siz {
		case 4:
			ctxt.Out.Write64(uint64(elf.R_AARCH64_ABS32) | uint64(elfsym)<<32)
		case 8:
			ctxt.Out.Write64(uint64(elf.R_AARCH64_ABS64) | uint64(elfsym)<<32)
		default:
			return false
		}
	case objabi.R_ADDRARM64:
		// two relocations: R_AARCH64_ADR_PREL_PG_HI21 and R_AARCH64_ADD_ABS_LO12_NC
		ctxt.Out.Write64(uint64(elf.R_AARCH64_ADR_PREL_PG_HI21) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_AARCH64_ADD_ABS_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_ARM64_TLS_LE:
		ctxt.Out.Write64(uint64(elf.R_AARCH64_TLSLE_MOVW_TPREL_G0) | uint64(elfsym)<<32)
	case objabi.R_ARM64_TLS_IE:
		ctxt.Out.Write64(uint64(elf.R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_ARM64_GOTPCREL:
		ctxt.Out.Write64(uint64(elf.R_AARCH64_ADR_GOT_PAGE) | uint64(elfsym)<<32)
		ctxt.Out.Write64(uint64(r.Xadd))
		ctxt.Out.Write64(uint64(sectoff + 4))
		ctxt.Out.Write64(uint64(elf.R_AARCH64_LD64_GOT_LO12_NC) | uint64(elfsym)<<32)
	case objabi.R_CALLARM64:
		if r.Siz != 4 {
			return false
		}
		ctxt.Out.Write64(uint64(elf.R_AARCH64_CALL26) | uint64(elfsym)<<32)

	}
	ctxt.Out.Write64(uint64(r.Xadd))

	return true
}

func elfsetupplt(ctxt *ld.Link) {
	// TODO(aram)
	return
}

func machoreloc1(arch *sys.Arch, out *ld.OutBuf, s *sym.Symbol, r *sym.Reloc, sectoff int64) bool {
	var v uint32

	rs := r.Xsym

	if rs.Type == sym.SHOSTOBJ || r.Type == objabi.R_CALLARM64 || r.Type == objabi.R_ADDRARM64 {
		if rs.Dynid < 0 {
			ld.Errorf(s, "reloc %d (%s) to non-macho symbol %s type=%d (%s)", r.Type, sym.RelocName(arch, r.Type), rs.Name, rs.Type, rs.Type)
			return false
		}

		v = uint32(rs.Dynid)
		v |= 1 << 27 // external relocation
	} else {
		v = uint32(rs.Sect.Extnum)
		if v == 0 {
			ld.Errorf(s, "reloc %d (%s) to symbol %s in non-macho section %s type=%d (%s)", r.Type, sym.RelocName(arch, r.Type), rs.Name, rs.Sect.Name, rs.Type, rs.Type)
			return false
		}
	}

	switch r.Type {
	default:
		return false
	case objabi.R_ADDR:
		v |= ld.MACHO_ARM64_RELOC_UNSIGNED << 28
	case objabi.R_CALLARM64:
		if r.Xadd != 0 {
			ld.Errorf(s, "ld64 doesn't allow BR26 reloc with non-zero addend: %s+%d", rs.Name, r.Xadd)
		}

		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_BRANCH26 << 28
	case objabi.R_ADDRARM64:
		r.Siz = 4
		// Two relocation entries: MACHO_ARM64_RELOC_PAGEOFF12 MACHO_ARM64_RELOC_PAGE21
		// if r.Xadd is non-zero, add two MACHO_ARM64_RELOC_ADDEND.
		if r.Xadd != 0 {
			out.Write32(uint32(sectoff + 4))
			out.Write32((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(r.Xadd&0xffffff))
		}
		out.Write32(uint32(sectoff + 4))
		out.Write32(v | (ld.MACHO_ARM64_RELOC_PAGEOFF12 << 28) | (2 << 25))
		if r.Xadd != 0 {
			out.Write32(uint32(sectoff))
			out.Write32((ld.MACHO_ARM64_RELOC_ADDEND << 28) | (2 << 25) | uint32(r.Xadd&0xffffff))
		}
		v |= 1 << 24 // pc-relative bit
		v |= ld.MACHO_ARM64_RELOC_PAGE21 << 28
	}

	switch r.Siz {
	default:
		return false
	case 1:
		v |= 0 << 25
	case 2:
		v |= 1 << 25
	case 4:
		v |= 2 << 25
	case 8:
		v |= 3 << 25
	}

	out.Write32(uint32(sectoff))
	out.Write32(v)
	return true
}

func archreloc(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, val int64) (int64, bool) {
	if ctxt.LinkMode == ld.LinkExternal {
		switch r.Type {
		default:
			return val, false
		case objabi.R_ARM64_GOTPCREL:
			var o1, o2 uint32
			if ctxt.Arch.ByteOrder == binary.BigEndian {
				o1 = uint32(val >> 32)
				o2 = uint32(val)
			} else {
				o1 = uint32(val)
				o2 = uint32(val >> 32)
			}
			// Any relocation against a function symbol is redirected to
			// be against a local symbol instead (see putelfsym in
			// symtab.go) but unfortunately the system linker was buggy
			// when confronted with a R_AARCH64_ADR_GOT_PAGE relocation
			// against a local symbol until May 2015
			// (https://sourceware.org/bugzilla/show_bug.cgi?id=18270). So
			// we convert the adrp; ld64 + R_ARM64_GOTPCREL into adrp;
			// add + R_ADDRARM64.
			if !(r.Sym.IsFileLocal() || r.Sym.Attr.VisibilityHidden() || r.Sym.Attr.Local()) && r.Sym.Type == sym.STEXT && ctxt.DynlinkingGo() {
				if o2&0xffc00000 != 0xf9400000 {
					ld.Errorf(s, "R_ARM64_GOTPCREL against unexpected instruction %x", o2)
				}
				o2 = 0x91000000 | (o2 & 0x000003ff)
				r.Type = objabi.R_ADDRARM64
			}
			if ctxt.Arch.ByteOrder == binary.BigEndian {
				val = int64(o1)<<32 | int64(o2)
			} else {
				val = int64(o2)<<32 | int64(o1)
			}
			fallthrough
		case objabi.R_ADDRARM64:
			r.Done = false

			// set up addend for eventual relocation via outer symbol.
			rs := r.Sym
			r.Xadd = r.Add
			for rs.Outer != nil {
				r.Xadd += ld.Symaddr(rs) - ld.Symaddr(rs.Outer)
				rs = rs.Outer
			}

			if rs.Type != sym.SHOSTOBJ && rs.Type != sym.SDYNIMPORT && rs.Sect == nil {
				ld.Errorf(s, "missing section for %s", rs.Name)
			}
			r.Xsym = rs

			// Note: ld64 currently has a bug that any non-zero addend for BR26 relocation
			// will make the linking fail because it thinks the code is not PIC even though
			// the BR26 relocation should be fully resolved at link time.
			// That is the reason why the next if block is disabled. When the bug in ld64
			// is fixed, we can enable this block and also enable duff's device in cmd/7g.
			if false && ctxt.HeadType == objabi.Hdarwin {
				var o0, o1 uint32

				if ctxt.Arch.ByteOrder == binary.BigEndian {
					o0 = uint32(val >> 32)
					o1 = uint32(val)
				} else {
					o0 = uint32(val)
					o1 = uint32(val >> 32)
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
				if ctxt.Arch.ByteOrder == binary.BigEndian {
					val = int64(o0)<<32 | int64(o1)
				} else {
					val = int64(o1)<<32 | int64(o0)
				}
			}

			return val, true
		case objabi.R_CALLARM64,
			objabi.R_ARM64_TLS_LE,
			objabi.R_ARM64_TLS_IE:
			r.Done = false
			r.Xsym = r.Sym
			r.Xadd = r.Add
			return val, true
		}
	}

	switch r.Type {
	case objabi.R_CONST:
		return r.Add, true
	case objabi.R_GOTOFF:
		return ld.Symaddr(r.Sym) + r.Add - ld.Symaddr(ctxt.Syms.Lookup(".got", 0)), true
	case objabi.R_ADDRARM64:
		t := ld.Symaddr(r.Sym) + r.Add - ((s.Value + int64(r.Off)) &^ 0xfff)
		if t >= 1<<32 || t < -1<<32 {
			ld.Errorf(s, "program too large, address relocation distance = %d", t)
		}

		var o0, o1 uint32

		if ctxt.Arch.ByteOrder == binary.BigEndian {
			o0 = uint32(val >> 32)
			o1 = uint32(val)
		} else {
			o0 = uint32(val)
			o1 = uint32(val >> 32)
		}

		o0 |= (uint32((t>>12)&3) << 29) | (uint32((t>>12>>2)&0x7ffff) << 5)
		o1 |= uint32(t&0xfff) << 10

		// when laid out, the instruction order must always be o1, o2.
		if ctxt.Arch.ByteOrder == binary.BigEndian {
			return int64(o0)<<32 | int64(o1), true
		}
		return int64(o1)<<32 | int64(o0), true
	case objabi.R_ARM64_TLS_LE:
		r.Done = false
		if ctxt.HeadType != objabi.Hlinux {
			ld.Errorf(s, "TLS reloc on unsupported OS %v", ctxt.HeadType)
		}
		// The TCB is two pointers. This is not documented anywhere, but is
		// de facto part of the ABI.
		v := r.Sym.Value + int64(2*ctxt.Arch.PtrSize)
		if v < 0 || v >= 32678 {
			ld.Errorf(s, "TLS offset out of range %d", v)
		}
		return val | (v << 5), true
	case objabi.R_CALLARM64:
		t := (ld.Symaddr(r.Sym) + r.Add) - (s.Value + int64(r.Off))
		if t >= 1<<27 || t < -1<<27 {
			ld.Errorf(s, "program too large, call relocation distance = %d", t)
		}
		return val | ((t >> 2) & 0x03ffffff), true
	}

	return val, false
}

func archrelocvariant(ctxt *ld.Link, r *sym.Reloc, s *sym.Symbol, t int64) int64 {
	log.Fatalf("unexpected relocation variant")
	return -1
}

func asmb(ctxt *ld.Link) {
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f asmb\n", ld.Cputime())
	}

	if ctxt.IsELF {
		ld.Asmbelfsetup()
	}

	sect := ld.Segtext.Sections[0]
	ctxt.Out.SeekSet(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
	ld.Codeblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
	for _, sect = range ld.Segtext.Sections[1:] {
		ctxt.Out.SeekSet(int64(sect.Vaddr - ld.Segtext.Vaddr + ld.Segtext.Fileoff))
		ld.Datblk(ctxt, int64(sect.Vaddr), int64(sect.Length))
	}

	if ld.Segrodata.Filelen > 0 {
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f rodatblk\n", ld.Cputime())
		}
		ctxt.Out.SeekSet(int64(ld.Segrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrodata.Vaddr), int64(ld.Segrodata.Filelen))
	}
	if ld.Segrelrodata.Filelen > 0 {
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f relrodatblk\n", ld.Cputime())
		}
		ctxt.Out.SeekSet(int64(ld.Segrelrodata.Fileoff))
		ld.Datblk(ctxt, int64(ld.Segrelrodata.Vaddr), int64(ld.Segrelrodata.Filelen))
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f datblk\n", ld.Cputime())
	}

	ctxt.Out.SeekSet(int64(ld.Segdata.Fileoff))
	ld.Datblk(ctxt, int64(ld.Segdata.Vaddr), int64(ld.Segdata.Filelen))

	ctxt.Out.SeekSet(int64(ld.Segdwarf.Fileoff))
	ld.Dwarfblk(ctxt, int64(ld.Segdwarf.Vaddr), int64(ld.Segdwarf.Filelen))

	machlink := uint32(0)
	if ctxt.HeadType == objabi.Hdarwin {
		machlink = uint32(ld.Domacholink(ctxt))
	}

	/* output symbol table */
	ld.Symsize = 0

	ld.Lcsize = 0
	symo := uint32(0)
	if !*ld.FlagS {
		// TODO: rationalize
		if ctxt.Debugvlog != 0 {
			ctxt.Logf("%5.2f sym\n", ld.Cputime())
		}
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
				symo = uint32(ld.Segdwarf.Fileoff + ld.Segdwarf.Filelen)
				symo = uint32(ld.Rnd(int64(symo), int64(*ld.FlagRound)))
			}

		case objabi.Hplan9:
			symo = uint32(ld.Segdata.Fileoff + ld.Segdata.Filelen)

		case objabi.Hdarwin:
			symo = uint32(ld.Segdwarf.Fileoff + uint64(ld.Rnd(int64(ld.Segdwarf.Filelen), int64(*ld.FlagRound))) + uint64(machlink))
		}

		ctxt.Out.SeekSet(int64(symo))
		switch ctxt.HeadType {
		default:
			if ctxt.IsELF {
				if ctxt.Debugvlog != 0 {
					ctxt.Logf("%5.2f elfsym\n", ld.Cputime())
				}
				ld.Asmelfsym(ctxt)
				ctxt.Out.Flush()
				ctxt.Out.Write(ld.Elfstrdat)

				if ctxt.LinkMode == ld.LinkExternal {
					ld.Elfemitreloc(ctxt)
				}
			}

		case objabi.Hplan9:
			ld.Asmplan9sym(ctxt)
			ctxt.Out.Flush()

			sym := ctxt.Syms.Lookup("pclntab", 0)
			if sym != nil {
				ld.Lcsize = int32(len(sym.P))
				ctxt.Out.Write(sym.P)
				ctxt.Out.Flush()
			}

		case objabi.Hdarwin:
			if ctxt.LinkMode == ld.LinkExternal {
				ld.Machoemitreloc(ctxt)
			}
		}
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f header\n", ld.Cputime())
	}
	ctxt.Out.SeekSet(0)
	switch ctxt.HeadType {
	default:
	case objabi.Hplan9: /* plan 9 */
		ctxt.Out.Write32(0x647)                      /* magic */
		ctxt.Out.Write32(uint32(ld.Segtext.Filelen)) /* sizes */
		ctxt.Out.Write32(uint32(ld.Segdata.Filelen))
		ctxt.Out.Write32(uint32(ld.Segdata.Length - ld.Segdata.Filelen))
		ctxt.Out.Write32(uint32(ld.Symsize))          /* nsyms */
		ctxt.Out.Write32(uint32(ld.Entryvalue(ctxt))) /* va of entry */
		ctxt.Out.Write32(0)
		ctxt.Out.Write32(uint32(ld.Lcsize))

	case objabi.Hlinux,
		objabi.Hfreebsd,
		objabi.Hnetbsd,
		objabi.Hopenbsd,
		objabi.Hnacl:
		ld.Asmbelf(ctxt, int64(symo))

	case objabi.Hdarwin:
		ld.Asmbmacho(ctxt)
	}

	ctxt.Out.Flush()
	if *ld.FlagC {
		fmt.Printf("textsize=%d\n", ld.Segtext.Filelen)
		fmt.Printf("datsize=%d\n", ld.Segdata.Filelen)
		fmt.Printf("bsssize=%d\n", ld.Segdata.Length-ld.Segdata.Filelen)
		fmt.Printf("symsize=%d\n", ld.Symsize)
		fmt.Printf("lcsize=%d\n", ld.Lcsize)
		fmt.Printf("total=%d\n", ld.Segtext.Filelen+ld.Segdata.Length+uint64(ld.Symsize)+uint64(ld.Lcsize))
	}
}
