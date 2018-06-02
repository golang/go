// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/obj.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/span.c
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
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// isRuntimeDepPkg returns whether pkg is the runtime package or its dependency
func isRuntimeDepPkg(pkg string) bool {
	switch pkg {
	case "runtime",
		"sync/atomic",      // runtime may call to sync/atomic, due to go:linkname
		"internal/bytealg", // for IndexByte
		"internal/cpu":     // for cpu features
		return true
	}
	return strings.HasPrefix(pkg, "runtime/internal/") && !strings.HasSuffix(pkg, "_test")
}

// Estimate the max size needed to hold any new trampolines created for this function. This
// is used to determine when the section can be split if it becomes too large, to ensure that
// the trampolines are in the same section as the function that uses them.
func maxSizeTrampolinesPPC64(s *sym.Symbol, isTramp bool) uint64 {
	// If thearch.Trampoline is nil, then trampoline support is not available on this arch.
	// A trampoline does not need any dependent trampolines.
	if thearch.Trampoline == nil || isTramp {
		return 0
	}

	n := uint64(0)
	for ri := range s.R {
		r := &s.R[ri]
		if r.Type.IsDirectJump() {
			n++
		}
	}
	// Trampolines in ppc64 are 4 instructions.
	return n * 16
}

// detect too-far jumps in function s, and add trampolines if necessary
// ARM, PPC64 & PPC64LE support trampoline insertion for internal and external linking
// On PPC64 & PPC64LE the text sections might be split but will still insert trampolines
// where necessary.
func trampoline(ctxt *Link, s *sym.Symbol) {
	if thearch.Trampoline == nil {
		return // no need or no support of trampolines on this arch
	}

	for ri := range s.R {
		r := &s.R[ri]
		if !r.Type.IsDirectJump() {
			continue
		}
		if Symaddr(r.Sym) == 0 && r.Sym.Type != sym.SDYNIMPORT {
			if r.Sym.File != s.File {
				if !isRuntimeDepPkg(s.File) || !isRuntimeDepPkg(r.Sym.File) {
					ctxt.ErrorUnresolved(s, r)
				}
				// runtime and its dependent packages may call to each other.
				// they are fine, as they will be laid down together.
			}
			continue
		}

		thearch.Trampoline(ctxt, r, s)
	}

}

// resolve relocations in s.
func relocsym(ctxt *Link, s *sym.Symbol) {
	for ri := int32(0); ri < int32(len(s.R)); ri++ {
		r := &s.R[ri]
		if r.Done {
			// Relocation already processed by an earlier phase.
			continue
		}
		r.Done = true
		off := r.Off
		siz := int32(r.Siz)
		if off < 0 || off+siz > int32(len(s.P)) {
			rname := ""
			if r.Sym != nil {
				rname = r.Sym.Name
			}
			Errorf(s, "invalid relocation %s: %d+%d not in [%d,%d)", rname, off, siz, 0, len(s.P))
			continue
		}

		if r.Sym != nil && ((r.Sym.Type == sym.Sxxx && !r.Sym.Attr.VisibilityHidden()) || r.Sym.Type == sym.SXREF) {
			// When putting the runtime but not main into a shared library
			// these symbols are undefined and that's OK.
			if ctxt.BuildMode == BuildModeShared {
				if r.Sym.Name == "main.main" || r.Sym.Name == "main.init" {
					r.Sym.Type = sym.SDYNIMPORT
				} else if strings.HasPrefix(r.Sym.Name, "go.info.") {
					// Skip go.info symbols. They are only needed to communicate
					// DWARF info between the compiler and linker.
					continue
				}
			} else {
				ctxt.ErrorUnresolved(s, r)
				continue
			}
		}

		if r.Type >= 256 {
			continue
		}
		if r.Siz == 0 { // informational relocation - no work to do
			continue
		}
		if r.Type == objabi.R_DWARFFILEREF {
			// These should have been processed previously during
			// line table writing.
			Errorf(s, "orphan R_DWARFFILEREF reloc to %v", r.Sym.Name)
			continue
		}

		// We need to be able to reference dynimport symbols when linking against
		// shared libraries, and Solaris and Darwin need it always
		if ctxt.HeadType != objabi.Hsolaris && ctxt.HeadType != objabi.Hdarwin && r.Sym != nil && r.Sym.Type == sym.SDYNIMPORT && !ctxt.DynlinkingGo() && !r.Sym.Attr.SubSymbol() {
			if !(ctxt.Arch.Family == sys.PPC64 && ctxt.LinkMode == LinkExternal && r.Sym.Name == ".TOC.") {
				Errorf(s, "unhandled relocation for %s (type %d (%s) rtype %d (%s))", r.Sym.Name, r.Sym.Type, r.Sym.Type, r.Type, sym.RelocName(ctxt.Arch, r.Type))
			}
		}
		if r.Sym != nil && r.Sym.Type != sym.STLSBSS && r.Type != objabi.R_WEAKADDROFF && !r.Sym.Attr.Reachable() {
			Errorf(s, "unreachable sym in relocation: %s", r.Sym.Name)
		}

		// TODO(mundaym): remove this special case - see issue 14218.
		if ctxt.Arch.Family == sys.S390X {
			switch r.Type {
			case objabi.R_PCRELDBL:
				r.Type = objabi.R_PCREL
				r.Variant = sym.RV_390_DBL
			case objabi.R_CALL:
				r.Variant = sym.RV_390_DBL
			}
		}

		var o int64
		switch r.Type {
		default:
			switch siz {
			default:
				Errorf(s, "bad reloc size %#x for %s", uint32(siz), r.Sym.Name)
			case 1:
				o = int64(s.P[off])
			case 2:
				o = int64(ctxt.Arch.ByteOrder.Uint16(s.P[off:]))
			case 4:
				o = int64(ctxt.Arch.ByteOrder.Uint32(s.P[off:]))
			case 8:
				o = int64(ctxt.Arch.ByteOrder.Uint64(s.P[off:]))
			}
			if !thearch.Archreloc(ctxt, r, s, &o) {
				Errorf(s, "unknown reloc to %v: %d (%s)", r.Sym.Name, r.Type, sym.RelocName(ctxt.Arch, r.Type))
			}
		case objabi.R_TLS_LE:
			isAndroidX86 := objabi.GOOS == "android" && (ctxt.Arch.InFamily(sys.AMD64, sys.I386))

			if ctxt.LinkMode == LinkExternal && ctxt.IsELF && !isAndroidX86 {
				r.Done = false
				if r.Sym == nil {
					r.Sym = ctxt.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if ctxt.Arch.Family != sys.AMD64 {
					o = r.Add
				}
				break
			}

			if ctxt.IsELF && ctxt.Arch.Family == sys.ARM {
				// On ELF ARM, the thread pointer is 8 bytes before
				// the start of the thread-local data block, so add 8
				// to the actual TLS offset (r->sym->value).
				// This 8 seems to be a fundamental constant of
				// ELF on ARM (or maybe Glibc on ARM); it is not
				// related to the fact that our own TLS storage happens
				// to take up 8 bytes.
				o = 8 + r.Sym.Value
			} else if ctxt.IsELF || ctxt.HeadType == objabi.Hplan9 || ctxt.HeadType == objabi.Hdarwin || isAndroidX86 {
				o = int64(ctxt.Tlsoffset) + r.Add
			} else if ctxt.HeadType == objabi.Hwindows {
				o = r.Add
			} else {
				log.Fatalf("unexpected R_TLS_LE relocation for %v", ctxt.HeadType)
			}
		case objabi.R_TLS_IE:
			isAndroidX86 := objabi.GOOS == "android" && (ctxt.Arch.InFamily(sys.AMD64, sys.I386))

			if ctxt.LinkMode == LinkExternal && ctxt.IsELF && !isAndroidX86 {
				r.Done = false
				if r.Sym == nil {
					r.Sym = ctxt.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if ctxt.Arch.Family != sys.AMD64 {
					o = r.Add
				}
				break
			}
			if ctxt.BuildMode == BuildModePIE && ctxt.IsELF {
				// We are linking the final executable, so we
				// can optimize any TLS IE relocation to LE.
				if thearch.TLSIEtoLE == nil {
					log.Fatalf("internal linking of TLS IE not supported on %v", ctxt.Arch.Family)
				}
				thearch.TLSIEtoLE(s, int(off), int(r.Siz))
				o = int64(ctxt.Tlsoffset)
				// TODO: o += r.Add when ctxt.Arch.Family != sys.AMD64?
				// Why do we treat r.Add differently on AMD64?
				// Is the external linker using Xadd at all?
			} else {
				log.Fatalf("cannot handle R_TLS_IE (sym %s) when linking internally", s.Name)
			}
		case objabi.R_ADDR:
			if ctxt.LinkMode == LinkExternal && r.Sym.Type != sym.SCONST {
				r.Done = false

				// set up addend for eventual relocation via outer symbol.
				rs := r.Sym

				r.Xadd = r.Add
				for rs.Outer != nil {
					r.Xadd += Symaddr(rs) - Symaddr(rs.Outer)
					rs = rs.Outer
				}

				if rs.Type != sym.SHOSTOBJ && rs.Type != sym.SDYNIMPORT && rs.Sect == nil {
					Errorf(s, "missing section for relocation target %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if ctxt.IsELF {
					if ctxt.Arch.Family == sys.AMD64 {
						o = 0
					}
				} else if ctxt.HeadType == objabi.Hdarwin {
					if rs.Type != sym.SHOSTOBJ {
						o += Symaddr(rs)
					}
				} else if ctxt.HeadType == objabi.Hwindows {
					// nothing to do
				} else {
					Errorf(s, "unhandled pcrel relocation to %s on %v", rs.Name, ctxt.HeadType)
				}

				break
			}

			o = Symaddr(r.Sym) + r.Add

			// On amd64, 4-byte offsets will be sign-extended, so it is impossible to
			// access more than 2GB of static data; fail at link time is better than
			// fail at runtime. See https://golang.org/issue/7980.
			// Instead of special casing only amd64, we treat this as an error on all
			// 64-bit architectures so as to be future-proof.
			if int32(o) < 0 && ctxt.Arch.PtrSize > 4 && siz == 4 {
				Errorf(s, "non-pc-relative relocation address for %s is too big: %#x (%#x + %#x)", r.Sym.Name, uint64(o), Symaddr(r.Sym), r.Add)
				errorexit()
			}
		case objabi.R_DWARFSECREF:
			if r.Sym.Sect == nil {
				Errorf(s, "missing DWARF section for relocation target %s", r.Sym.Name)
			}

			if ctxt.LinkMode == LinkExternal {
				r.Done = false

				// On most platforms, the external linker needs to adjust DWARF references
				// as it combines DWARF sections. However, on Darwin, dsymutil does the
				// DWARF linking, and it understands how to follow section offsets.
				// Leaving in the relocation records confuses it (see
				// https://golang.org/issue/22068) so drop them for Darwin.
				if ctxt.HeadType == objabi.Hdarwin {
					r.Done = true
				}

				// PE code emits IMAGE_REL_I386_SECREL and IMAGE_REL_AMD64_SECREL
				// for R_DWARFSECREF relocations, while R_ADDR is replaced with
				// IMAGE_REL_I386_DIR32, IMAGE_REL_AMD64_ADDR64 and IMAGE_REL_AMD64_ADDR32.
				// Do not replace R_DWARFSECREF with R_ADDR for windows -
				// let PE code emit correct relocations.
				if ctxt.HeadType != objabi.Hwindows {
					r.Type = objabi.R_ADDR
				}

				r.Xsym = ctxt.Syms.ROLookup(r.Sym.Sect.Name, 0)
				r.Xadd = r.Add + Symaddr(r.Sym) - int64(r.Sym.Sect.Vaddr)

				o = r.Xadd
				if ctxt.IsELF && ctxt.Arch.Family == sys.AMD64 {
					o = 0
				}
				break
			}
			o = Symaddr(r.Sym) + r.Add - int64(r.Sym.Sect.Vaddr)
		case objabi.R_WEAKADDROFF:
			if !r.Sym.Attr.Reachable() {
				continue
			}
			fallthrough
		case objabi.R_ADDROFF:
			// The method offset tables using this relocation expect the offset to be relative
			// to the start of the first text section, even if there are multiple.
			if r.Sym.Sect.Name == ".text" {
				o = Symaddr(r.Sym) - int64(Segtext.Sections[0].Vaddr) + r.Add
			} else {
				o = Symaddr(r.Sym) - int64(r.Sym.Sect.Vaddr) + r.Add
			}

		case objabi.R_ADDRCUOFF:
			// debug_range and debug_loc elements use this relocation type to get an
			// offset from the start of the compile unit.
			o = Symaddr(r.Sym) + r.Add - Symaddr(r.Sym.Lib.Textp[0])

			// r->sym can be null when CALL $(constant) is transformed from absolute PC to relative PC call.
		case objabi.R_GOTPCREL:
			if ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin && r.Sym != nil && r.Sym.Type != sym.SCONST {
				r.Done = false
				r.Xadd = r.Add
				r.Xadd -= int64(r.Siz) // relative to address after the relocated chunk
				r.Xsym = r.Sym

				o = r.Xadd
				o += int64(r.Siz)
				break
			}
			fallthrough
		case objabi.R_CALL, objabi.R_PCREL:
			if ctxt.LinkMode == LinkExternal && r.Sym != nil && r.Sym.Type != sym.SCONST && (r.Sym.Sect != s.Sect || r.Type == objabi.R_GOTPCREL) {
				r.Done = false

				// set up addend for eventual relocation via outer symbol.
				rs := r.Sym

				r.Xadd = r.Add
				for rs.Outer != nil {
					r.Xadd += Symaddr(rs) - Symaddr(rs.Outer)
					rs = rs.Outer
				}

				r.Xadd -= int64(r.Siz) // relative to address after the relocated chunk
				if rs.Type != sym.SHOSTOBJ && rs.Type != sym.SDYNIMPORT && rs.Sect == nil {
					Errorf(s, "missing section for relocation target %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if ctxt.IsELF {
					if ctxt.Arch.Family == sys.AMD64 {
						o = 0
					}
				} else if ctxt.HeadType == objabi.Hdarwin {
					if r.Type == objabi.R_CALL {
						if ctxt.LinkMode == LinkExternal && rs.Type == sym.SDYNIMPORT {
							switch ctxt.Arch.Family {
							case sys.AMD64:
								// AMD64 dynamic relocations are relative to the end of the relocation.
								o += int64(r.Siz)
							case sys.I386:
								// I386 dynamic relocations are relative to the start of the section.
								o -= int64(r.Off)                         // offset in symbol
								o -= int64(s.Value - int64(s.Sect.Vaddr)) // offset of symbol in section
							}
						} else {
							if rs.Type != sym.SHOSTOBJ {
								o += int64(uint64(Symaddr(rs)) - rs.Sect.Vaddr)
							}
							o -= int64(r.Off) // relative to section offset, not symbol
						}
					} else if ctxt.Arch.Family == sys.ARM {
						// see ../arm/asm.go:/machoreloc1
						o += Symaddr(rs) - s.Value - int64(r.Off)
					} else {
						o += int64(r.Siz)
					}
				} else if ctxt.HeadType == objabi.Hwindows && ctxt.Arch.Family == sys.AMD64 { // only amd64 needs PCREL
					// PE/COFF's PC32 relocation uses the address after the relocated
					// bytes as the base. Compensate by skewing the addend.
					o += int64(r.Siz)
				} else {
					Errorf(s, "unhandled pcrel relocation to %s on %v", rs.Name, ctxt.HeadType)
				}

				break
			}

			o = 0
			if r.Sym != nil {
				o += Symaddr(r.Sym)
			}

			o += r.Add - (s.Value + int64(r.Off) + int64(r.Siz))
		case objabi.R_SIZE:
			o = r.Sym.Size + r.Add
		}

		if r.Variant != sym.RV_NONE {
			o = thearch.Archrelocvariant(ctxt, r, s, o)
		}

		if false {
			nam := "<nil>"
			var addr int64
			if r.Sym != nil {
				nam = r.Sym.Name
				addr = Symaddr(r.Sym)
			}
			xnam := "<nil>"
			if r.Xsym != nil {
				xnam = r.Xsym.Name
			}
			fmt.Printf("relocate %s %#x (%#x+%#x, size %d) => %s %#x +%#x (xsym: %s +%#x) [type %d (%s)/%d, %x]\n", s.Name, s.Value+int64(off), s.Value, r.Off, r.Siz, nam, addr, r.Add, xnam, r.Xadd, r.Type, sym.RelocName(ctxt.Arch, r.Type), r.Variant, o)
		}
		switch siz {
		default:
			Errorf(s, "bad reloc size %#x for %s", uint32(siz), r.Sym.Name)
			fallthrough

			// TODO(rsc): Remove.
		case 1:
			s.P[off] = byte(int8(o))
		case 2:
			if o != int64(int16(o)) {
				Errorf(s, "relocation address for %s is too big: %#x", r.Sym.Name, o)
			}
			i16 := int16(o)
			ctxt.Arch.ByteOrder.PutUint16(s.P[off:], uint16(i16))
		case 4:
			if r.Type == objabi.R_PCREL || r.Type == objabi.R_CALL {
				if o != int64(int32(o)) {
					Errorf(s, "pc-relative relocation address for %s is too big: %#x", r.Sym.Name, o)
				}
			} else {
				if o != int64(int32(o)) && o != int64(uint32(o)) {
					Errorf(s, "non-pc-relative relocation address for %s is too big: %#x", r.Sym.Name, uint64(o))
				}
			}

			fl := int32(o)
			ctxt.Arch.ByteOrder.PutUint32(s.P[off:], uint32(fl))
		case 8:
			ctxt.Arch.ByteOrder.PutUint64(s.P[off:], uint64(o))
		}
	}
}

func (ctxt *Link) reloc() {
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f reloc\n", Cputime())
	}

	for _, s := range ctxt.Textp {
		relocsym(ctxt, s)
	}
	for _, s := range datap {
		relocsym(ctxt, s)
	}
	for _, s := range dwarfp {
		relocsym(ctxt, s)
	}
}

func windynrelocsym(ctxt *Link, s *sym.Symbol) {
	rel := ctxt.Syms.Lookup(".rel", 0)
	if s == rel {
		return
	}
	for ri := range s.R {
		r := &s.R[ri]
		targ := r.Sym
		if targ == nil {
			continue
		}
		if !targ.Attr.Reachable() {
			if r.Type == objabi.R_WEAKADDROFF {
				continue
			}
			Errorf(s, "dynamic relocation to unreachable symbol %s", targ.Name)
		}
		if r.Sym.Plt == -2 && r.Sym.Got != -2 { // make dynimport JMP table for PE object files.
			targ.Plt = int32(rel.Size)
			r.Sym = rel
			r.Add = int64(targ.Plt)

			// jmp *addr
			if ctxt.Arch.Family == sys.I386 {
				rel.AddUint8(0xff)
				rel.AddUint8(0x25)
				rel.AddAddr(ctxt.Arch, targ)
				rel.AddUint8(0x90)
				rel.AddUint8(0x90)
			} else {
				rel.AddUint8(0xff)
				rel.AddUint8(0x24)
				rel.AddUint8(0x25)
				rel.AddAddrPlus4(targ, 0)
				rel.AddUint8(0x90)
			}
		} else if r.Sym.Plt >= 0 {
			r.Sym = rel
			r.Add = int64(targ.Plt)
		}
	}
}

func dynrelocsym(ctxt *Link, s *sym.Symbol) {
	if ctxt.HeadType == objabi.Hwindows {
		if ctxt.LinkMode == LinkInternal {
			windynrelocsym(ctxt, s)
		}
		return
	}

	for ri := range s.R {
		r := &s.R[ri]
		if ctxt.BuildMode == BuildModePIE && ctxt.LinkMode == LinkInternal {
			// It's expected that some relocations will be done
			// later by relocsym (R_TLS_LE, R_ADDROFF), so
			// don't worry if Adddynrel returns false.
			thearch.Adddynrel(ctxt, s, r)
			continue
		}
		if r.Sym != nil && r.Sym.Type == sym.SDYNIMPORT || r.Type >= 256 {
			if r.Sym != nil && !r.Sym.Attr.Reachable() {
				Errorf(s, "dynamic relocation to unreachable symbol %s", r.Sym.Name)
			}
			if !thearch.Adddynrel(ctxt, s, r) {
				Errorf(s, "unsupported dynamic relocation for symbol %s (type=%d (%s) stype=%d (%s))", r.Sym.Name, r.Type, sym.RelocName(ctxt.Arch, r.Type), r.Sym.Type, r.Sym.Type)
			}
		}
	}
}

func dynreloc(ctxt *Link, data *[sym.SXREF][]*sym.Symbol) {
	// -d suppresses dynamic loader format, so we may as well not
	// compute these sections or mark their symbols as reachable.
	if *FlagD && ctxt.HeadType != objabi.Hwindows {
		return
	}
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f dynreloc\n", Cputime())
	}

	for _, s := range ctxt.Textp {
		dynrelocsym(ctxt, s)
	}
	for _, syms := range data {
		for _, s := range syms {
			dynrelocsym(ctxt, s)
		}
	}
	if ctxt.IsELF {
		elfdynhash(ctxt)
	}
}

func Codeblk(ctxt *Link, addr int64, size int64) {
	CodeblkPad(ctxt, addr, size, zeros[:])
}
func CodeblkPad(ctxt *Link, addr int64, size int64, pad []byte) {
	if *flagA {
		ctxt.Logf("codeblk [%#x,%#x) at offset %#x\n", addr, addr+size, ctxt.Out.Offset())
	}

	blk(ctxt, ctxt.Textp, addr, size, pad)

	/* again for printing */
	if !*flagA {
		return
	}

	syms := ctxt.Textp
	for i, s := range syms {
		if !s.Attr.Reachable() {
			continue
		}
		if s.Value >= addr {
			syms = syms[i:]
			break
		}
	}

	eaddr := addr + size
	for _, s := range syms {
		if !s.Attr.Reachable() {
			continue
		}
		if s.Value >= eaddr {
			break
		}

		if addr < s.Value {
			ctxt.Logf("%-20s %.8x|", "_", uint64(addr))
			for ; addr < s.Value; addr++ {
				ctxt.Logf(" %.2x", 0)
			}
			ctxt.Logf("\n")
		}

		ctxt.Logf("%.6x\t%-20s\n", uint64(addr), s.Name)
		q := s.P

		for len(q) >= 16 {
			ctxt.Logf("%.6x\t% x\n", uint64(addr), q[:16])
			addr += 16
			q = q[16:]
		}

		if len(q) > 0 {
			ctxt.Logf("%.6x\t% x\n", uint64(addr), q)
			addr += int64(len(q))
		}
	}

	if addr < eaddr {
		ctxt.Logf("%-20s %.8x|", "_", uint64(addr))
		for ; addr < eaddr; addr++ {
			ctxt.Logf(" %.2x", 0)
		}
	}
}

func blk(ctxt *Link, syms []*sym.Symbol, addr, size int64, pad []byte) {
	for i, s := range syms {
		if !s.Attr.SubSymbol() && s.Value >= addr {
			syms = syms[i:]
			break
		}
	}

	eaddr := addr + size
	for _, s := range syms {
		if s.Attr.SubSymbol() {
			continue
		}
		if s.Value >= eaddr {
			break
		}
		if s.Value < addr {
			Errorf(s, "phase error: addr=%#x but sym=%#x type=%d", addr, s.Value, s.Type)
			errorexit()
		}
		if addr < s.Value {
			ctxt.Out.WriteStringPad("", int(s.Value-addr), pad)
			addr = s.Value
		}
		ctxt.Out.Write(s.P)
		addr += int64(len(s.P))
		if addr < s.Value+s.Size {
			ctxt.Out.WriteStringPad("", int(s.Value+s.Size-addr), pad)
			addr = s.Value + s.Size
		}
		if addr != s.Value+s.Size {
			Errorf(s, "phase error: addr=%#x value+size=%#x", addr, s.Value+s.Size)
			errorexit()
		}
		if s.Value+s.Size >= eaddr {
			break
		}
	}

	if addr < eaddr {
		ctxt.Out.WriteStringPad("", int(eaddr-addr), pad)
	}
	ctxt.Out.Flush()
}

func Datblk(ctxt *Link, addr int64, size int64) {
	if *flagA {
		ctxt.Logf("datblk [%#x,%#x) at offset %#x\n", addr, addr+size, ctxt.Out.Offset())
	}

	blk(ctxt, datap, addr, size, zeros[:])

	/* again for printing */
	if !*flagA {
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
			ctxt.Logf("\t%.8x| 00 ...\n", uint64(addr))
			addr = sym.Value
		}

		ctxt.Logf("%s\n\t%.8x|", sym.Name, uint64(addr))
		for i, b := range sym.P {
			if i > 0 && i%16 == 0 {
				ctxt.Logf("\n\t%.8x|", uint64(addr)+uint64(i))
			}
			ctxt.Logf(" %.2x", b)
		}

		addr += int64(len(sym.P))
		for ; addr < sym.Value+sym.Size; addr++ {
			ctxt.Logf(" %.2x", 0)
		}
		ctxt.Logf("\n")

		if ctxt.LinkMode != LinkExternal {
			continue
		}
		for _, r := range sym.R {
			rsname := ""
			if r.Sym != nil {
				rsname = r.Sym.Name
			}
			typ := "?"
			switch r.Type {
			case objabi.R_ADDR:
				typ = "addr"
			case objabi.R_PCREL:
				typ = "pcrel"
			case objabi.R_CALL:
				typ = "call"
			}
			ctxt.Logf("\treloc %.8x/%d %s %s+%#x [%#x]\n", uint(sym.Value+int64(r.Off)), r.Siz, typ, rsname, r.Add, r.Sym.Value+r.Add)
		}
	}

	if addr < eaddr {
		ctxt.Logf("\t%.8x| 00 ...\n", uint(addr))
	}
	ctxt.Logf("\t%.8x|\n", uint(eaddr))
}

func Dwarfblk(ctxt *Link, addr int64, size int64) {
	if *flagA {
		ctxt.Logf("dwarfblk [%#x,%#x) at offset %#x\n", addr, addr+size, ctxt.Out.Offset())
	}

	blk(ctxt, dwarfp, addr, size, zeros[:])
}

var zeros [512]byte

var (
	strdata  = make(map[string]string)
	strnames []string
)

func addstrdata1(ctxt *Link, arg string) {
	eq := strings.Index(arg, "=")
	dot := strings.LastIndex(arg[:eq+1], ".")
	if eq < 0 || dot < 0 {
		Exitf("-X flag requires argument of the form importpath.name=value")
	}
	pkg := arg[:dot]
	if ctxt.BuildMode == BuildModePlugin && pkg == "main" {
		pkg = *flagPluginPath
	}
	pkg = objabi.PathToPrefix(pkg)
	name := pkg + arg[dot:eq]
	value := arg[eq+1:]
	if _, ok := strdata[name]; !ok {
		strnames = append(strnames, name)
	}
	strdata[name] = value
}

func addstrdata(ctxt *Link, name, value string) {
	s := ctxt.Syms.ROLookup(name, 0)
	if s == nil || s.Gotype == nil {
		// Not defined in the loaded packages.
		return
	}
	if s.Gotype.Name != "type.string" {
		Errorf(s, "cannot set with -X: not a var of type string (%s)", s.Gotype.Name)
		return
	}
	if s.Type == sym.SBSS {
		s.Type = sym.SDATA
	}

	p := fmt.Sprintf("%s.str", s.Name)
	sp := ctxt.Syms.Lookup(p, 0)

	Addstring(sp, value)
	sp.Type = sym.SRODATA

	s.Size = 0
	s.P = s.P[:0]
	s.R = s.R[:0]
	reachable := s.Attr.Reachable()
	s.AddAddr(ctxt.Arch, sp)
	s.AddUint(ctxt.Arch, uint64(len(value)))

	// addstring, addaddr, etc., mark the symbols as reachable.
	// In this case that is not necessarily true, so stick to what
	// we know before entering this function.
	s.Attr.Set(sym.AttrReachable, reachable)

	sp.Attr.Set(sym.AttrReachable, reachable)
}

func (ctxt *Link) dostrdata() {
	for _, name := range strnames {
		addstrdata(ctxt, name, strdata[name])
	}
}

func Addstring(s *sym.Symbol, str string) int64 {
	if s.Type == 0 {
		s.Type = sym.SNOPTRDATA
	}
	s.Attr |= sym.AttrReachable
	r := s.Size
	if s.Name == ".shstrtab" {
		elfsetstring(s, str, int(r))
	}
	s.P = append(s.P, str...)
	s.P = append(s.P, 0)
	s.Size = int64(len(s.P))
	return r
}

// addgostring adds str, as a Go string value, to s. symname is the name of the
// symbol used to define the string data and must be unique per linked object.
func addgostring(ctxt *Link, s *sym.Symbol, symname, str string) {
	sdata := ctxt.Syms.Lookup(symname, 0)
	if sdata.Type != sym.Sxxx {
		Errorf(s, "duplicate symname in addgostring: %s", symname)
	}
	sdata.Attr |= sym.AttrReachable
	sdata.Attr |= sym.AttrLocal
	sdata.Type = sym.SRODATA
	sdata.Size = int64(len(str))
	sdata.P = []byte(str)
	s.AddAddr(ctxt.Arch, sdata)
	s.AddUint(ctxt.Arch, uint64(len(str)))
}

func addinitarrdata(ctxt *Link, s *sym.Symbol) {
	p := s.Name + ".ptr"
	sp := ctxt.Syms.Lookup(p, 0)
	sp.Type = sym.SINITARR
	sp.Size = 0
	sp.Attr |= sym.AttrDuplicateOK
	sp.AddAddr(ctxt.Arch, s)
}

func dosymtype(ctxt *Link) {
	switch ctxt.BuildMode {
	case BuildModeCArchive, BuildModeCShared:
		for _, s := range ctxt.Syms.Allsym {
			// Create a new entry in the .init_array section that points to the
			// library initializer function.
			if s.Name == *flagEntrySymbol {
				addinitarrdata(ctxt, s)
			}
		}
	}
}

// symalign returns the required alignment for the given symbol s.
func symalign(s *sym.Symbol) int32 {
	min := int32(thearch.Minalign)
	if s.Align >= min {
		return s.Align
	} else if s.Align != 0 {
		return min
	}
	if strings.HasPrefix(s.Name, "go.string.") || strings.HasPrefix(s.Name, "type..namedata.") {
		// String data is just bytes.
		// If we align it, we waste a lot of space to padding.
		return min
	}
	align := int32(thearch.Maxalign)
	for int64(align) > s.Size && align > min {
		align >>= 1
	}
	return align
}

func aligndatsize(datsize int64, s *sym.Symbol) int64 {
	return Rnd(datsize, int64(symalign(s)))
}

const debugGCProg = false

type GCProg struct {
	ctxt *Link
	sym  *sym.Symbol
	w    gcprog.Writer
}

func (p *GCProg) Init(ctxt *Link, name string) {
	p.ctxt = ctxt
	p.sym = ctxt.Syms.Lookup(name, 0)
	p.w.Init(p.writeByte(ctxt))
	if debugGCProg {
		fmt.Fprintf(os.Stderr, "ld: start GCProg %s\n", name)
		p.w.Debug(os.Stderr)
	}
}

func (p *GCProg) writeByte(ctxt *Link) func(x byte) {
	return func(x byte) {
		p.sym.AddUint8(x)
	}
}

func (p *GCProg) End(size int64) {
	p.w.ZeroUntil(size / int64(p.ctxt.Arch.PtrSize))
	p.w.End()
	if debugGCProg {
		fmt.Fprintf(os.Stderr, "ld: end GCProg\n")
	}
}

func (p *GCProg) AddSym(s *sym.Symbol) {
	typ := s.Gotype
	// Things without pointers should be in sym.SNOPTRDATA or sym.SNOPTRBSS;
	// everything we see should have pointers and should therefore have a type.
	if typ == nil {
		switch s.Name {
		case "runtime.data", "runtime.edata", "runtime.bss", "runtime.ebss":
			// Ignore special symbols that are sometimes laid out
			// as real symbols. See comment about dyld on darwin in
			// the address function.
			return
		}
		Errorf(s, "missing Go type information for global symbol: size %d", s.Size)
		return
	}

	ptrsize := int64(p.ctxt.Arch.PtrSize)
	nptr := decodetypePtrdata(p.ctxt.Arch, typ) / ptrsize

	if debugGCProg {
		fmt.Fprintf(os.Stderr, "gcprog sym: %s at %d (ptr=%d+%d)\n", s.Name, s.Value, s.Value/ptrsize, nptr)
	}

	if decodetypeUsegcprog(p.ctxt.Arch, typ) == 0 {
		// Copy pointers from mask into program.
		mask := decodetypeGcmask(p.ctxt, typ)
		for i := int64(0); i < nptr; i++ {
			if (mask[i/8]>>uint(i%8))&1 != 0 {
				p.w.Ptr(s.Value/ptrsize + i)
			}
		}
		return
	}

	// Copy program.
	prog := decodetypeGcprog(p.ctxt, typ)
	p.w.ZeroUntil(s.Value / ptrsize)
	p.w.Append(prog[4:], nptr)
}

// dataSortKey is used to sort a slice of data symbol *sym.Symbol pointers.
// The sort keys are kept inline to improve cache behavior while sorting.
type dataSortKey struct {
	size int64
	name string
	sym  *sym.Symbol
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

// cutoff is the maximum data section size permitted by the linker
// (see issue #9862).
const cutoff = 2e9 // 2 GB (or so; looks better in errors than 2^31)

func checkdatsize(ctxt *Link, datsize int64, symn sym.SymKind) {
	if datsize > cutoff {
		Errorf(nil, "too much data in section %v (over %v bytes)", symn, cutoff)
	}
}

// datap is a collection of reachable data symbols in address order.
// Generated by dodata.
var datap []*sym.Symbol

func (ctxt *Link) dodata() {
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f dodata\n", Cputime())
	}

	if ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin {
		// The values in moduledata are filled out by relocations
		// pointing to the addresses of these special symbols.
		// Typically these symbols have no size and are not laid
		// out with their matching section.
		//
		// However on darwin, dyld will find the special symbol
		// in the first loaded module, even though it is local.
		//
		// (An hypothesis, formed without looking in the dyld sources:
		// these special symbols have no size, so their address
		// matches a real symbol. The dynamic linker assumes we
		// want the normal symbol with the same address and finds
		// it in the other module.)
		//
		// To work around this we lay out the symbls whose
		// addresses are vital for multi-module programs to work
		// as normal symbols, and give them a little size.
		bss := ctxt.Syms.Lookup("runtime.bss", 0)
		bss.Size = 8
		bss.Attr.Set(sym.AttrSpecial, false)

		ctxt.Syms.Lookup("runtime.ebss", 0).Attr.Set(sym.AttrSpecial, false)

		data := ctxt.Syms.Lookup("runtime.data", 0)
		data.Size = 8
		data.Attr.Set(sym.AttrSpecial, false)

		ctxt.Syms.Lookup("runtime.edata", 0).Attr.Set(sym.AttrSpecial, false)

		types := ctxt.Syms.Lookup("runtime.types", 0)
		types.Type = sym.STYPE
		types.Size = 8
		types.Attr.Set(sym.AttrSpecial, false)

		etypes := ctxt.Syms.Lookup("runtime.etypes", 0)
		etypes.Type = sym.SFUNCTAB
		etypes.Attr.Set(sym.AttrSpecial, false)
	}

	// Collect data symbols by type into data.
	var data [sym.SXREF][]*sym.Symbol
	for _, s := range ctxt.Syms.Allsym {
		if !s.Attr.Reachable() || s.Attr.Special() || s.Attr.SubSymbol() {
			continue
		}
		if s.Type <= sym.STEXT || s.Type >= sym.SXREF {
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
	if ctxt.HeadType == objabi.Hdarwin {
		machosymorder(ctxt)
	}
	dynreloc(ctxt, &data)

	if ctxt.UseRelro() {
		// "read only" data with relocations needs to go in its own section
		// when building a shared library. We do this by boosting objects of
		// type SXXX with relocations to type SXXXRELRO.
		for _, symnro := range sym.ReadOnly {
			symnrelro := sym.RelROMap[symnro]

			ro := []*sym.Symbol{}
			relro := data[symnrelro]

			for _, s := range data[symnro] {
				isRelro := len(s.R) > 0
				switch s.Type {
				case sym.STYPE, sym.STYPERELRO, sym.SGOFUNCRELRO:
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
					Errorf(s, "inconsistent types for symbol and its Outer %s (%v != %v)",
						s.Outer.Name, s.Type, s.Outer.Type)
				}
			}

			data[symnro] = ro
			data[symnrelro] = relro
		}
	}

	// Sort symbols.
	var dataMaxAlign [sym.SXREF]int32
	var wg sync.WaitGroup
	for symn := range data {
		symn := sym.SymKind(symn)
		wg.Add(1)
		go func() {
			data[symn], dataMaxAlign[symn] = dodataSect(ctxt, symn, data[symn])
			wg.Done()
		}()
	}
	wg.Wait()

	// Allocate sections.
	// Data is processed before segtext, because we need
	// to see all symbols in the .data and .bss sections in order
	// to generate garbage collection information.
	datsize := int64(0)

	// Writable data sections that do not need any specialized handling.
	writable := []sym.SymKind{
		sym.SELFSECT,
		sym.SMACHO,
		sym.SMACHOGOT,
		sym.SWINDOWS,
	}
	for _, symn := range writable {
		for _, s := range data[symn] {
			sect := addsection(ctxt.Arch, &Segdata, s.Name, 06)
			sect.Align = symalign(s)
			datsize = Rnd(datsize, int64(sect.Align))
			sect.Vaddr = uint64(datsize)
			s.Sect = sect
			s.Type = sym.SDATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
			sect.Length = uint64(datsize) - sect.Vaddr
		}
		checkdatsize(ctxt, datsize, symn)
	}

	// .got (and .toc on ppc64)
	if len(data[sym.SELFGOT]) > 0 {
		sect := addsection(ctxt.Arch, &Segdata, ".got", 06)
		sect.Align = dataMaxAlign[sym.SELFGOT]
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for _, s := range data[sym.SELFGOT] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Type = sym.SDATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)

			// Resolve .TOC. symbol for this object file (ppc64)
			toc := ctxt.Syms.ROLookup(".TOC.", int(s.Version))
			if toc != nil {
				toc.Sect = sect
				toc.Outer = s
				toc.Sub = s.Sub
				s.Sub = toc

				toc.Value = 0x8000
			}

			datsize += s.Size
		}
		checkdatsize(ctxt, datsize, sym.SELFGOT)
		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* pointer-free data */
	sect := addsection(ctxt.Arch, &Segdata, ".noptrdata", 06)
	sect.Align = dataMaxAlign[sym.SNOPTRDATA]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.noptrdata", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.enoptrdata", 0).Sect = sect
	for _, s := range data[sym.SNOPTRDATA] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = sym.SDATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, sym.SNOPTRDATA)
	sect.Length = uint64(datsize) - sect.Vaddr

	hasinitarr := ctxt.linkShared

	/* shared library initializer */
	switch ctxt.BuildMode {
	case BuildModeCArchive, BuildModeCShared, BuildModeShared, BuildModePlugin:
		hasinitarr = true
	}
	if hasinitarr {
		sect := addsection(ctxt.Arch, &Segdata, ".init_array", 06)
		sect.Align = dataMaxAlign[sym.SINITARR]
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for _, s := range data[sym.SINITARR] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
		}
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(ctxt, datsize, sym.SINITARR)
	}

	/* data */
	sect = addsection(ctxt.Arch, &Segdata, ".data", 06)
	sect.Align = dataMaxAlign[sym.SDATA]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.data", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.edata", 0).Sect = sect
	var gc GCProg
	gc.Init(ctxt, "runtime.gcdata")
	for _, s := range data[sym.SDATA] {
		s.Sect = sect
		s.Type = sym.SDATA
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		gc.AddSym(s)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, sym.SDATA)
	sect.Length = uint64(datsize) - sect.Vaddr
	gc.End(int64(sect.Length))

	/* bss */
	sect = addsection(ctxt.Arch, &Segdata, ".bss", 06)
	sect.Align = dataMaxAlign[sym.SBSS]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.bss", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.ebss", 0).Sect = sect
	gc = GCProg{}
	gc.Init(ctxt, "runtime.gcbss")
	for _, s := range data[sym.SBSS] {
		s.Sect = sect
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		gc.AddSym(s)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, sym.SBSS)
	sect.Length = uint64(datsize) - sect.Vaddr
	gc.End(int64(sect.Length))

	/* pointer-free bss */
	sect = addsection(ctxt.Arch, &Segdata, ".noptrbss", 06)
	sect.Align = dataMaxAlign[sym.SNOPTRBSS]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.noptrbss", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.enoptrbss", 0).Sect = sect
	for _, s := range data[sym.SNOPTRBSS] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}

	sect.Length = uint64(datsize) - sect.Vaddr
	ctxt.Syms.Lookup("runtime.end", 0).Sect = sect
	checkdatsize(ctxt, datsize, sym.SNOPTRBSS)

	if len(data[sym.STLSBSS]) > 0 {
		var sect *sym.Section
		if ctxt.IsELF && (ctxt.LinkMode == LinkExternal || !*FlagD) {
			sect = addsection(ctxt.Arch, &Segdata, ".tbss", 06)
			sect.Align = int32(ctxt.Arch.PtrSize)
			sect.Vaddr = 0
		}
		datsize = 0

		for _, s := range data[sym.STLSBSS] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = datsize
			datsize += s.Size
		}
		checkdatsize(ctxt, datsize, sym.STLSBSS)

		if sect != nil {
			sect.Length = uint64(datsize)
		}
	}

	/*
	 * We finished data, begin read-only data.
	 * Not all systems support a separate read-only non-executable data section.
	 * ELF and Windows PE systems do.
	 * OS X and Plan 9 do not.
	 * And if we're using external linking mode, the point is moot,
	 * since it's not our decision; that code expects the sections in
	 * segtext.
	 */
	var segro *sym.Segment
	if ctxt.IsELF && ctxt.LinkMode == LinkInternal {
		segro = &Segrodata
	} else if ctxt.HeadType == objabi.Hwindows {
		segro = &Segrodata
	} else {
		segro = &Segtext
	}

	datsize = 0

	/* read-only executable ELF, Mach-O sections */
	if len(data[sym.STEXT]) != 0 {
		Errorf(nil, "dodata found an sym.STEXT symbol: %s", data[sym.STEXT][0].Name)
	}
	for _, s := range data[sym.SELFRXSECT] {
		sect := addsection(ctxt.Arch, &Segtext, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = sym.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(ctxt, datsize, sym.SELFRXSECT)
	}

	/* read-only data */
	sect = addsection(ctxt.Arch, segro, ".rodata", 04)

	sect.Vaddr = 0
	ctxt.Syms.Lookup("runtime.rodata", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.erodata", 0).Sect = sect
	if !ctxt.UseRelro() {
		ctxt.Syms.Lookup("runtime.types", 0).Sect = sect
		ctxt.Syms.Lookup("runtime.etypes", 0).Sect = sect
	}
	for _, symn := range sym.ReadOnly {
		align := dataMaxAlign[symn]
		if sect.Align < align {
			sect.Align = align
		}
	}
	datsize = Rnd(datsize, int64(sect.Align))
	for _, symn := range sym.ReadOnly {
		for _, s := range data[symn] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Type = sym.SRODATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
		}
		checkdatsize(ctxt, datsize, symn)
	}
	sect.Length = uint64(datsize) - sect.Vaddr

	/* read-only ELF, Mach-O sections */
	for _, s := range data[sym.SELFROSECT] {
		sect = addsection(ctxt.Arch, segro, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = sym.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(ctxt, datsize, sym.SELFROSECT)

	for _, s := range data[sym.SMACHOPLT] {
		sect = addsection(ctxt.Arch, segro, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = sym.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(ctxt, datsize, sym.SMACHOPLT)

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
	addrelrosection := func(suffix string) *sym.Section {
		return addsection(ctxt.Arch, segro, suffix, 04)
	}

	if ctxt.UseRelro() {
		addrelrosection = func(suffix string) *sym.Section {
			seg := &Segrelrodata
			if ctxt.LinkMode == LinkExternal {
				// Using a separate segment with an external
				// linker results in some programs moving
				// their data sections unexpectedly, which
				// corrupts the moduledata. So we use the
				// rodata segment and let the external linker
				// sort out a rel.ro segment.
				seg = &Segrodata
			}
			return addsection(ctxt.Arch, seg, ".data.rel.ro"+suffix, 06)
		}
		/* data only written by relocations */
		sect = addrelrosection("")

		sect.Vaddr = 0
		ctxt.Syms.Lookup("runtime.types", 0).Sect = sect
		ctxt.Syms.Lookup("runtime.etypes", 0).Sect = sect
		for _, symnro := range sym.ReadOnly {
			symn := sym.RelROMap[symnro]
			align := dataMaxAlign[symn]
			if sect.Align < align {
				sect.Align = align
			}
		}
		datsize = Rnd(datsize, int64(sect.Align))
		for _, symnro := range sym.ReadOnly {
			symn := sym.RelROMap[symnro]
			for _, s := range data[symn] {
				datsize = aligndatsize(datsize, s)
				if s.Outer != nil && s.Outer.Sect != nil && s.Outer.Sect != sect {
					Errorf(s, "s.Outer (%s) in different section from s, %s != %s", s.Outer.Name, s.Outer.Sect.Name, sect.Name)
				}
				s.Sect = sect
				s.Type = sym.SRODATA
				s.Value = int64(uint64(datsize) - sect.Vaddr)
				datsize += s.Size
			}
			checkdatsize(ctxt, datsize, symn)
		}

		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* typelink */
	sect = addrelrosection(".typelink")
	sect.Align = dataMaxAlign[sym.STYPELINK]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	typelink := ctxt.Syms.Lookup("runtime.typelink", 0)
	typelink.Sect = sect
	typelink.Type = sym.SRODATA
	datsize += typelink.Size
	checkdatsize(ctxt, datsize, sym.STYPELINK)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* itablink */
	sect = addrelrosection(".itablink")
	sect.Align = dataMaxAlign[sym.SITABLINK]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.itablink", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.eitablink", 0).Sect = sect
	for _, s := range data[sym.SITABLINK] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = sym.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, sym.SITABLINK)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* gosymtab */
	sect = addrelrosection(".gosymtab")
	sect.Align = dataMaxAlign[sym.SSYMTAB]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.symtab", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.esymtab", 0).Sect = sect
	for _, s := range data[sym.SSYMTAB] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = sym.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, sym.SSYMTAB)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* gopclntab */
	sect = addrelrosection(".gopclntab")
	sect.Align = dataMaxAlign[sym.SPCLNTAB]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.pclntab", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.epclntab", 0).Sect = sect
	for _, s := range data[sym.SPCLNTAB] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = sym.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, sym.SRODATA)
	sect.Length = uint64(datsize) - sect.Vaddr

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if datsize != int64(uint32(datsize)) {
		Errorf(nil, "read-only data segment too large: %d", datsize)
	}

	for symn := sym.SELFRXSECT; symn < sym.SXREF; symn++ {
		datap = append(datap, data[symn]...)
	}

	dwarfgeneratedebugsyms(ctxt)

	var i int
	for ; i < len(dwarfp); i++ {
		s := dwarfp[i]
		if s.Type != sym.SDWARFSECT {
			break
		}

		sect = addsection(ctxt.Arch, &Segdwarf, s.Name, 04)
		sect.Align = 1
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = sym.SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(ctxt, datsize, sym.SDWARFSECT)

	for i < len(dwarfp) {
		curType := dwarfp[i].Type
		var sect *sym.Section
		switch curType {
		case sym.SDWARFINFO:
			sect = addsection(ctxt.Arch, &Segdwarf, ".debug_info", 04)
		case sym.SDWARFRANGE:
			sect = addsection(ctxt.Arch, &Segdwarf, ".debug_ranges", 04)
		case sym.SDWARFLOC:
			sect = addsection(ctxt.Arch, &Segdwarf, ".debug_loc", 04)
		default:
			Errorf(dwarfp[i], "unknown DWARF section %v", curType)
		}

		sect.Align = 1
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for ; i < len(dwarfp); i++ {
			s := dwarfp[i]
			if s.Type != curType {
				break
			}
			s.Sect = sect
			s.Type = sym.SRODATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			s.Attr |= sym.AttrLocal
			datsize += s.Size
		}
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(ctxt, datsize, curType)
	}

	/* number the sections */
	n := int32(1)

	for _, sect := range Segtext.Sections {
		sect.Extnum = int16(n)
		n++
	}
	for _, sect := range Segrodata.Sections {
		sect.Extnum = int16(n)
		n++
	}
	for _, sect := range Segrelrodata.Sections {
		sect.Extnum = int16(n)
		n++
	}
	for _, sect := range Segdata.Sections {
		sect.Extnum = int16(n)
		n++
	}
	for _, sect := range Segdwarf.Sections {
		sect.Extnum = int16(n)
		n++
	}
}

func dodataSect(ctxt *Link, symn sym.SymKind, syms []*sym.Symbol) (result []*sym.Symbol, maxAlign int32) {
	if ctxt.HeadType == objabi.Hdarwin {
		// Some symbols may no longer belong in syms
		// due to movement in machosymorder.
		newSyms := make([]*sym.Symbol, 0, len(syms))
		for _, s := range syms {
			if s.Type == symn {
				newSyms = append(newSyms, s)
			}
		}
		syms = newSyms
	}

	var head, tail *sym.Symbol
	symsSort := make([]dataSortKey, 0, len(syms))
	for _, s := range syms {
		if s.Attr.OnList() {
			log.Fatalf("symbol %s listed multiple times", s.Name)
		}
		s.Attr |= sym.AttrOnList
		switch {
		case s.Size < int64(len(s.P)):
			Errorf(s, "initialize bounds (%d < %d)", s.Size, len(s.P))
		case s.Size < 0:
			Errorf(s, "negative size (%d bytes)", s.Size)
		case s.Size > cutoff:
			Errorf(s, "symbol too large (%d bytes)", s.Size)
		}

		// If the usually-special section-marker symbols are being laid
		// out as regular symbols, put them either at the beginning or
		// end of their section.
		if ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin {
			switch s.Name {
			case "runtime.text", "runtime.bss", "runtime.data", "runtime.types":
				head = s
				continue
			case "runtime.etext", "runtime.ebss", "runtime.edata", "runtime.etypes":
				tail = s
				continue
			}
		}

		key := dataSortKey{
			size: s.Size,
			name: s.Name,
			sym:  s,
		}

		switch s.Type {
		case sym.SELFGOT:
			// For ppc64, we want to interleave the .got and .toc sections
			// from input files. Both are type sym.SELFGOT, so in that case
			// we skip size comparison and fall through to the name
			// comparison (conveniently, .got sorts before .toc).
			key.size = 0
		}

		symsSort = append(symsSort, key)
	}

	sort.Sort(bySizeAndName(symsSort))

	off := 0
	if head != nil {
		syms[0] = head
		off++
	}
	for i, symSort := range symsSort {
		syms[i+off] = symSort.sym
		align := symalign(symSort.sym)
		if maxAlign < align {
			maxAlign = align
		}
	}
	if tail != nil {
		syms[len(syms)-1] = tail
	}

	if ctxt.IsELF && symn == sym.SELFROSECT {
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

			// Make sure alignment doesn't introduce a gap.
			// Setting the alignment explicitly prevents
			// symalign from basing it on the size and
			// getting it wrong.
			rel.Align = int32(ctxt.Arch.RegSize)
			plt.Align = int32(ctxt.Arch.RegSize)
		}
	}

	return syms, maxAlign
}

// Add buildid to beginning of text segment, on non-ELF systems.
// Non-ELF binary formats are not always flexible enough to
// give us a place to put the Go build ID. On those systems, we put it
// at the very beginning of the text segment.
// This ``header'' is read by cmd/go.
func (ctxt *Link) textbuildid() {
	if ctxt.IsELF || ctxt.BuildMode == BuildModePlugin || *flagBuildid == "" {
		return
	}

	s := ctxt.Syms.Lookup("go.buildid", 0)
	s.Attr |= sym.AttrReachable
	// The \xff is invalid UTF-8, meant to make it less likely
	// to find one of these accidentally.
	data := "\xff Go build ID: " + strconv.Quote(*flagBuildid) + "\n \xff"
	s.Type = sym.STEXT
	s.P = []byte(data)
	s.Size = int64(len(s.P))

	ctxt.Textp = append(ctxt.Textp, nil)
	copy(ctxt.Textp[1:], ctxt.Textp)
	ctxt.Textp[0] = s
}

// assign addresses to text
func (ctxt *Link) textaddress() {
	addsection(ctxt.Arch, &Segtext, ".text", 05)

	// Assign PCs in text segment.
	// Could parallelize, by assigning to text
	// and then letting threads copy down, but probably not worth it.
	sect := Segtext.Sections[0]

	sect.Align = int32(Funcalign)

	text := ctxt.Syms.Lookup("runtime.text", 0)
	text.Sect = sect

	if ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin {
		etext := ctxt.Syms.Lookup("runtime.etext", 0)
		etext.Sect = sect

		ctxt.Textp = append(ctxt.Textp, etext, nil)
		copy(ctxt.Textp[1:], ctxt.Textp)
		ctxt.Textp[0] = text
	}

	va := uint64(*FlagTextAddr)
	n := 1
	sect.Vaddr = va
	ntramps := 0
	for _, s := range ctxt.Textp {
		sect, n, va = assignAddress(ctxt, sect, n, s, va, false)

		trampoline(ctxt, s) // resolve jumps, may add trampolines if jump too far

		// lay down trampolines after each function
		for ; ntramps < len(ctxt.tramps); ntramps++ {
			tramp := ctxt.tramps[ntramps]
			sect, n, va = assignAddress(ctxt, sect, n, tramp, va, true)
		}
	}

	sect.Length = va - sect.Vaddr
	ctxt.Syms.Lookup("runtime.etext", 0).Sect = sect

	// merge tramps into Textp, keeping Textp in address order
	if ntramps != 0 {
		newtextp := make([]*sym.Symbol, 0, len(ctxt.Textp)+ntramps)
		i := 0
		for _, s := range ctxt.Textp {
			for ; i < ntramps && ctxt.tramps[i].Value < s.Value; i++ {
				newtextp = append(newtextp, ctxt.tramps[i])
			}
			newtextp = append(newtextp, s)
		}
		newtextp = append(newtextp, ctxt.tramps[i:ntramps]...)

		ctxt.Textp = newtextp
	}
}

// assigns address for a text symbol, returns (possibly new) section, its number, and the address
// Note: once we have trampoline insertion support for external linking, this function
// will not need to create new text sections, and so no need to return sect and n.
func assignAddress(ctxt *Link, sect *sym.Section, n int, s *sym.Symbol, va uint64, isTramp bool) (*sym.Section, int, uint64) {
	if thearch.AssignAddress != nil {
		return thearch.AssignAddress(ctxt, sect, n, s, va, isTramp)
	}

	s.Sect = sect
	if s.Attr.SubSymbol() {
		return sect, n, va
	}
	if s.Align != 0 {
		va = uint64(Rnd(int64(va), int64(s.Align)))
	} else {
		va = uint64(Rnd(int64(va), int64(Funcalign)))
	}
	s.Value = 0
	for sub := s; sub != nil; sub = sub.Sub {
		sub.Value += int64(va)
	}

	funcsize := uint64(MINFUNC) // spacing required for findfunctab
	if s.Size > MINFUNC {
		funcsize = uint64(s.Size)
	}

	// On ppc64x a text section should not be larger than 2^26 bytes due to the size of
	// call target offset field in the bl instruction.  Splitting into smaller text
	// sections smaller than this limit allows the GNU linker to modify the long calls
	// appropriately.  The limit allows for the space needed for tables inserted by the linker.

	// If this function doesn't fit in the current text section, then create a new one.

	// Only break at outermost syms.

	if ctxt.Arch.InFamily(sys.PPC64) && s.Outer == nil && ctxt.IsELF && ctxt.LinkMode == LinkExternal && va-sect.Vaddr+funcsize+maxSizeTrampolinesPPC64(s, isTramp) > 0x1c00000 {
		// Set the length for the previous text section
		sect.Length = va - sect.Vaddr

		// Create new section, set the starting Vaddr
		sect = addsection(ctxt.Arch, &Segtext, ".text", 05)
		sect.Vaddr = va
		s.Sect = sect

		// Create a symbol for the start of the secondary text sections
		ctxt.Syms.Lookup(fmt.Sprintf("runtime.text.%d", n), 0).Sect = sect
		n++
	}
	va += funcsize

	return sect, n, va
}

// assign addresses
func (ctxt *Link) address() {
	va := uint64(*FlagTextAddr)
	Segtext.Rwx = 05
	Segtext.Vaddr = va
	Segtext.Fileoff = uint64(HEADR)
	for _, s := range Segtext.Sections {
		va = uint64(Rnd(int64(va), int64(s.Align)))
		s.Vaddr = va
		va += s.Length
	}

	Segtext.Length = va - uint64(*FlagTextAddr)
	Segtext.Filelen = Segtext.Length
	if ctxt.HeadType == objabi.Hnacl {
		va += 32 // room for the "halt sled"
	}

	if len(Segrodata.Sections) > 0 {
		// align to page boundary so as not to mix
		// rodata and executable text.
		//
		// Note: gold or GNU ld will reduce the size of the executable
		// file by arranging for the relro segment to end at a page
		// boundary, and overlap the end of the text segment with the
		// start of the relro segment in the file.  The PT_LOAD segments
		// will be such that the last page of the text segment will be
		// mapped twice, once r-x and once starting out rw- and, after
		// relocation processing, changed to r--.
		//
		// Ideally the last page of the text segment would not be
		// writable even for this short period.
		va = uint64(Rnd(int64(va), int64(*FlagRound)))

		Segrodata.Rwx = 04
		Segrodata.Vaddr = va
		Segrodata.Fileoff = va - Segtext.Vaddr + Segtext.Fileoff
		Segrodata.Filelen = 0
		if ctxt.HeadType == objabi.Hwindows {
			Segrodata.Fileoff = Segtext.Fileoff + uint64(Rnd(int64(Segtext.Length), PEFILEALIGN))
		}
		for _, s := range Segrodata.Sections {
			va = uint64(Rnd(int64(va), int64(s.Align)))
			s.Vaddr = va
			va += s.Length
		}

		Segrodata.Length = va - Segrodata.Vaddr
		Segrodata.Filelen = Segrodata.Length
	}
	if len(Segrelrodata.Sections) > 0 {
		// align to page boundary so as not to mix
		// rodata, rel-ro data, and executable text.
		va = uint64(Rnd(int64(va), int64(*FlagRound)))

		Segrelrodata.Rwx = 06
		Segrelrodata.Vaddr = va
		Segrelrodata.Fileoff = va - Segrodata.Vaddr + Segrodata.Fileoff
		Segrelrodata.Filelen = 0
		for _, s := range Segrelrodata.Sections {
			va = uint64(Rnd(int64(va), int64(s.Align)))
			s.Vaddr = va
			va += s.Length
		}

		Segrelrodata.Length = va - Segrelrodata.Vaddr
		Segrelrodata.Filelen = Segrelrodata.Length
	}

	va = uint64(Rnd(int64(va), int64(*FlagRound)))
	Segdata.Rwx = 06
	Segdata.Vaddr = va
	Segdata.Fileoff = va - Segtext.Vaddr + Segtext.Fileoff
	Segdata.Filelen = 0
	if ctxt.HeadType == objabi.Hwindows {
		Segdata.Fileoff = Segrodata.Fileoff + uint64(Rnd(int64(Segrodata.Length), PEFILEALIGN))
	}
	if ctxt.HeadType == objabi.Hplan9 {
		Segdata.Fileoff = Segtext.Fileoff + Segtext.Filelen
	}
	var data *sym.Section
	var noptr *sym.Section
	var bss *sym.Section
	var noptrbss *sym.Section
	for i, s := range Segdata.Sections {
		if ctxt.IsELF && s.Name == ".tbss" {
			continue
		}
		vlen := int64(s.Length)
		if i+1 < len(Segdata.Sections) && !(ctxt.IsELF && Segdata.Sections[i+1].Name == ".tbss") {
			vlen = int64(Segdata.Sections[i+1].Vaddr - s.Vaddr)
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

	va = uint64(Rnd(int64(va), int64(*FlagRound)))
	Segdwarf.Rwx = 06
	Segdwarf.Vaddr = va
	Segdwarf.Fileoff = Segdata.Fileoff + uint64(Rnd(int64(Segdata.Filelen), int64(*FlagRound)))
	Segdwarf.Filelen = 0
	if ctxt.HeadType == objabi.Hwindows {
		Segdwarf.Fileoff = Segdata.Fileoff + uint64(Rnd(int64(Segdata.Filelen), PEFILEALIGN))
	}
	for i, s := range Segdwarf.Sections {
		vlen := int64(s.Length)
		if i+1 < len(Segdwarf.Sections) {
			vlen = int64(Segdwarf.Sections[i+1].Vaddr - s.Vaddr)
		}
		s.Vaddr = va
		va += uint64(vlen)
		if ctxt.HeadType == objabi.Hwindows {
			va = uint64(Rnd(int64(va), PEFILEALIGN))
		}
		Segdwarf.Length = va - Segdwarf.Vaddr
	}

	Segdwarf.Filelen = va - Segdwarf.Vaddr

	var (
		text     = Segtext.Sections[0]
		rodata   = ctxt.Syms.Lookup("runtime.rodata", 0).Sect
		itablink = ctxt.Syms.Lookup("runtime.itablink", 0).Sect
		symtab   = ctxt.Syms.Lookup("runtime.symtab", 0).Sect
		pclntab  = ctxt.Syms.Lookup("runtime.pclntab", 0).Sect
		types    = ctxt.Syms.Lookup("runtime.types", 0).Sect
	)
	lasttext := text
	// Could be multiple .text sections
	for _, sect := range Segtext.Sections {
		if sect.Name == ".text" {
			lasttext = sect
		}
	}

	for _, s := range datap {
		if s.Sect != nil {
			s.Value += int64(s.Sect.Vaddr)
		}
		for sub := s.Sub; sub != nil; sub = sub.Sub {
			sub.Value += s.Value
		}
	}

	for _, s := range dwarfp {
		if s.Sect != nil {
			s.Value += int64(s.Sect.Vaddr)
		}
		for sub := s.Sub; sub != nil; sub = sub.Sub {
			sub.Value += s.Value
		}
	}

	if ctxt.BuildMode == BuildModeShared {
		s := ctxt.Syms.Lookup("go.link.abihashbytes", 0)
		sectSym := ctxt.Syms.Lookup(".note.go.abihash", 0)
		s.Sect = sectSym.Sect
		s.Value = int64(sectSym.Sect.Vaddr + 16)
	}

	ctxt.xdefine("runtime.text", sym.STEXT, int64(text.Vaddr))
	ctxt.xdefine("runtime.etext", sym.STEXT, int64(lasttext.Vaddr+lasttext.Length))

	// If there are multiple text sections, create runtime.text.n for
	// their section Vaddr, using n for index
	n := 1
	for _, sect := range Segtext.Sections[1:] {
		if sect.Name != ".text" {
			break
		}
		symname := fmt.Sprintf("runtime.text.%d", n)
		ctxt.xdefine(symname, sym.STEXT, int64(sect.Vaddr))
		n++
	}

	ctxt.xdefine("runtime.rodata", sym.SRODATA, int64(rodata.Vaddr))
	ctxt.xdefine("runtime.erodata", sym.SRODATA, int64(rodata.Vaddr+rodata.Length))
	ctxt.xdefine("runtime.types", sym.SRODATA, int64(types.Vaddr))
	ctxt.xdefine("runtime.etypes", sym.SRODATA, int64(types.Vaddr+types.Length))
	ctxt.xdefine("runtime.itablink", sym.SRODATA, int64(itablink.Vaddr))
	ctxt.xdefine("runtime.eitablink", sym.SRODATA, int64(itablink.Vaddr+itablink.Length))

	s := ctxt.Syms.Lookup("runtime.gcdata", 0)
	s.Attr |= sym.AttrLocal
	ctxt.xdefine("runtime.egcdata", sym.SRODATA, Symaddr(s)+s.Size)
	ctxt.Syms.Lookup("runtime.egcdata", 0).Sect = s.Sect

	s = ctxt.Syms.Lookup("runtime.gcbss", 0)
	s.Attr |= sym.AttrLocal
	ctxt.xdefine("runtime.egcbss", sym.SRODATA, Symaddr(s)+s.Size)
	ctxt.Syms.Lookup("runtime.egcbss", 0).Sect = s.Sect

	ctxt.xdefine("runtime.symtab", sym.SRODATA, int64(symtab.Vaddr))
	ctxt.xdefine("runtime.esymtab", sym.SRODATA, int64(symtab.Vaddr+symtab.Length))
	ctxt.xdefine("runtime.pclntab", sym.SRODATA, int64(pclntab.Vaddr))
	ctxt.xdefine("runtime.epclntab", sym.SRODATA, int64(pclntab.Vaddr+pclntab.Length))
	ctxt.xdefine("runtime.noptrdata", sym.SNOPTRDATA, int64(noptr.Vaddr))
	ctxt.xdefine("runtime.enoptrdata", sym.SNOPTRDATA, int64(noptr.Vaddr+noptr.Length))
	ctxt.xdefine("runtime.bss", sym.SBSS, int64(bss.Vaddr))
	ctxt.xdefine("runtime.ebss", sym.SBSS, int64(bss.Vaddr+bss.Length))
	ctxt.xdefine("runtime.data", sym.SDATA, int64(data.Vaddr))
	ctxt.xdefine("runtime.edata", sym.SDATA, int64(data.Vaddr+data.Length))
	ctxt.xdefine("runtime.noptrbss", sym.SNOPTRBSS, int64(noptrbss.Vaddr))
	ctxt.xdefine("runtime.enoptrbss", sym.SNOPTRBSS, int64(noptrbss.Vaddr+noptrbss.Length))
	ctxt.xdefine("runtime.end", sym.SBSS, int64(Segdata.Vaddr+Segdata.Length))
}

// add a trampoline with symbol s (to be laid down after the current function)
func (ctxt *Link) AddTramp(s *sym.Symbol) {
	s.Type = sym.STEXT
	s.Attr |= sym.AttrReachable
	s.Attr |= sym.AttrOnList
	ctxt.tramps = append(ctxt.tramps, s)
	if *FlagDebugTramp > 0 && ctxt.Debugvlog > 0 {
		ctxt.Logf("trampoline %s inserted\n", s)
	}
}
