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
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
)

func Symgrow(s *Symbol, siz int64) {
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

func Addrel(s *Symbol) *Reloc {
	s.R = append(s.R, Reloc{})
	return &s.R[len(s.R)-1]
}

func setuintxx(ctxt *Link, s *Symbol, off int64, v uint64, wid int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Attr |= AttrReachable
	if s.Size < off+wid {
		s.Size = off + wid
		Symgrow(s, s.Size)
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

func Addbytes(s *Symbol, bytes []byte) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Attr |= AttrReachable
	s.P = append(s.P, bytes...)
	s.Size = int64(len(s.P))

	return s.Size
}

func adduintxx(ctxt *Link, s *Symbol, v uint64, wid int) int64 {
	off := s.Size
	setuintxx(ctxt, s, off, v, int64(wid))
	return off
}

func Adduint8(ctxt *Link, s *Symbol, v uint8) int64 {
	off := s.Size
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Attr |= AttrReachable
	s.Size++
	s.P = append(s.P, v)

	return off
}

func Adduint16(ctxt *Link, s *Symbol, v uint16) int64 {
	return adduintxx(ctxt, s, uint64(v), 2)
}

func Adduint32(ctxt *Link, s *Symbol, v uint32) int64 {
	return adduintxx(ctxt, s, uint64(v), 4)
}

func Adduint64(ctxt *Link, s *Symbol, v uint64) int64 {
	return adduintxx(ctxt, s, v, 8)
}

func adduint(ctxt *Link, s *Symbol, v uint64) int64 {
	return adduintxx(ctxt, s, v, SysArch.PtrSize)
}

func setuint8(ctxt *Link, s *Symbol, r int64, v uint8) int64 {
	return setuintxx(ctxt, s, r, uint64(v), 1)
}

func setuint32(ctxt *Link, s *Symbol, r int64, v uint32) int64 {
	return setuintxx(ctxt, s, r, uint64(v), 4)
}

func setuint(ctxt *Link, s *Symbol, r int64, v uint64) int64 {
	return setuintxx(ctxt, s, r, v, int64(SysArch.PtrSize))
}

func Addaddrplus(ctxt *Link, s *Symbol, t *Symbol, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Attr |= AttrReachable
	i := s.Size
	s.Size += int64(ctxt.Arch.PtrSize)
	Symgrow(s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.PtrSize)
	r.Type = objabi.R_ADDR
	r.Add = add
	return i + int64(r.Siz)
}

func Addpcrelplus(ctxt *Link, s *Symbol, t *Symbol, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Attr |= AttrReachable
	i := s.Size
	s.Size += 4
	Symgrow(s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Add = add
	r.Type = objabi.R_PCREL
	r.Siz = 4
	if SysArch.Family == sys.S390X {
		r.Variant = RV_390_DBL
	}
	return i + int64(r.Siz)
}

func Addaddr(ctxt *Link, s *Symbol, t *Symbol) int64 {
	return Addaddrplus(ctxt, s, t, 0)
}

func setaddrplus(ctxt *Link, s *Symbol, off int64, t *Symbol, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Attr |= AttrReachable
	if off+int64(ctxt.Arch.PtrSize) > s.Size {
		s.Size = off + int64(ctxt.Arch.PtrSize)
		Symgrow(s, s.Size)
	}

	r := Addrel(s)
	r.Sym = t
	r.Off = int32(off)
	r.Siz = uint8(ctxt.Arch.PtrSize)
	r.Type = objabi.R_ADDR
	r.Add = add
	return off + int64(r.Siz)
}

func setaddr(ctxt *Link, s *Symbol, off int64, t *Symbol) int64 {
	return setaddrplus(ctxt, s, off, t, 0)
}

func addsize(ctxt *Link, s *Symbol, t *Symbol) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Attr |= AttrReachable
	i := s.Size
	s.Size += int64(ctxt.Arch.PtrSize)
	Symgrow(s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.PtrSize)
	r.Type = objabi.R_SIZE
	return i + int64(r.Siz)
}

func addaddrplus4(ctxt *Link, s *Symbol, t *Symbol, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	s.Attr |= AttrReachable
	i := s.Size
	s.Size += 4
	Symgrow(s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = 4
	r.Type = objabi.R_ADDR
	r.Add = add
	return i + int64(r.Siz)
}

/*
 * divide-and-conquer list-link (by Sub) sort of Symbol* by Value.
 * Used for sub-symbols when loading host objects (see e.g. ldelf.go).
 */

func listsort(l *Symbol) *Symbol {
	if l == nil || l.Sub == nil {
		return l
	}

	l1 := l
	l2 := l
	for {
		l2 = l2.Sub
		if l2 == nil {
			break
		}
		l2 = l2.Sub
		if l2 == nil {
			break
		}
		l1 = l1.Sub
	}

	l2 = l1.Sub
	l1.Sub = nil
	l1 = listsort(l)
	l2 = listsort(l2)

	/* set up lead element */
	if l1.Value < l2.Value {
		l = l1
		l1 = l1.Sub
	} else {
		l = l2
		l2 = l2.Sub
	}

	le := l

	for {
		if l1 == nil {
			for l2 != nil {
				le.Sub = l2
				le = l2
				l2 = l2.Sub
			}

			le.Sub = nil
			break
		}

		if l2 == nil {
			for l1 != nil {
				le.Sub = l1
				le = l1
				l1 = l1.Sub
			}

			break
		}

		if l1.Value < l2.Value {
			le.Sub = l1
			le = l1
			l1 = l1.Sub
		} else {
			le.Sub = l2
			le = l2
			l2 = l2.Sub
		}
	}

	le.Sub = nil
	return l
}

// isRuntimeDepPkg returns whether pkg is the runtime package or its dependency
func isRuntimeDepPkg(pkg string) bool {
	switch pkg {
	case "runtime",
		"sync/atomic": // runtime may call to sync/atomic, due to go:linkname
		return true
	}
	return strings.HasPrefix(pkg, "runtime/internal/") && !strings.HasSuffix(pkg, "_test")
}

// Estimate the max size needed to hold any new trampolines created for this function. This
// is used to determine when the section can be split if it becomes too large, to ensure that
// the trampolines are in the same section as the function that uses them.
func maxSizeTrampolinesPPC64(s *Symbol, isTramp bool) uint64 {
	// If Thearch.Trampoline is nil, then trampoline support is not available on this arch.
	// A trampoline does not need any dependent trampolines.
	if Thearch.Trampoline == nil || isTramp {
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
func trampoline(ctxt *Link, s *Symbol) {
	if Thearch.Trampoline == nil {
		return // no need or no support of trampolines on this arch
	}

	for ri := range s.R {
		r := &s.R[ri]
		if !r.Type.IsDirectJump() {
			continue
		}
		if Symaddr(r.Sym) == 0 && r.Sym.Type != SDYNIMPORT {
			if r.Sym.File != s.File {
				if !isRuntimeDepPkg(s.File) || !isRuntimeDepPkg(r.Sym.File) {
					Errorf(s, "unresolved inter-package jump to %s(%s)", r.Sym, r.Sym.File)
				}
				// runtime and its dependent packages may call to each other.
				// they are fine, as they will be laid down together.
			}
			continue
		}

		Thearch.Trampoline(ctxt, r, s)
	}

}

// resolve relocations in s.
func relocsym(ctxt *Link, s *Symbol) {
	var r *Reloc
	var rs *Symbol
	var i16 int16
	var off int32
	var siz int32
	var fl int32
	var o int64

	for ri := int32(0); ri < int32(len(s.R)); ri++ {
		r = &s.R[ri]

		r.Done = 1
		off = r.Off
		siz = int32(r.Siz)
		if off < 0 || off+siz > int32(len(s.P)) {
			rname := ""
			if r.Sym != nil {
				rname = r.Sym.Name
			}
			Errorf(s, "invalid relocation %s: %d+%d not in [%d,%d)", rname, off, siz, 0, len(s.P))
			continue
		}

		if r.Sym != nil && (r.Sym.Type&(SMASK|SHIDDEN) == 0 || r.Sym.Type&SMASK == SXREF) {
			// When putting the runtime but not main into a shared library
			// these symbols are undefined and that's OK.
			if Buildmode == BuildmodeShared {
				if r.Sym.Name == "main.main" || r.Sym.Name == "main.init" {
					r.Sym.Type = SDYNIMPORT
				} else if strings.HasPrefix(r.Sym.Name, "go.info.") {
					// Skip go.info symbols. They are only needed to communicate
					// DWARF info between the compiler and linker.
					continue
				}
			} else {
				Errorf(s, "relocation target %s not defined", r.Sym.Name)
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
		if Headtype != objabi.Hsolaris && r.Sym != nil && r.Sym.Type == SDYNIMPORT && !ctxt.DynlinkingGo() {
			if !(SysArch.Family == sys.PPC64 && Linkmode == LinkExternal && r.Sym.Name == ".TOC.") {
				Errorf(s, "unhandled relocation for %s (type %d rtype %d)", r.Sym.Name, r.Sym.Type, r.Type)
			}
		}
		if r.Sym != nil && r.Sym.Type != STLSBSS && r.Type != objabi.R_WEAKADDROFF && !r.Sym.Attr.Reachable() {
			Errorf(s, "unreachable sym in relocation: %s", r.Sym.Name)
		}

		// TODO(mundaym): remove this special case - see issue 14218.
		if SysArch.Family == sys.S390X {
			switch r.Type {
			case objabi.R_PCRELDBL:
				r.Type = objabi.R_PCREL
				r.Variant = RV_390_DBL
			case objabi.R_CALL:
				r.Variant = RV_390_DBL
			}
		}

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
			if Thearch.Archreloc(ctxt, r, s, &o) < 0 {
				Errorf(s, "unknown reloc to %v: %v", r.Sym.Name, r.Type)
			}

		case objabi.R_TLS_LE:
			isAndroidX86 := objabi.GOOS == "android" && (SysArch.InFamily(sys.AMD64, sys.I386))

			if Linkmode == LinkExternal && Iself && !isAndroidX86 {
				r.Done = 0
				if r.Sym == nil {
					r.Sym = ctxt.Tlsg
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
			} else if Iself || Headtype == objabi.Hplan9 || Headtype == objabi.Hdarwin || isAndroidX86 {
				o = int64(ctxt.Tlsoffset) + r.Add
			} else if Headtype == objabi.Hwindows {
				o = r.Add
			} else {
				log.Fatalf("unexpected R_TLS_LE relocation for %v", Headtype)
			}

		case objabi.R_TLS_IE:
			isAndroidX86 := objabi.GOOS == "android" && (SysArch.InFamily(sys.AMD64, sys.I386))

			if Linkmode == LinkExternal && Iself && !isAndroidX86 {
				r.Done = 0
				if r.Sym == nil {
					r.Sym = ctxt.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if SysArch.Family != sys.AMD64 {
					o = r.Add
				}
				break
			}
			if Buildmode == BuildmodePIE && Iself {
				// We are linking the final executable, so we
				// can optimize any TLS IE relocation to LE.
				if Thearch.TLSIEtoLE == nil {
					log.Fatalf("internal linking of TLS IE not supported on %v", SysArch.Family)
				}
				Thearch.TLSIEtoLE(s, int(off), int(r.Siz))
				o = int64(ctxt.Tlsoffset)
				// TODO: o += r.Add when SysArch.Family != sys.AMD64?
				// Why do we treat r.Add differently on AMD64?
				// Is the external linker using Xadd at all?
			} else {
				log.Fatalf("cannot handle R_TLS_IE (sym %s) when linking internally", s.Name)
			}

		case objabi.R_ADDR:
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
					Errorf(s, "missing section for relocation target %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if Iself {
					if SysArch.Family == sys.AMD64 {
						o = 0
					}
				} else if Headtype == objabi.Hdarwin {
					// ld64 for arm64 has a bug where if the address pointed to by o exists in the
					// symbol table (dynid >= 0), or is inside a symbol that exists in the symbol
					// table, then it will add o twice into the relocated value.
					// The workaround is that on arm64 don't ever add symaddr to o and always use
					// extern relocation by requiring rs->dynid >= 0.
					if rs.Type != SHOSTOBJ {
						if SysArch.Family == sys.ARM64 && rs.Dynid < 0 {
							Errorf(s, "R_ADDR reloc to %s+%d is not supported on darwin/arm64", rs.Name, o)
						}
						if SysArch.Family != sys.ARM64 {
							o += Symaddr(rs)
						}
					}
				} else if Headtype == objabi.Hwindows {
					// nothing to do
				} else {
					Errorf(s, "unhandled pcrel relocation to %s on %v", rs.Name, Headtype)
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
				Errorf(s, "non-pc-relative relocation address for %s is too big: %#x (%#x + %#x)", r.Sym.Name, uint64(o), Symaddr(r.Sym), r.Add)
				errorexit()
			}

		case objabi.R_DWARFREF:
			var sectName string
			var vaddr int64
			switch {
			case r.Sym.Sect != nil:
				sectName = r.Sym.Sect.Name
				vaddr = int64(r.Sym.Sect.Vaddr)
			case r.Sym.Type == SDWARFRANGE:
				sectName = ".debug_ranges"
			default:
				Errorf(s, "missing DWARF section for relocation target %s", r.Sym.Name)
			}

			if Linkmode == LinkExternal {
				r.Done = 0
				// PE code emits IMAGE_REL_I386_SECREL and IMAGE_REL_AMD64_SECREL
				// for R_DWARFREF relocations, while R_ADDR is replaced with
				// IMAGE_REL_I386_DIR32, IMAGE_REL_AMD64_ADDR64 and IMAGE_REL_AMD64_ADDR32.
				// Do not replace R_DWARFREF with R_ADDR for windows -
				// let PE code emit correct relocations.
				if Headtype != objabi.Hwindows {
					r.Type = objabi.R_ADDR
				}

				r.Xsym = ctxt.Syms.ROLookup(sectName, 0)
				r.Xadd = r.Add + Symaddr(r.Sym) - vaddr

				o = r.Xadd
				rs = r.Xsym
				if Iself && SysArch.Family == sys.AMD64 {
					o = 0
				}
				break
			}
			o = Symaddr(r.Sym) + r.Add - vaddr

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

			// r->sym can be null when CALL $(constant) is transformed from absolute PC to relative PC call.
		case objabi.R_GOTPCREL:
			if ctxt.DynlinkingGo() && Headtype == objabi.Hdarwin && r.Sym != nil && r.Sym.Type != SCONST {
				r.Done = 0
				r.Xadd = r.Add
				r.Xadd -= int64(r.Siz) // relative to address after the relocated chunk
				r.Xsym = r.Sym

				o = r.Xadd
				o += int64(r.Siz)
				break
			}
			fallthrough
		case objabi.R_CALL, objabi.R_PCREL:
			if Linkmode == LinkExternal && r.Sym != nil && r.Sym.Type != SCONST && (r.Sym.Sect != s.Sect || r.Type == objabi.R_GOTPCREL) {
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
					Errorf(s, "missing section for relocation target %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if Iself {
					if SysArch.Family == sys.AMD64 {
						o = 0
					}
				} else if Headtype == objabi.Hdarwin {
					if r.Type == objabi.R_CALL {
						if rs.Type != SHOSTOBJ {
							o += int64(uint64(Symaddr(rs)) - rs.Sect.Vaddr)
						}
						o -= int64(r.Off) // relative to section offset, not symbol
					} else if SysArch.Family == sys.ARM {
						// see ../arm/asm.go:/machoreloc1
						o += Symaddr(rs) - int64(s.Value) - int64(r.Off)
					} else {
						o += int64(r.Siz)
					}
				} else if Headtype == objabi.Hwindows && SysArch.Family == sys.AMD64 { // only amd64 needs PCREL
					// PE/COFF's PC32 relocation uses the address after the relocated
					// bytes as the base. Compensate by skewing the addend.
					o += int64(r.Siz)
				} else {
					Errorf(s, "unhandled pcrel relocation to %s on %v", rs.Name, Headtype)
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

		if r.Variant != RV_NONE {
			o = Thearch.Archrelocvariant(ctxt, r, s, o)
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
			Errorf(s, "bad reloc size %#x for %s", uint32(siz), r.Sym.Name)
			fallthrough

			// TODO(rsc): Remove.
		case 1:
			s.P[off] = byte(int8(o))

		case 2:
			if o != int64(int16(o)) {
				Errorf(s, "relocation address for %s is too big: %#x", r.Sym.Name, o)
			}
			i16 = int16(o)
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

			fl = int32(o)
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
	for _, sym := range datap {
		relocsym(ctxt, sym)
	}
	for _, s := range dwarfp {
		relocsym(ctxt, s)
	}
}

func dynrelocsym(ctxt *Link, s *Symbol) {
	if Headtype == objabi.Hwindows && Linkmode != LinkExternal {
		rel := ctxt.Syms.Lookup(".rel", 0)
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
				if SysArch.Family == sys.I386 {
					Adduint8(ctxt, rel, 0xff)
					Adduint8(ctxt, rel, 0x25)
					Addaddr(ctxt, rel, targ)
					Adduint8(ctxt, rel, 0x90)
					Adduint8(ctxt, rel, 0x90)
				} else {
					Adduint8(ctxt, rel, 0xff)
					Adduint8(ctxt, rel, 0x24)
					Adduint8(ctxt, rel, 0x25)
					addaddrplus4(ctxt, rel, targ, 0)
					Adduint8(ctxt, rel, 0x90)
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
		if Buildmode == BuildmodePIE && Linkmode == LinkInternal {
			// It's expected that some relocations will be done
			// later by relocsym (R_TLS_LE, R_ADDROFF), so
			// don't worry if Adddynrel returns false.
			Thearch.Adddynrel(ctxt, s, r)
			continue
		}
		if r.Sym != nil && r.Sym.Type == SDYNIMPORT || r.Type >= 256 {
			if r.Sym != nil && !r.Sym.Attr.Reachable() {
				Errorf(s, "dynamic relocation to unreachable symbol %s", r.Sym.Name)
			}
			if !Thearch.Adddynrel(ctxt, s, r) {
				Errorf(s, "unsupported dynamic relocation for symbol %s (type=%d stype=%d)", r.Sym.Name, r.Type, r.Sym.Type)
			}
		}
	}
}

func dynreloc(ctxt *Link, data *[SXREF][]*Symbol) {
	// -d suppresses dynamic loader format, so we may as well not
	// compute these sections or mark their symbols as reachable.
	if *FlagD && Headtype != objabi.Hwindows {
		return
	}
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f reloc\n", Cputime())
	}

	for _, s := range ctxt.Textp {
		dynrelocsym(ctxt, s)
	}
	for _, syms := range data {
		for _, sym := range syms {
			dynrelocsym(ctxt, sym)
		}
	}
	if Iself {
		elfdynhash(ctxt)
	}
}

func Codeblk(ctxt *Link, addr int64, size int64) {
	CodeblkPad(ctxt, addr, size, zeros[:])
}
func CodeblkPad(ctxt *Link, addr int64, size int64, pad []byte) {
	if *flagA {
		ctxt.Logf("codeblk [%#x,%#x) at offset %#x\n", addr, addr+size, coutbuf.Offset())
	}

	blk(ctxt, ctxt.Textp, addr, size, pad)

	/* again for printing */
	if !*flagA {
		return
	}

	syms := ctxt.Textp
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
			ctxt.Logf("%-20s %.8x|", "_", uint64(addr))
			for ; addr < sym.Value; addr++ {
				ctxt.Logf(" %.2x", 0)
			}
			ctxt.Logf("\n")
		}

		ctxt.Logf("%.6x\t%-20s\n", uint64(addr), sym.Name)
		q = sym.P

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

func blk(ctxt *Link, syms []*Symbol, addr, size int64, pad []byte) {
	for i, s := range syms {
		if s.Type&SSUB == 0 && s.Value >= addr {
			syms = syms[i:]
			break
		}
	}

	eaddr := addr + size
	for _, s := range syms {
		if s.Type&SSUB != 0 {
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
			Errorf(s, "phase error: addr=%#x value+size=%#x", addr, s.Value+s.Size)
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

func Datblk(ctxt *Link, addr int64, size int64) {
	if *flagA {
		ctxt.Logf("datblk [%#x,%#x) at offset %#x\n", addr, addr+size, coutbuf.Offset())
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
		ctxt.Logf("dwarfblk [%#x,%#x) at offset %#x\n", addr, addr+size, coutbuf.Offset())
	}

	blk(ctxt, dwarfp, addr, size, zeros[:])
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

var strdata []*Symbol

func addstrdata1(ctxt *Link, arg string) {
	eq := strings.Index(arg, "=")
	dot := strings.LastIndex(arg[:eq+1], ".")
	if eq < 0 || dot < 0 {
		Exitf("-X flag requires argument of the form importpath.name=value")
	}
	addstrdata(ctxt, objabi.PathToPrefix(arg[:dot])+arg[dot:eq], arg[eq+1:])
}

func addstrdata(ctxt *Link, name string, value string) {
	p := fmt.Sprintf("%s.str", name)
	sp := ctxt.Syms.Lookup(p, 0)

	Addstring(sp, value)
	sp.Type = SRODATA

	s := ctxt.Syms.Lookup(name, 0)
	s.Size = 0
	s.Attr |= AttrDuplicateOK
	reachable := s.Attr.Reachable()
	Addaddr(ctxt, s, sp)
	adduintxx(ctxt, s, uint64(len(value)), SysArch.PtrSize)

	// addstring, addaddr, etc., mark the symbols as reachable.
	// In this case that is not necessarily true, so stick to what
	// we know before entering this function.
	s.Attr.Set(AttrReachable, reachable)

	strdata = append(strdata, s)

	sp.Attr.Set(AttrReachable, reachable)
}

func (ctxt *Link) checkstrdata() {
	for _, s := range strdata {
		if s.Type == STEXT {
			Errorf(s, "cannot use -X with text symbol")
		} else if s.Gotype != nil && s.Gotype.Name != "type.string" {
			Errorf(s, "cannot use -X with non-string symbol")
		}
	}
}

func Addstring(s *Symbol, str string) int64 {
	if s.Type == 0 {
		s.Type = SNOPTRDATA
	}
	s.Attr |= AttrReachable
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
func addgostring(ctxt *Link, s *Symbol, symname, str string) {
	sym := ctxt.Syms.Lookup(symname, 0)
	if sym.Type != Sxxx {
		Errorf(s, "duplicate symname in addgostring: %s", symname)
	}
	sym.Attr |= AttrReachable
	sym.Attr |= AttrLocal
	sym.Type = SRODATA
	sym.Size = int64(len(str))
	sym.P = []byte(str)
	Addaddr(ctxt, s, sym)
	adduint(ctxt, s, uint64(len(str)))
}

func addinitarrdata(ctxt *Link, s *Symbol) {
	p := s.Name + ".ptr"
	sp := ctxt.Syms.Lookup(p, 0)
	sp.Type = SINITARR
	sp.Size = 0
	sp.Attr |= AttrDuplicateOK
	Addaddr(ctxt, sp, s)
}

func dosymtype(ctxt *Link) {
	switch Buildmode {
	case BuildmodeCArchive, BuildmodeCShared:
		for _, s := range ctxt.Syms.Allsym {
			// Create a new entry in the .init_array section that points to the
			// library initializer function.
			switch Buildmode {
			case BuildmodeCArchive, BuildmodeCShared:
				if s.Name == *flagEntrySymbol {
					addinitarrdata(ctxt, s)
				}
			}
		}
	}
}

// symalign returns the required alignment for the given symbol s.
func symalign(s *Symbol) int32 {
	min := int32(Thearch.Minalign)
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
	align := int32(Thearch.Maxalign)
	for int64(align) > s.Size && align > min {
		align >>= 1
	}
	return align
}

func aligndatsize(datsize int64, s *Symbol) int64 {
	return Rnd(datsize, int64(symalign(s)))
}

const debugGCProg = false

type GCProg struct {
	ctxt *Link
	sym  *Symbol
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
		Adduint8(ctxt, p.sym, x)
	}
}

func (p *GCProg) End(size int64) {
	p.w.ZeroUntil(size / int64(SysArch.PtrSize))
	p.w.End()
	if debugGCProg {
		fmt.Fprintf(os.Stderr, "ld: end GCProg\n")
	}
}

func (p *GCProg) AddSym(s *Symbol) {
	typ := s.Gotype
	// Things without pointers should be in SNOPTRDATA or SNOPTRBSS;
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

	ptrsize := int64(SysArch.PtrSize)
	nptr := decodetypePtrdata(p.ctxt.Arch, typ) / ptrsize

	if debugGCProg {
		fmt.Fprintf(os.Stderr, "gcprog sym: %s at %d (ptr=%d+%d)\n", s.Name, s.Value, s.Value/ptrsize, nptr)
	}

	if decodetypeUsegcprog(typ) == 0 {
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

// dataSortKey is used to sort a slice of data symbol *Symbol pointers.
// The sort keys are kept inline to improve cache behavior while sorting.
type dataSortKey struct {
	size int64
	name string
	sym  *Symbol
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

func checkdatsize(ctxt *Link, datsize int64, symn SymKind) {
	if datsize > cutoff {
		Errorf(nil, "too much data in section %v (over %d bytes)", symn, cutoff)
	}
}

// datap is a collection of reachable data symbols in address order.
// Generated by dodata.
var datap []*Symbol

func (ctxt *Link) dodata() {
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f dodata\n", Cputime())
	}

	if ctxt.DynlinkingGo() && Headtype == objabi.Hdarwin {
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
		bss.Attr.Set(AttrSpecial, false)

		ctxt.Syms.Lookup("runtime.ebss", 0).Attr.Set(AttrSpecial, false)

		data := ctxt.Syms.Lookup("runtime.data", 0)
		data.Size = 8
		data.Attr.Set(AttrSpecial, false)

		ctxt.Syms.Lookup("runtime.edata", 0).Attr.Set(AttrSpecial, false)

		types := ctxt.Syms.Lookup("runtime.types", 0)
		types.Type = STYPE
		types.Size = 8
		types.Attr.Set(AttrSpecial, false)

		etypes := ctxt.Syms.Lookup("runtime.etypes", 0)
		etypes.Type = SFUNCTAB
		etypes.Attr.Set(AttrSpecial, false)
	}

	// Collect data symbols by type into data.
	var data [SXREF][]*Symbol
	for _, s := range ctxt.Syms.Allsym {
		if !s.Attr.Reachable() || s.Attr.Special() {
			continue
		}
		if s.Type <= STEXT || s.Type >= SXREF {
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
	if Headtype == objabi.Hdarwin {
		machosymorder(ctxt)
	}
	dynreloc(ctxt, &data)

	if UseRelro() {
		// "read only" data with relocations needs to go in its own section
		// when building a shared library. We do this by boosting objects of
		// type SXXX with relocations to type SXXXRELRO.
		for _, symnro := range readOnly {
			symnrelro := relROMap[symnro]

			ro := []*Symbol{}
			relro := data[symnrelro]

			for _, s := range data[symnro] {
				isRelro := len(s.R) > 0
				switch s.Type {
				case STYPE, STYPERELRO, SGOFUNCRELRO:
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
	var dataMaxAlign [SXREF]int32
	var wg sync.WaitGroup
	for symn := range data {
		symn := SymKind(symn)
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
	writable := []SymKind{
		SELFSECT,
		SMACHO,
		SMACHOGOT,
		SWINDOWS,
	}
	for _, symn := range writable {
		for _, s := range data[symn] {
			sect := addsection(&Segdata, s.Name, 06)
			sect.Align = symalign(s)
			datsize = Rnd(datsize, int64(sect.Align))
			sect.Vaddr = uint64(datsize)
			s.Sect = sect
			s.Type = SDATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
			sect.Length = uint64(datsize) - sect.Vaddr
		}
		checkdatsize(ctxt, datsize, symn)
	}

	// .got (and .toc on ppc64)
	if len(data[SELFGOT]) > 0 {
		sect := addsection(&Segdata, ".got", 06)
		sect.Align = dataMaxAlign[SELFGOT]
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		var toc *Symbol
		for _, s := range data[SELFGOT] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Type = SDATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)

			// Resolve .TOC. symbol for this object file (ppc64)
			toc = ctxt.Syms.ROLookup(".TOC.", int(s.Version))
			if toc != nil {
				toc.Sect = sect
				toc.Outer = s
				toc.Sub = s.Sub
				s.Sub = toc

				toc.Value = 0x8000
			}

			datsize += s.Size
		}
		checkdatsize(ctxt, datsize, SELFGOT)
		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* pointer-free data */
	sect := addsection(&Segdata, ".noptrdata", 06)
	sect.Align = dataMaxAlign[SNOPTRDATA]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.noptrdata", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.enoptrdata", 0).Sect = sect
	for _, s := range data[SNOPTRDATA] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SDATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, SNOPTRDATA)
	sect.Length = uint64(datsize) - sect.Vaddr

	hasinitarr := *FlagLinkshared

	/* shared library initializer */
	switch Buildmode {
	case BuildmodeCArchive, BuildmodeCShared, BuildmodeShared, BuildmodePlugin:
		hasinitarr = true
	}
	if hasinitarr {
		sect := addsection(&Segdata, ".init_array", 06)
		sect.Align = dataMaxAlign[SINITARR]
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for _, s := range data[SINITARR] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
		}
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(ctxt, datsize, SINITARR)
	}

	/* data */
	sect = addsection(&Segdata, ".data", 06)
	sect.Align = dataMaxAlign[SDATA]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.data", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.edata", 0).Sect = sect
	var gc GCProg
	gc.Init(ctxt, "runtime.gcdata")
	for _, s := range data[SDATA] {
		s.Sect = sect
		s.Type = SDATA
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		gc.AddSym(s)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, SDATA)
	sect.Length = uint64(datsize) - sect.Vaddr
	gc.End(int64(sect.Length))

	/* bss */
	sect = addsection(&Segdata, ".bss", 06)
	sect.Align = dataMaxAlign[SBSS]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.bss", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.ebss", 0).Sect = sect
	gc = GCProg{}
	gc.Init(ctxt, "runtime.gcbss")
	for _, s := range data[SBSS] {
		s.Sect = sect
		datsize = aligndatsize(datsize, s)
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		gc.AddSym(s)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, SBSS)
	sect.Length = uint64(datsize) - sect.Vaddr
	gc.End(int64(sect.Length))

	/* pointer-free bss */
	sect = addsection(&Segdata, ".noptrbss", 06)
	sect.Align = dataMaxAlign[SNOPTRBSS]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.noptrbss", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.enoptrbss", 0).Sect = sect
	for _, s := range data[SNOPTRBSS] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}

	sect.Length = uint64(datsize) - sect.Vaddr
	ctxt.Syms.Lookup("runtime.end", 0).Sect = sect
	checkdatsize(ctxt, datsize, SNOPTRBSS)

	if len(data[STLSBSS]) > 0 {
		var sect *Section
		if Iself && (Linkmode == LinkExternal || !*FlagD) {
			sect = addsection(&Segdata, ".tbss", 06)
			sect.Align = int32(SysArch.PtrSize)
			sect.Vaddr = 0
		}
		datsize = 0

		for _, s := range data[STLSBSS] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Value = datsize
			datsize += s.Size
		}
		checkdatsize(ctxt, datsize, STLSBSS)

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
	if len(data[STEXT]) != 0 {
		Errorf(nil, "dodata found an STEXT symbol: %s", data[STEXT][0].Name)
	}
	for _, s := range data[SELFRXSECT] {
		sect := addsection(&Segtext, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(ctxt, datsize, SELFRXSECT)
	}

	/* read-only data */
	sect = addsection(segro, ".rodata", 04)

	sect.Vaddr = 0
	ctxt.Syms.Lookup("runtime.rodata", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.erodata", 0).Sect = sect
	if !UseRelro() {
		ctxt.Syms.Lookup("runtime.types", 0).Sect = sect
		ctxt.Syms.Lookup("runtime.etypes", 0).Sect = sect
	}
	for _, symn := range readOnly {
		align := dataMaxAlign[symn]
		if sect.Align < align {
			sect.Align = align
		}
	}
	datsize = Rnd(datsize, int64(sect.Align))
	for _, symn := range readOnly {
		for _, s := range data[symn] {
			datsize = aligndatsize(datsize, s)
			s.Sect = sect
			s.Type = SRODATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			datsize += s.Size
		}
		checkdatsize(ctxt, datsize, symn)
	}
	sect.Length = uint64(datsize) - sect.Vaddr

	/* read-only ELF, Mach-O sections */
	for _, s := range data[SELFROSECT] {
		sect = addsection(segro, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(ctxt, datsize, SELFROSECT)

	for _, s := range data[SMACHOPLT] {
		sect = addsection(segro, s.Name, 04)
		sect.Align = symalign(s)
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(ctxt, datsize, SMACHOPLT)

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
	addrelrosection := func(suffix string) *Section {
		return addsection(segro, suffix, 04)
	}

	if UseRelro() {
		addrelrosection = func(suffix string) *Section {
			seg := &Segrelrodata
			if Linkmode == LinkExternal {
				// Using a separate segment with an external
				// linker results in some programs moving
				// their data sections unexpectedly, which
				// corrupts the moduledata. So we use the
				// rodata segment and let the external linker
				// sort out a rel.ro segment.
				seg = &Segrodata
			}
			return addsection(seg, ".data.rel.ro"+suffix, 06)
		}
		/* data only written by relocations */
		sect = addrelrosection("")

		sect.Vaddr = 0
		ctxt.Syms.Lookup("runtime.types", 0).Sect = sect
		ctxt.Syms.Lookup("runtime.etypes", 0).Sect = sect
		for _, symnro := range readOnly {
			symn := relROMap[symnro]
			align := dataMaxAlign[symn]
			if sect.Align < align {
				sect.Align = align
			}
		}
		datsize = Rnd(datsize, int64(sect.Align))
		for _, symnro := range readOnly {
			symn := relROMap[symnro]
			for _, s := range data[symn] {
				datsize = aligndatsize(datsize, s)
				if s.Outer != nil && s.Outer.Sect != nil && s.Outer.Sect != sect {
					Errorf(s, "s.Outer (%s) in different section from s, %s != %s", s.Outer.Name, s.Outer.Sect.Name, sect.Name)
				}
				s.Sect = sect
				s.Type = SRODATA
				s.Value = int64(uint64(datsize) - sect.Vaddr)
				datsize += s.Size
			}
			checkdatsize(ctxt, datsize, symn)
		}

		sect.Length = uint64(datsize) - sect.Vaddr
	}

	/* typelink */
	sect = addrelrosection(".typelink")
	sect.Align = dataMaxAlign[STYPELINK]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	typelink := ctxt.Syms.Lookup("runtime.typelink", 0)
	typelink.Sect = sect
	typelink.Type = SRODATA
	datsize += typelink.Size
	checkdatsize(ctxt, datsize, STYPELINK)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* itablink */
	sect = addrelrosection(".itablink")
	sect.Align = dataMaxAlign[SITABLINK]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.itablink", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.eitablink", 0).Sect = sect
	for _, s := range data[SITABLINK] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, SITABLINK)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* gosymtab */
	sect = addrelrosection(".gosymtab")
	sect.Align = dataMaxAlign[SSYMTAB]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.symtab", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.esymtab", 0).Sect = sect
	for _, s := range data[SSYMTAB] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, SSYMTAB)
	sect.Length = uint64(datsize) - sect.Vaddr

	/* gopclntab */
	sect = addrelrosection(".gopclntab")
	sect.Align = dataMaxAlign[SPCLNTAB]
	datsize = Rnd(datsize, int64(sect.Align))
	sect.Vaddr = uint64(datsize)
	ctxt.Syms.Lookup("runtime.pclntab", 0).Sect = sect
	ctxt.Syms.Lookup("runtime.epclntab", 0).Sect = sect
	for _, s := range data[SPCLNTAB] {
		datsize = aligndatsize(datsize, s)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
	}
	checkdatsize(ctxt, datsize, SRODATA)
	sect.Length = uint64(datsize) - sect.Vaddr

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if datsize != int64(uint32(datsize)) {
		Errorf(nil, "read-only data segment too large: %d", datsize)
	}

	for symn := SELFRXSECT; symn < SXREF; symn++ {
		datap = append(datap, data[symn]...)
	}

	dwarfgeneratedebugsyms(ctxt)

	var s *Symbol
	var i int
	for i, s = range dwarfp {
		if s.Type != SDWARFSECT {
			break
		}

		sect = addsection(&Segdwarf, s.Name, 04)
		sect.Align = 1
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		s.Sect = sect
		s.Type = SRODATA
		s.Value = int64(uint64(datsize) - sect.Vaddr)
		datsize += s.Size
		sect.Length = uint64(datsize) - sect.Vaddr
	}
	checkdatsize(ctxt, datsize, SDWARFSECT)

	if i < len(dwarfp) {
		sect = addsection(&Segdwarf, ".debug_info", 04)
		sect.Align = 1
		datsize = Rnd(datsize, int64(sect.Align))
		sect.Vaddr = uint64(datsize)
		for _, s := range dwarfp[i:] {
			if s.Type != SDWARFINFO {
				break
			}
			s.Sect = sect
			s.Type = SRODATA
			s.Value = int64(uint64(datsize) - sect.Vaddr)
			s.Attr |= AttrLocal
			datsize += s.Size
		}
		sect.Length = uint64(datsize) - sect.Vaddr
		checkdatsize(ctxt, datsize, SDWARFINFO)
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

func dodataSect(ctxt *Link, symn SymKind, syms []*Symbol) (result []*Symbol, maxAlign int32) {
	if Headtype == objabi.Hdarwin {
		// Some symbols may no longer belong in syms
		// due to movement in machosymorder.
		newSyms := make([]*Symbol, 0, len(syms))
		for _, s := range syms {
			if s.Type == symn {
				newSyms = append(newSyms, s)
			}
		}
		syms = newSyms
	}

	var head, tail *Symbol
	symsSort := make([]dataSortKey, 0, len(syms))
	for _, s := range syms {
		if s.Attr.OnList() {
			log.Fatalf("symbol %s listed multiple times", s.Name)
		}
		s.Attr |= AttrOnList
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
		if ctxt.DynlinkingGo() && Headtype == objabi.Hdarwin {
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
		case SELFGOT:
			// For ppc64, we want to interleave the .got and .toc sections
			// from input files. Both are type SELFGOT, so in that case
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

	if Iself && symn == SELFROSECT {
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
			rel.Align = int32(SysArch.RegSize)
			plt.Align = int32(SysArch.RegSize)
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
	if Iself || Buildmode == BuildmodePlugin || *flagBuildid == "" {
		return
	}

	sym := ctxt.Syms.Lookup("go.buildid", 0)
	sym.Attr |= AttrReachable
	// The \xff is invalid UTF-8, meant to make it less likely
	// to find one of these accidentally.
	data := "\xff Go build ID: " + strconv.Quote(*flagBuildid) + "\n \xff"
	sym.Type = STEXT
	sym.P = []byte(data)
	sym.Size = int64(len(sym.P))

	ctxt.Textp = append(ctxt.Textp, nil)
	copy(ctxt.Textp[1:], ctxt.Textp)
	ctxt.Textp[0] = sym
}

// assign addresses to text
func (ctxt *Link) textaddress() {
	addsection(&Segtext, ".text", 05)

	// Assign PCs in text segment.
	// Could parallelize, by assigning to text
	// and then letting threads copy down, but probably not worth it.
	sect := Segtext.Sections[0]

	sect.Align = int32(Funcalign)

	text := ctxt.Syms.Lookup("runtime.text", 0)
	text.Sect = sect

	if ctxt.DynlinkingGo() && Headtype == objabi.Hdarwin {
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
	for _, sym := range ctxt.Textp {
		sect, n, va = assignAddress(ctxt, sect, n, sym, va, false)

		trampoline(ctxt, sym) // resolve jumps, may add trampolines if jump too far

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
		newtextp := make([]*Symbol, 0, len(ctxt.Textp)+ntramps)
		i := 0
		for _, sym := range ctxt.Textp {
			for ; i < ntramps && ctxt.tramps[i].Value < sym.Value; i++ {
				newtextp = append(newtextp, ctxt.tramps[i])
			}
			newtextp = append(newtextp, sym)
		}
		newtextp = append(newtextp, ctxt.tramps[i:ntramps]...)

		ctxt.Textp = newtextp
	}
}

// assigns address for a text symbol, returns (possibly new) section, its number, and the address
// Note: once we have trampoline insertion support for external linking, this function
// will not need to create new text sections, and so no need to return sect and n.
func assignAddress(ctxt *Link, sect *Section, n int, sym *Symbol, va uint64, isTramp bool) (*Section, int, uint64) {
	sym.Sect = sect
	if sym.Type&SSUB != 0 {
		return sect, n, va
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

	funcsize := uint64(MINFUNC) // spacing required for findfunctab
	if sym.Size > MINFUNC {
		funcsize = uint64(sym.Size)
	}

	// On ppc64x a text section should not be larger than 2^26 bytes due to the size of
	// call target offset field in the bl instruction.  Splitting into smaller text
	// sections smaller than this limit allows the GNU linker to modify the long calls
	// appropriately.  The limit allows for the space needed for tables inserted by the linker.

	// If this function doesn't fit in the current text section, then create a new one.

	// Only break at outermost syms.

	if SysArch.InFamily(sys.PPC64) && sym.Outer == nil && Iself && Linkmode == LinkExternal && va-sect.Vaddr+funcsize+maxSizeTrampolinesPPC64(sym, isTramp) > 0x1c00000 {

		// Set the length for the previous text section
		sect.Length = va - sect.Vaddr

		// Create new section, set the starting Vaddr
		sect = addsection(&Segtext, ".text", 05)
		sect.Vaddr = va
		sym.Sect = sect

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
	if Headtype == objabi.Hnacl {
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
	if Headtype == objabi.Hwindows {
		Segdata.Fileoff = Segtext.Fileoff + uint64(Rnd(int64(Segtext.Length), PEFILEALIGN))
	}
	if Headtype == objabi.Hplan9 {
		Segdata.Fileoff = Segtext.Fileoff + Segtext.Filelen
	}
	var data *Section
	var noptr *Section
	var bss *Section
	var noptrbss *Section
	var vlen int64
	for i, s := range Segdata.Sections {
		if Iself && s.Name == ".tbss" {
			continue
		}
		vlen = int64(s.Length)
		if i+1 < len(Segdata.Sections) && !(Iself && Segdata.Sections[i+1].Name == ".tbss") {
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
	if Headtype == objabi.Hwindows {
		Segdwarf.Fileoff = Segdata.Fileoff + uint64(Rnd(int64(Segdata.Filelen), int64(PEFILEALIGN)))
	}
	for i, s := range Segdwarf.Sections {
		vlen = int64(s.Length)
		if i+1 < len(Segdwarf.Sections) {
			vlen = int64(Segdwarf.Sections[i+1].Vaddr - s.Vaddr)
		}
		s.Vaddr = va
		va += uint64(vlen)
		if Headtype == objabi.Hwindows {
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

	for _, sym := range dwarfp {
		if sym.Sect != nil {
			sym.Value += int64(sym.Sect.Vaddr)
		}
		for sub := sym.Sub; sub != nil; sub = sub.Sub {
			sub.Value += sym.Value
		}
	}

	if Buildmode == BuildmodeShared {
		s := ctxt.Syms.Lookup("go.link.abihashbytes", 0)
		sectSym := ctxt.Syms.Lookup(".note.go.abihash", 0)
		s.Sect = sectSym.Sect
		s.Value = int64(sectSym.Sect.Vaddr + 16)
	}

	ctxt.xdefine("runtime.text", STEXT, int64(text.Vaddr))
	ctxt.xdefine("runtime.etext", STEXT, int64(lasttext.Vaddr+lasttext.Length))

	// If there are multiple text sections, create runtime.text.n for
	// their section Vaddr, using n for index
	n := 1
	for _, sect := range Segtext.Sections[1:] {
		if sect.Name == ".text" {
			symname := fmt.Sprintf("runtime.text.%d", n)
			ctxt.xdefine(symname, STEXT, int64(sect.Vaddr))
			n++
		} else {
			break
		}
	}

	ctxt.xdefine("runtime.rodata", SRODATA, int64(rodata.Vaddr))
	ctxt.xdefine("runtime.erodata", SRODATA, int64(rodata.Vaddr+rodata.Length))
	ctxt.xdefine("runtime.types", SRODATA, int64(types.Vaddr))
	ctxt.xdefine("runtime.etypes", SRODATA, int64(types.Vaddr+types.Length))
	ctxt.xdefine("runtime.itablink", SRODATA, int64(itablink.Vaddr))
	ctxt.xdefine("runtime.eitablink", SRODATA, int64(itablink.Vaddr+itablink.Length))

	sym := ctxt.Syms.Lookup("runtime.gcdata", 0)
	sym.Attr |= AttrLocal
	ctxt.xdefine("runtime.egcdata", SRODATA, Symaddr(sym)+sym.Size)
	ctxt.Syms.Lookup("runtime.egcdata", 0).Sect = sym.Sect

	sym = ctxt.Syms.Lookup("runtime.gcbss", 0)
	sym.Attr |= AttrLocal
	ctxt.xdefine("runtime.egcbss", SRODATA, Symaddr(sym)+sym.Size)
	ctxt.Syms.Lookup("runtime.egcbss", 0).Sect = sym.Sect

	ctxt.xdefine("runtime.symtab", SRODATA, int64(symtab.Vaddr))
	ctxt.xdefine("runtime.esymtab", SRODATA, int64(symtab.Vaddr+symtab.Length))
	ctxt.xdefine("runtime.pclntab", SRODATA, int64(pclntab.Vaddr))
	ctxt.xdefine("runtime.epclntab", SRODATA, int64(pclntab.Vaddr+pclntab.Length))
	ctxt.xdefine("runtime.noptrdata", SNOPTRDATA, int64(noptr.Vaddr))
	ctxt.xdefine("runtime.enoptrdata", SNOPTRDATA, int64(noptr.Vaddr+noptr.Length))
	ctxt.xdefine("runtime.bss", SBSS, int64(bss.Vaddr))
	ctxt.xdefine("runtime.ebss", SBSS, int64(bss.Vaddr+bss.Length))
	ctxt.xdefine("runtime.data", SDATA, int64(data.Vaddr))
	ctxt.xdefine("runtime.edata", SDATA, int64(data.Vaddr+data.Length))
	ctxt.xdefine("runtime.noptrbss", SNOPTRBSS, int64(noptrbss.Vaddr))
	ctxt.xdefine("runtime.enoptrbss", SNOPTRBSS, int64(noptrbss.Vaddr+noptrbss.Length))
	ctxt.xdefine("runtime.end", SBSS, int64(Segdata.Vaddr+Segdata.Length))
}

// add a trampoline with symbol s (to be laid down after the current function)
func (ctxt *Link) AddTramp(s *Symbol) {
	s.Type = STEXT
	s.Attr |= AttrReachable
	s.Attr |= AttrOnList
	ctxt.tramps = append(ctxt.tramps, s)
	if *FlagDebugTramp > 0 && ctxt.Debugvlog > 0 {
		ctxt.Logf("trampoline %s inserted\n", s)
	}
}
