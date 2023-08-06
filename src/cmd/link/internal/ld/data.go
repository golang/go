// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/obj.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/span.c
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
	"bytes"
	"cmd/internal/gcprog"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/loadpe"
	"cmd/link/internal/sym"
	"compress/zlib"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// isRuntimeDepPkg reports whether pkg is the runtime package or its dependency.
func isRuntimeDepPkg(pkg string) bool {
	switch pkg {
	case "runtime",
		"sync/atomic",          // runtime may call to sync/atomic, due to go:linkname
		"internal/abi",         // used by reflectcall (and maybe more)
		"internal/bytealg",     // for IndexByte
		"internal/chacha8rand", // for rand
		"internal/cpu":         // for cpu features
		return true
	}
	return strings.HasPrefix(pkg, "runtime/internal/") && !strings.HasSuffix(pkg, "_test")
}

// Estimate the max size needed to hold any new trampolines created for this function. This
// is used to determine when the section can be split if it becomes too large, to ensure that
// the trampolines are in the same section as the function that uses them.
func maxSizeTrampolines(ctxt *Link, ldr *loader.Loader, s loader.Sym, isTramp bool) uint64 {
	// If thearch.Trampoline is nil, then trampoline support is not available on this arch.
	// A trampoline does not need any dependent trampolines.
	if thearch.Trampoline == nil || isTramp {
		return 0
	}

	n := uint64(0)
	relocs := ldr.Relocs(s)
	for ri := 0; ri < relocs.Count(); ri++ {
		r := relocs.At(ri)
		if r.Type().IsDirectCallOrJump() {
			n++
		}
	}

	switch {
	case ctxt.IsARM():
		return n * 20 // Trampolines in ARM range from 3 to 5 instructions.
	case ctxt.IsARM64():
		return n * 12 // Trampolines in ARM64 are 3 instructions.
	case ctxt.IsPPC64():
		return n * 16 // Trampolines in PPC64 are 4 instructions.
	case ctxt.IsRISCV64():
		return n * 8 // Trampolines in RISCV64 are 2 instructions.
	}
	panic("unreachable")
}

// Detect too-far jumps in function s, and add trampolines if necessary.
// ARM, PPC64, PPC64LE and RISCV64 support trampoline insertion for internal
// and external linking. On PPC64 and PPC64LE the text sections might be split
// but will still insert trampolines where necessary.
func trampoline(ctxt *Link, s loader.Sym) {
	if thearch.Trampoline == nil {
		return // no need or no support of trampolines on this arch
	}

	ldr := ctxt.loader
	relocs := ldr.Relocs(s)
	for ri := 0; ri < relocs.Count(); ri++ {
		r := relocs.At(ri)
		rt := r.Type()
		if !rt.IsDirectCallOrJump() && !isPLTCall(rt) {
			continue
		}
		rs := r.Sym()
		if !ldr.AttrReachable(rs) || ldr.SymType(rs) == sym.Sxxx {
			continue // something is wrong. skip it here and we'll emit a better error later
		}

		if ldr.SymValue(rs) == 0 && ldr.SymType(rs) != sym.SDYNIMPORT && ldr.SymType(rs) != sym.SUNDEFEXT {
			// Symbols in the same package are laid out together.
			// Except that if SymPkg(s) == "", it is a host object symbol
			// which may call an external symbol via PLT.
			if ldr.SymPkg(s) != "" && ldr.SymPkg(rs) == ldr.SymPkg(s) {
				// RISC-V is only able to reach +/-1MiB via a JAL instruction.
				// We need to generate a trampoline when an address is
				// currently unknown.
				if !ctxt.Target.IsRISCV64() {
					continue
				}
			}
			// Runtime packages are laid out together.
			if isRuntimeDepPkg(ldr.SymPkg(s)) && isRuntimeDepPkg(ldr.SymPkg(rs)) {
				continue
			}
		}
		thearch.Trampoline(ctxt, ldr, ri, rs, s)
	}
}

// whether rt is a (host object) relocation that will be turned into
// a call to PLT.
func isPLTCall(rt objabi.RelocType) bool {
	const pcrel = 1
	switch rt {
	// ARM64
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_CALL26),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_AARCH64_JUMP26),
		objabi.MachoRelocOffset + MACHO_ARM64_RELOC_BRANCH26*2 + pcrel:
		return true

	// ARM
	case objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_CALL),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_PC24),
		objabi.ElfRelocOffset + objabi.RelocType(elf.R_ARM_JUMP24):
		return true
	}
	// TODO: other architectures.
	return false
}

// FoldSubSymbolOffset computes the offset of symbol s to its top-level outer
// symbol. Returns the top-level symbol and the offset.
// This is used in generating external relocations.
func FoldSubSymbolOffset(ldr *loader.Loader, s loader.Sym) (loader.Sym, int64) {
	outer := ldr.OuterSym(s)
	off := int64(0)
	if outer != 0 {
		off += ldr.SymValue(s) - ldr.SymValue(outer)
		s = outer
	}
	return s, off
}

// relocsym resolve relocations in "s", updating the symbol's content
// in "P".
// The main loop walks through the list of relocations attached to "s"
// and resolves them where applicable. Relocations are often
// architecture-specific, requiring calls into the 'archreloc' and/or
// 'archrelocvariant' functions for the architecture. When external
// linking is in effect, it may not be  possible to completely resolve
// the address/offset for a symbol, in which case the goal is to lay
// the groundwork for turning a given relocation into an external reloc
// (to be applied by the external linker). For more on how relocations
// work in general, see
//
//	"Linkers and Loaders", by John R. Levine (Morgan Kaufmann, 1999), ch. 7
//
// This is a performance-critical function for the linker; be careful
// to avoid introducing unnecessary allocations in the main loop.
func (st *relocSymState) relocsym(s loader.Sym, P []byte) {
	ldr := st.ldr
	relocs := ldr.Relocs(s)
	if relocs.Count() == 0 {
		return
	}
	target := st.target
	syms := st.syms
	nExtReloc := 0 // number of external relocations
	for ri := 0; ri < relocs.Count(); ri++ {
		r := relocs.At(ri)
		off := r.Off()
		siz := int32(r.Siz())
		rs := r.Sym()
		rt := r.Type()
		weak := r.Weak()
		if off < 0 || off+siz > int32(len(P)) {
			rname := ""
			if rs != 0 {
				rname = ldr.SymName(rs)
			}
			st.err.Errorf(s, "invalid relocation %s: %d+%d not in [%d,%d)", rname, off, siz, 0, len(P))
			continue
		}
		if siz == 0 { // informational relocation - no work to do
			continue
		}

		var rst sym.SymKind
		if rs != 0 {
			rst = ldr.SymType(rs)
		}

		if rs != 0 && (rst == sym.Sxxx || rst == sym.SXREF) {
			// When putting the runtime but not main into a shared library
			// these symbols are undefined and that's OK.
			if target.IsShared() || target.IsPlugin() {
				if ldr.SymName(rs) == "main.main" || (!target.IsPlugin() && ldr.SymName(rs) == "main..inittask") {
					sb := ldr.MakeSymbolUpdater(rs)
					sb.SetType(sym.SDYNIMPORT)
				} else if strings.HasPrefix(ldr.SymName(rs), "go:info.") {
					// Skip go.info symbols. They are only needed to communicate
					// DWARF info between the compiler and linker.
					continue
				}
			} else if target.IsPPC64() && ldr.SymName(rs) == ".TOC." {
				// TOC symbol doesn't have a type but we do assign a value
				// (see the address pass) and we can resolve it.
				// TODO: give it a type.
			} else {
				st.err.errorUnresolved(ldr, s, rs)
				continue
			}
		}

		if rt >= objabi.ElfRelocOffset {
			continue
		}

		// We need to be able to reference dynimport symbols when linking against
		// shared libraries, and AIX, Darwin, OpenBSD and Solaris always need it.
		if !target.IsAIX() && !target.IsDarwin() && !target.IsSolaris() && !target.IsOpenbsd() && rs != 0 && rst == sym.SDYNIMPORT && !target.IsDynlinkingGo() && !ldr.AttrSubSymbol(rs) {
			if !(target.IsPPC64() && target.IsExternal() && ldr.SymName(rs) == ".TOC.") {
				st.err.Errorf(s, "unhandled relocation for %s (type %d (%s) rtype %d (%s))", ldr.SymName(rs), rst, rst, rt, sym.RelocName(target.Arch, rt))
			}
		}
		if rs != 0 && rst != sym.STLSBSS && !weak && rt != objabi.R_METHODOFF && !ldr.AttrReachable(rs) {
			st.err.Errorf(s, "unreachable sym in relocation: %s", ldr.SymName(rs))
		}

		var rv sym.RelocVariant
		if target.IsPPC64() || target.IsS390X() {
			rv = ldr.RelocVariant(s, ri)
		}

		// TODO(mundaym): remove this special case - see issue 14218.
		if target.IsS390X() {
			switch rt {
			case objabi.R_PCRELDBL:
				rt = objabi.R_PCREL
				rv = sym.RV_390_DBL
			case objabi.R_CALL:
				rv = sym.RV_390_DBL
			}
		}

		var o int64
		switch rt {
		default:
			switch siz {
			default:
				st.err.Errorf(s, "bad reloc size %#x for %s", uint32(siz), ldr.SymName(rs))
			case 1:
				o = int64(P[off])
			case 2:
				o = int64(target.Arch.ByteOrder.Uint16(P[off:]))
			case 4:
				o = int64(target.Arch.ByteOrder.Uint32(P[off:]))
			case 8:
				o = int64(target.Arch.ByteOrder.Uint64(P[off:]))
			}
			out, n, ok := thearch.Archreloc(target, ldr, syms, r, s, o)
			if target.IsExternal() {
				nExtReloc += n
			}
			if ok {
				o = out
			} else {
				st.err.Errorf(s, "unknown reloc to %v: %d (%s)", ldr.SymName(rs), rt, sym.RelocName(target.Arch, rt))
			}
		case objabi.R_TLS_LE:
			if target.IsExternal() && target.IsElf() {
				nExtReloc++
				o = 0
				if !target.IsAMD64() {
					o = r.Add()
				}
				break
			}

			if target.IsElf() && target.IsARM() {
				// On ELF ARM, the thread pointer is 8 bytes before
				// the start of the thread-local data block, so add 8
				// to the actual TLS offset (r->sym->value).
				// This 8 seems to be a fundamental constant of
				// ELF on ARM (or maybe Glibc on ARM); it is not
				// related to the fact that our own TLS storage happens
				// to take up 8 bytes.
				o = 8 + ldr.SymValue(rs)
			} else if target.IsElf() || target.IsPlan9() || target.IsDarwin() {
				o = int64(syms.Tlsoffset) + r.Add()
			} else if target.IsWindows() {
				o = r.Add()
			} else {
				log.Fatalf("unexpected R_TLS_LE relocation for %v", target.HeadType)
			}
		case objabi.R_TLS_IE:
			if target.IsExternal() && target.IsElf() {
				nExtReloc++
				o = 0
				if !target.IsAMD64() {
					o = r.Add()
				}
				if target.Is386() {
					nExtReloc++ // need two ELF relocations on 386, see ../x86/asm.go:elfreloc1
				}
				break
			}
			if target.IsPIE() && target.IsElf() {
				// We are linking the final executable, so we
				// can optimize any TLS IE relocation to LE.
				if thearch.TLSIEtoLE == nil {
					log.Fatalf("internal linking of TLS IE not supported on %v", target.Arch.Family)
				}
				thearch.TLSIEtoLE(P, int(off), int(siz))
				o = int64(syms.Tlsoffset)
			} else {
				log.Fatalf("cannot handle R_TLS_IE (sym %s) when linking internally", ldr.SymName(s))
			}
		case objabi.R_ADDR, objabi.R_PEIMAGEOFF:
			if weak && !ldr.AttrReachable(rs) {
				// Redirect it to runtime.unreachableMethod, which will throw if called.
				rs = syms.unreachableMethod
			}
			if target.IsExternal() {
				nExtReloc++

				// set up addend for eventual relocation via outer symbol.
				rs := rs
				rs, off := FoldSubSymbolOffset(ldr, rs)
				xadd := r.Add() + off
				rst := ldr.SymType(rs)
				if rst != sym.SHOSTOBJ && rst != sym.SDYNIMPORT && rst != sym.SUNDEFEXT && ldr.SymSect(rs) == nil {
					st.err.Errorf(s, "missing section for relocation target %s", ldr.SymName(rs))
				}

				o = xadd
				if target.IsElf() {
					if target.IsAMD64() {
						o = 0
					}
				} else if target.IsDarwin() {
					if ldr.SymType(s).IsDWARF() {
						// We generally use symbol-targeted relocations.
						// DWARF tools seem to only handle section-targeted relocations,
						// so generate section-targeted relocations in DWARF sections.
						// See also machoreloc1.
						o += ldr.SymValue(rs)
					}
				} else if target.IsWindows() {
					// nothing to do
				} else if target.IsAIX() {
					o = ldr.SymValue(rs) + xadd
				} else {
					st.err.Errorf(s, "unhandled pcrel relocation to %s on %v", ldr.SymName(rs), target.HeadType)
				}

				break
			}

			// On AIX, a second relocation must be done by the loader,
			// as section addresses can change once loaded.
			// The "default" symbol address is still needed by the loader so
			// the current relocation can't be skipped.
			if target.IsAIX() && rst != sym.SDYNIMPORT {
				// It's not possible to make a loader relocation in a
				// symbol which is not inside .data section.
				// FIXME: It should be forbidden to have R_ADDR from a
				// symbol which isn't in .data. However, as .text has the
				// same address once loaded, this is possible.
				if ldr.SymSect(s).Seg == &Segdata {
					Xcoffadddynrel(target, ldr, syms, s, r, ri)
				}
			}

			o = ldr.SymValue(rs) + r.Add()
			if rt == objabi.R_PEIMAGEOFF {
				// The R_PEIMAGEOFF offset is a RVA, so subtract
				// the base address for the executable.
				o -= PEBASE
			}

			// On amd64, 4-byte offsets will be sign-extended, so it is impossible to
			// access more than 2GB of static data; fail at link time is better than
			// fail at runtime. See https://golang.org/issue/7980.
			// Instead of special casing only amd64, we treat this as an error on all
			// 64-bit architectures so as to be future-proof.
			if int32(o) < 0 && target.Arch.PtrSize > 4 && siz == 4 {
				st.err.Errorf(s, "non-pc-relative relocation address for %s is too big: %#x (%#x + %#x)", ldr.SymName(rs), uint64(o), ldr.SymValue(rs), r.Add())
				errorexit()
			}
		case objabi.R_DWARFSECREF:
			if ldr.SymSect(rs) == nil {
				st.err.Errorf(s, "missing DWARF section for relocation target %s", ldr.SymName(rs))
			}

			if target.IsExternal() {
				// On most platforms, the external linker needs to adjust DWARF references
				// as it combines DWARF sections. However, on Darwin, dsymutil does the
				// DWARF linking, and it understands how to follow section offsets.
				// Leaving in the relocation records confuses it (see
				// https://golang.org/issue/22068) so drop them for Darwin.
				if !target.IsDarwin() {
					nExtReloc++
				}

				xadd := r.Add() + ldr.SymValue(rs) - int64(ldr.SymSect(rs).Vaddr)

				o = xadd
				if target.IsElf() && target.IsAMD64() {
					o = 0
				}
				break
			}
			o = ldr.SymValue(rs) + r.Add() - int64(ldr.SymSect(rs).Vaddr)
		case objabi.R_METHODOFF:
			if !ldr.AttrReachable(rs) {
				// Set it to a sentinel value. The runtime knows this is not pointing to
				// anything valid.
				o = -1
				break
			}
			fallthrough
		case objabi.R_ADDROFF:
			if weak && !ldr.AttrReachable(rs) {
				continue
			}
			sect := ldr.SymSect(rs)
			if sect == nil {
				if rst == sym.SDYNIMPORT {
					st.err.Errorf(s, "cannot target DYNIMPORT sym in section-relative reloc: %s", ldr.SymName(rs))
				} else if rst == sym.SUNDEFEXT {
					st.err.Errorf(s, "undefined symbol in relocation: %s", ldr.SymName(rs))
				} else {
					st.err.Errorf(s, "missing section for relocation target %s", ldr.SymName(rs))
				}
				continue
			}

			// The method offset tables using this relocation expect the offset to be relative
			// to the start of the first text section, even if there are multiple.
			if sect.Name == ".text" {
				o = ldr.SymValue(rs) - int64(Segtext.Sections[0].Vaddr) + r.Add()
			} else {
				o = ldr.SymValue(rs) - int64(ldr.SymSect(rs).Vaddr) + r.Add()
			}

		case objabi.R_ADDRCUOFF:
			// debug_range and debug_loc elements use this relocation type to get an
			// offset from the start of the compile unit.
			o = ldr.SymValue(rs) + r.Add() - ldr.SymValue(loader.Sym(ldr.SymUnit(rs).Textp[0]))

		// r.Sym() can be 0 when CALL $(constant) is transformed from absolute PC to relative PC call.
		case objabi.R_GOTPCREL:
			if target.IsDynlinkingGo() && target.IsDarwin() && rs != 0 {
				nExtReloc++
				o = r.Add()
				break
			}
			if target.Is386() && target.IsExternal() && target.IsELF {
				nExtReloc++ // need two ELF relocations on 386, see ../x86/asm.go:elfreloc1
			}
			fallthrough
		case objabi.R_CALL, objabi.R_PCREL:
			if target.IsExternal() && rs != 0 && rst == sym.SUNDEFEXT {
				// pass through to the external linker.
				nExtReloc++
				o = 0
				break
			}
			if target.IsExternal() && rs != 0 && (ldr.SymSect(rs) != ldr.SymSect(s) || rt == objabi.R_GOTPCREL) {
				nExtReloc++

				// set up addend for eventual relocation via outer symbol.
				rs := rs
				rs, off := FoldSubSymbolOffset(ldr, rs)
				xadd := r.Add() + off - int64(siz) // relative to address after the relocated chunk
				rst := ldr.SymType(rs)
				if rst != sym.SHOSTOBJ && rst != sym.SDYNIMPORT && ldr.SymSect(rs) == nil {
					st.err.Errorf(s, "missing section for relocation target %s", ldr.SymName(rs))
				}

				o = xadd
				if target.IsElf() {
					if target.IsAMD64() {
						o = 0
					}
				} else if target.IsDarwin() {
					if rt == objabi.R_CALL {
						if target.IsExternal() && rst == sym.SDYNIMPORT {
							if target.IsAMD64() {
								// AMD64 dynamic relocations are relative to the end of the relocation.
								o += int64(siz)
							}
						} else {
							if rst != sym.SHOSTOBJ {
								o += int64(uint64(ldr.SymValue(rs)) - ldr.SymSect(rs).Vaddr)
							}
							o -= int64(off) // relative to section offset, not symbol
						}
					} else {
						o += int64(siz)
					}
				} else if target.IsWindows() && target.IsAMD64() { // only amd64 needs PCREL
					// PE/COFF's PC32 relocation uses the address after the relocated
					// bytes as the base. Compensate by skewing the addend.
					o += int64(siz)
				} else {
					st.err.Errorf(s, "unhandled pcrel relocation to %s on %v", ldr.SymName(rs), target.HeadType)
				}

				break
			}

			o = 0
			if rs != 0 {
				o = ldr.SymValue(rs)
			}

			o += r.Add() - (ldr.SymValue(s) + int64(off) + int64(siz))
		case objabi.R_SIZE:
			o = ldr.SymSize(rs) + r.Add()

		case objabi.R_XCOFFREF:
			if !target.IsAIX() {
				st.err.Errorf(s, "find XCOFF R_REF on non-XCOFF files")
			}
			if !target.IsExternal() {
				st.err.Errorf(s, "find XCOFF R_REF with internal linking")
			}
			nExtReloc++
			continue

		case objabi.R_DWARFFILEREF:
			// We don't renumber files in dwarf.go:writelines anymore.
			continue

		case objabi.R_CONST:
			o = r.Add()

		case objabi.R_GOTOFF:
			o = ldr.SymValue(rs) + r.Add() - ldr.SymValue(syms.GOT)
		}

		if target.IsPPC64() || target.IsS390X() {
			if rv != sym.RV_NONE {
				o = thearch.Archrelocvariant(target, ldr, r, rv, s, o, P)
			}
		}

		switch siz {
		default:
			st.err.Errorf(s, "bad reloc size %#x for %s", uint32(siz), ldr.SymName(rs))
		case 1:
			P[off] = byte(int8(o))
		case 2:
			if (rt == objabi.R_PCREL || rt == objabi.R_CALL) && o != int64(int16(o)) {
				st.err.Errorf(s, "pc-relative relocation address for %s is too big: %#x", ldr.SymName(rs), o)
			} else if o != int64(int16(o)) && o != int64(uint16(o)) {
				st.err.Errorf(s, "non-pc-relative relocation address for %s is too big: %#x", ldr.SymName(rs), uint64(o))
			}
			target.Arch.ByteOrder.PutUint16(P[off:], uint16(o))
		case 4:
			if (rt == objabi.R_PCREL || rt == objabi.R_CALL) && o != int64(int32(o)) {
				st.err.Errorf(s, "pc-relative relocation address for %s is too big: %#x", ldr.SymName(rs), o)
			} else if o != int64(int32(o)) && o != int64(uint32(o)) {
				st.err.Errorf(s, "non-pc-relative relocation address for %s is too big: %#x", ldr.SymName(rs), uint64(o))
			}
			target.Arch.ByteOrder.PutUint32(P[off:], uint32(o))
		case 8:
			target.Arch.ByteOrder.PutUint64(P[off:], uint64(o))
		}
	}
	if target.IsExternal() {
		// We'll stream out the external relocations in asmb2 (e.g. elfrelocsect)
		// and we only need the count here.
		atomic.AddUint32(&ldr.SymSect(s).Relcount, uint32(nExtReloc))
	}
}

// Convert a Go relocation to an external relocation.
func extreloc(ctxt *Link, ldr *loader.Loader, s loader.Sym, r loader.Reloc) (loader.ExtReloc, bool) {
	var rr loader.ExtReloc
	target := &ctxt.Target
	siz := int32(r.Siz())
	if siz == 0 { // informational relocation - no work to do
		return rr, false
	}

	rt := r.Type()
	if rt >= objabi.ElfRelocOffset {
		return rr, false
	}
	rr.Type = rt
	rr.Size = uint8(siz)

	// TODO(mundaym): remove this special case - see issue 14218.
	if target.IsS390X() {
		switch rt {
		case objabi.R_PCRELDBL:
			rt = objabi.R_PCREL
		}
	}

	switch rt {
	default:
		return thearch.Extreloc(target, ldr, r, s)

	case objabi.R_TLS_LE, objabi.R_TLS_IE:
		if target.IsElf() {
			rs := r.Sym()
			rr.Xsym = rs
			if rr.Xsym == 0 {
				rr.Xsym = ctxt.Tlsg
			}
			rr.Xadd = r.Add()
			break
		}
		return rr, false

	case objabi.R_ADDR, objabi.R_PEIMAGEOFF:
		// set up addend for eventual relocation via outer symbol.
		rs := r.Sym()
		if r.Weak() && !ldr.AttrReachable(rs) {
			rs = ctxt.ArchSyms.unreachableMethod
		}
		rs, off := FoldSubSymbolOffset(ldr, rs)
		rr.Xadd = r.Add() + off
		rr.Xsym = rs

	case objabi.R_DWARFSECREF:
		// On most platforms, the external linker needs to adjust DWARF references
		// as it combines DWARF sections. However, on Darwin, dsymutil does the
		// DWARF linking, and it understands how to follow section offsets.
		// Leaving in the relocation records confuses it (see
		// https://golang.org/issue/22068) so drop them for Darwin.
		if target.IsDarwin() {
			return rr, false
		}
		rs := r.Sym()
		rr.Xsym = loader.Sym(ldr.SymSect(rs).Sym)
		rr.Xadd = r.Add() + ldr.SymValue(rs) - int64(ldr.SymSect(rs).Vaddr)

	// r.Sym() can be 0 when CALL $(constant) is transformed from absolute PC to relative PC call.
	case objabi.R_GOTPCREL, objabi.R_CALL, objabi.R_PCREL:
		rs := r.Sym()
		if rt == objabi.R_GOTPCREL && target.IsDynlinkingGo() && target.IsDarwin() && rs != 0 {
			rr.Xadd = r.Add()
			rr.Xadd -= int64(siz) // relative to address after the relocated chunk
			rr.Xsym = rs
			break
		}
		if rs != 0 && ldr.SymType(rs) == sym.SUNDEFEXT {
			// pass through to the external linker.
			rr.Xadd = 0
			if target.IsElf() {
				rr.Xadd -= int64(siz)
			}
			rr.Xsym = rs
			break
		}
		if rs != 0 && (ldr.SymSect(rs) != ldr.SymSect(s) || rt == objabi.R_GOTPCREL) {
			// set up addend for eventual relocation via outer symbol.
			rs := rs
			rs, off := FoldSubSymbolOffset(ldr, rs)
			rr.Xadd = r.Add() + off
			rr.Xadd -= int64(siz) // relative to address after the relocated chunk
			rr.Xsym = rs
			break
		}
		return rr, false

	case objabi.R_XCOFFREF:
		return ExtrelocSimple(ldr, r), true

	// These reloc types don't need external relocations.
	case objabi.R_ADDROFF, objabi.R_METHODOFF, objabi.R_ADDRCUOFF,
		objabi.R_SIZE, objabi.R_CONST, objabi.R_GOTOFF:
		return rr, false
	}
	return rr, true
}

// ExtrelocSimple creates a simple external relocation from r, with the same
// symbol and addend.
func ExtrelocSimple(ldr *loader.Loader, r loader.Reloc) loader.ExtReloc {
	var rr loader.ExtReloc
	rs := r.Sym()
	rr.Xsym = rs
	rr.Xadd = r.Add()
	rr.Type = r.Type()
	rr.Size = r.Siz()
	return rr
}

// ExtrelocViaOuterSym creates an external relocation from r targeting the
// outer symbol and folding the subsymbol's offset into the addend.
func ExtrelocViaOuterSym(ldr *loader.Loader, r loader.Reloc, s loader.Sym) loader.ExtReloc {
	// set up addend for eventual relocation via outer symbol.
	var rr loader.ExtReloc
	rs := r.Sym()
	rs, off := FoldSubSymbolOffset(ldr, rs)
	rr.Xadd = r.Add() + off
	rst := ldr.SymType(rs)
	if rst != sym.SHOSTOBJ && rst != sym.SDYNIMPORT && rst != sym.SUNDEFEXT && ldr.SymSect(rs) == nil {
		ldr.Errorf(s, "missing section for %s", ldr.SymName(rs))
	}
	rr.Xsym = rs
	rr.Type = r.Type()
	rr.Size = r.Siz()
	return rr
}

// relocSymState hold state information needed when making a series of
// successive calls to relocsym(). The items here are invariant
// (meaning that they are set up once initially and then don't change
// during the execution of relocsym), with the exception of a slice
// used to facilitate batch allocation of external relocations. Calls
// to relocsym happen in parallel; the assumption is that each
// parallel thread will have its own state object.
type relocSymState struct {
	target *Target
	ldr    *loader.Loader
	err    *ErrorReporter
	syms   *ArchSyms
}

// makeRelocSymState creates a relocSymState container object to
// pass to relocsym(). If relocsym() calls happen in parallel,
// each parallel thread should have its own state object.
func (ctxt *Link) makeRelocSymState() *relocSymState {
	return &relocSymState{
		target: &ctxt.Target,
		ldr:    ctxt.loader,
		err:    &ctxt.ErrorReporter,
		syms:   &ctxt.ArchSyms,
	}
}

// windynrelocsym examines a text symbol 's' and looks for relocations
// from it that correspond to references to symbols defined in DLLs,
// then fixes up those relocations as needed. A reference to a symbol
// XYZ from some DLL will fall into one of two categories: an indirect
// ref via "__imp_XYZ", or a direct ref to "XYZ". Here's an example of
// an indirect ref (this is an excerpt from objdump -ldr):
//
//	     1c1: 48 89 c6                     	movq	%rax, %rsi
//	     1c4: ff 15 00 00 00 00            	callq	*(%rip)
//			00000000000001c6:  IMAGE_REL_AMD64_REL32	__imp__errno
//
// In the assembly above, the code loads up the value of __imp_errno
// and then does an indirect call to that value.
//
// Here is what a direct reference might look like:
//
//	     137: e9 20 06 00 00               	jmp	0x75c <pow+0x75c>
//	     13c: e8 00 00 00 00               	callq	0x141 <pow+0x141>
//			000000000000013d:  IMAGE_REL_AMD64_REL32	_errno
//
// The assembly below dispenses with the import symbol and just makes
// a direct call to _errno.
//
// The code below handles indirect refs by redirecting the target of
// the relocation from "__imp_XYZ" to "XYZ" (since the latter symbol
// is what the Windows loader is expected to resolve). For direct refs
// the call is redirected to a stub, where the stub first loads the
// symbol and then direct an indirect call to that value.
//
// Note that for a given symbol (as above) it is perfectly legal to
// have both direct and indirect references.
func windynrelocsym(ctxt *Link, rel *loader.SymbolBuilder, s loader.Sym) error {
	var su *loader.SymbolBuilder
	relocs := ctxt.loader.Relocs(s)
	for ri := 0; ri < relocs.Count(); ri++ {
		r := relocs.At(ri)
		if r.IsMarker() {
			continue // skip marker relocations
		}
		targ := r.Sym()
		if targ == 0 {
			continue
		}
		if !ctxt.loader.AttrReachable(targ) {
			if r.Weak() {
				continue
			}
			return fmt.Errorf("dynamic relocation to unreachable symbol %s",
				ctxt.loader.SymName(targ))
		}
		tgot := ctxt.loader.SymGot(targ)
		if tgot == loadpe.RedirectToDynImportGotToken {

			// Consistency check: name should be __imp_X
			sname := ctxt.loader.SymName(targ)
			if !strings.HasPrefix(sname, "__imp_") {
				return fmt.Errorf("internal error in windynrelocsym: redirect GOT token applied to non-import symbol %s", sname)
			}

			// Locate underlying symbol (which originally had type
			// SDYNIMPORT but has since been retyped to SWINDOWS).
			ds, err := loadpe.LookupBaseFromImport(targ, ctxt.loader, ctxt.Arch)
			if err != nil {
				return err
			}
			dstyp := ctxt.loader.SymType(ds)
			if dstyp != sym.SWINDOWS {
				return fmt.Errorf("internal error in windynrelocsym: underlying sym for %q has wrong type %s", sname, dstyp.String())
			}

			// Redirect relocation to the dynimport.
			r.SetSym(ds)
			continue
		}

		tplt := ctxt.loader.SymPlt(targ)
		if tplt == loadpe.CreateImportStubPltToken {

			// Consistency check: don't want to see both PLT and GOT tokens.
			if tgot != -1 {
				return fmt.Errorf("internal error in windynrelocsym: invalid GOT setting %d for reloc to %s", tgot, ctxt.loader.SymName(targ))
			}

			// make dynimport JMP table for PE object files.
			tplt := int32(rel.Size())
			ctxt.loader.SetPlt(targ, tplt)

			if su == nil {
				su = ctxt.loader.MakeSymbolUpdater(s)
			}
			r.SetSym(rel.Sym())
			r.SetAdd(int64(tplt))

			// jmp *addr
			switch ctxt.Arch.Family {
			default:
				return fmt.Errorf("internal error in windynrelocsym: unsupported arch %v", ctxt.Arch.Family)
			case sys.I386:
				rel.AddUint8(0xff)
				rel.AddUint8(0x25)
				rel.AddAddrPlus(ctxt.Arch, targ, 0)
				rel.AddUint8(0x90)
				rel.AddUint8(0x90)
			case sys.AMD64:
				rel.AddUint8(0xff)
				rel.AddUint8(0x24)
				rel.AddUint8(0x25)
				rel.AddAddrPlus4(ctxt.Arch, targ, 0)
				rel.AddUint8(0x90)
			}
		} else if tplt >= 0 {
			if su == nil {
				su = ctxt.loader.MakeSymbolUpdater(s)
			}
			r.SetSym(rel.Sym())
			r.SetAdd(int64(tplt))
		}
	}
	return nil
}

// windynrelocsyms generates jump table to C library functions that will be
// added later. windynrelocsyms writes the table into .rel symbol.
func (ctxt *Link) windynrelocsyms() {
	if !(ctxt.IsWindows() && iscgo && ctxt.IsInternal()) {
		return
	}

	rel := ctxt.loader.CreateSymForUpdate(".rel", 0)
	rel.SetType(sym.STEXT)

	for _, s := range ctxt.Textp {
		if err := windynrelocsym(ctxt, rel, s); err != nil {
			ctxt.Errorf(s, "%v", err)
		}
	}

	ctxt.Textp = append(ctxt.Textp, rel.Sym())
}

func dynrelocsym(ctxt *Link, s loader.Sym) {
	target := &ctxt.Target
	ldr := ctxt.loader
	syms := &ctxt.ArchSyms
	relocs := ldr.Relocs(s)
	for ri := 0; ri < relocs.Count(); ri++ {
		r := relocs.At(ri)
		if r.IsMarker() {
			continue // skip marker relocations
		}
		rSym := r.Sym()
		if r.Weak() && !ldr.AttrReachable(rSym) {
			continue
		}
		if ctxt.BuildMode == BuildModePIE && ctxt.LinkMode == LinkInternal {
			// It's expected that some relocations will be done
			// later by relocsym (R_TLS_LE, R_ADDROFF), so
			// don't worry if Adddynrel returns false.
			thearch.Adddynrel(target, ldr, syms, s, r, ri)
			continue
		}

		if rSym != 0 && ldr.SymType(rSym) == sym.SDYNIMPORT || r.Type() >= objabi.ElfRelocOffset {
			if rSym != 0 && !ldr.AttrReachable(rSym) {
				ctxt.Errorf(s, "dynamic relocation to unreachable symbol %s", ldr.SymName(rSym))
			}
			if !thearch.Adddynrel(target, ldr, syms, s, r, ri) {
				ctxt.Errorf(s, "unsupported dynamic relocation for symbol %s (type=%d (%s) stype=%d (%s))", ldr.SymName(rSym), r.Type(), sym.RelocName(ctxt.Arch, r.Type()), ldr.SymType(rSym), ldr.SymType(rSym))
			}
		}
	}
}

func (state *dodataState) dynreloc(ctxt *Link) {
	if ctxt.HeadType == objabi.Hwindows {
		return
	}
	// -d suppresses dynamic loader format, so we may as well not
	// compute these sections or mark their symbols as reachable.
	if *FlagD {
		return
	}

	for _, s := range ctxt.Textp {
		dynrelocsym(ctxt, s)
	}
	for _, syms := range state.data {
		for _, s := range syms {
			dynrelocsym(ctxt, s)
		}
	}
	if ctxt.IsELF {
		elfdynhash(ctxt)
	}
}

func CodeblkPad(ctxt *Link, out *OutBuf, addr int64, size int64, pad []byte) {
	writeBlocks(ctxt, out, ctxt.outSem, ctxt.loader, ctxt.Textp, addr, size, pad)
}

const blockSize = 1 << 20 // 1MB chunks written at a time.

// writeBlocks writes a specified chunk of symbols to the output buffer. It
// breaks the write up into ≥blockSize chunks to write them out, and schedules
// as many goroutines as necessary to accomplish this task. This call then
// blocks, waiting on the writes to complete. Note that we use the sem parameter
// to limit the number of concurrent writes taking place.
func writeBlocks(ctxt *Link, out *OutBuf, sem chan int, ldr *loader.Loader, syms []loader.Sym, addr, size int64, pad []byte) {
	for i, s := range syms {
		if ldr.SymValue(s) >= addr && !ldr.AttrSubSymbol(s) {
			syms = syms[i:]
			break
		}
	}

	var wg sync.WaitGroup
	max, lastAddr, written := int64(blockSize), addr+size, int64(0)
	for addr < lastAddr {
		// Find the last symbol we'd write.
		idx := -1
		for i, s := range syms {
			if ldr.AttrSubSymbol(s) {
				continue
			}

			// If the next symbol's size would put us out of bounds on the total length,
			// stop looking.
			end := ldr.SymValue(s) + ldr.SymSize(s)
			if end > lastAddr {
				break
			}

			// We're gonna write this symbol.
			idx = i

			// If we cross over the max size, we've got enough symbols.
			if end > addr+max {
				break
			}
		}

		// If we didn't find any symbols to write, we're done here.
		if idx < 0 {
			break
		}

		// Compute the length to write, including padding.
		// We need to write to the end address (lastAddr), or the next symbol's
		// start address, whichever comes first. If there is no more symbols,
		// just write to lastAddr. This ensures we don't leave holes between the
		// blocks or at the end.
		length := int64(0)
		if idx+1 < len(syms) {
			// Find the next top-level symbol.
			// Skip over sub symbols so we won't split a container symbol
			// into two blocks.
			next := syms[idx+1]
			for ldr.AttrSubSymbol(next) {
				idx++
				next = syms[idx+1]
			}
			length = ldr.SymValue(next) - addr
		}
		if length == 0 || length > lastAddr-addr {
			length = lastAddr - addr
		}

		// Start the block output operator.
		if o, err := out.View(uint64(out.Offset() + written)); err == nil {
			sem <- 1
			wg.Add(1)
			go func(o *OutBuf, ldr *loader.Loader, syms []loader.Sym, addr, size int64, pad []byte) {
				writeBlock(ctxt, o, ldr, syms, addr, size, pad)
				wg.Done()
				<-sem
			}(o, ldr, syms, addr, length, pad)
		} else { // output not mmaped, don't parallelize.
			writeBlock(ctxt, out, ldr, syms, addr, length, pad)
		}

		// Prepare for the next loop.
		if idx != -1 {
			syms = syms[idx+1:]
		}
		written += length
		addr += length
	}
	wg.Wait()
}

func writeBlock(ctxt *Link, out *OutBuf, ldr *loader.Loader, syms []loader.Sym, addr, size int64, pad []byte) {

	st := ctxt.makeRelocSymState()

	// This doesn't distinguish the memory size from the file
	// size, and it lays out the file based on Symbol.Value, which
	// is the virtual address. DWARF compression changes file sizes,
	// so dwarfcompress will fix this up later if necessary.
	eaddr := addr + size
	for _, s := range syms {
		if ldr.AttrSubSymbol(s) {
			continue
		}
		val := ldr.SymValue(s)
		if val >= eaddr {
			break
		}
		if val < addr {
			ldr.Errorf(s, "phase error: addr=%#x but val=%#x sym=%s type=%v sect=%v sect.addr=%#x", addr, val, ldr.SymName(s), ldr.SymType(s), ldr.SymSect(s).Name, ldr.SymSect(s).Vaddr)
			errorexit()
		}
		if addr < val {
			out.WriteStringPad("", int(val-addr), pad)
			addr = val
		}
		P := out.WriteSym(ldr, s)
		st.relocsym(s, P)
		if ldr.IsGeneratedSym(s) {
			f := ctxt.generatorSyms[s]
			f(ctxt, s)
		}
		addr += int64(len(P))
		siz := ldr.SymSize(s)
		if addr < val+siz {
			out.WriteStringPad("", int(val+siz-addr), pad)
			addr = val + siz
		}
		if addr != val+siz {
			ldr.Errorf(s, "phase error: addr=%#x value+size=%#x", addr, val+siz)
			errorexit()
		}
		if val+siz >= eaddr {
			break
		}
	}

	if addr < eaddr {
		out.WriteStringPad("", int(eaddr-addr), pad)
	}
}

type writeFn func(*Link, *OutBuf, int64, int64)

// writeParallel handles scheduling parallel execution of data write functions.
func writeParallel(wg *sync.WaitGroup, fn writeFn, ctxt *Link, seek, vaddr, length uint64) {
	if out, err := ctxt.Out.View(seek); err != nil {
		ctxt.Out.SeekSet(int64(seek))
		fn(ctxt, ctxt.Out, int64(vaddr), int64(length))
	} else {
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(ctxt, out, int64(vaddr), int64(length))
		}()
	}
}

func datblk(ctxt *Link, out *OutBuf, addr, size int64) {
	writeDatblkToOutBuf(ctxt, out, addr, size)
}

// Used only on Wasm for now.
func DatblkBytes(ctxt *Link, addr int64, size int64) []byte {
	buf := make([]byte, size)
	out := &OutBuf{heap: buf}
	writeDatblkToOutBuf(ctxt, out, addr, size)
	return buf
}

func writeDatblkToOutBuf(ctxt *Link, out *OutBuf, addr int64, size int64) {
	writeBlocks(ctxt, out, ctxt.outSem, ctxt.loader, ctxt.datap, addr, size, zeros[:])
}

func dwarfblk(ctxt *Link, out *OutBuf, addr int64, size int64) {
	// Concatenate the section symbol lists into a single list to pass
	// to writeBlocks.
	//
	// NB: ideally we would do a separate writeBlocks call for each
	// section, but this would run the risk of undoing any file offset
	// adjustments made during layout.
	n := 0
	for i := range dwarfp {
		n += len(dwarfp[i].syms)
	}
	syms := make([]loader.Sym, 0, n)
	for i := range dwarfp {
		syms = append(syms, dwarfp[i].syms...)
	}
	writeBlocks(ctxt, out, ctxt.outSem, ctxt.loader, syms, addr, size, zeros[:])
}

func pdatablk(ctxt *Link, out *OutBuf, addr int64, size int64) {
	writeBlocks(ctxt, out, ctxt.outSem, ctxt.loader, sehp.pdata, addr, size, zeros[:])
}

func xdatablk(ctxt *Link, out *OutBuf, addr int64, size int64) {
	writeBlocks(ctxt, out, ctxt.outSem, ctxt.loader, sehp.xdata, addr, size, zeros[:])
}

var covCounterDataStartOff, covCounterDataLen uint64

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

// addstrdata sets the initial value of the string variable name to value.
func addstrdata(arch *sys.Arch, l *loader.Loader, name, value string) {
	s := l.Lookup(name, 0)
	if s == 0 {
		return
	}
	if goType := l.SymGoType(s); goType == 0 {
		return
	} else if typeName := l.SymName(goType); typeName != "type:string" {
		Errorf(nil, "%s: cannot set with -X: not a var of type string (%s)", name, typeName)
		return
	}
	if !l.AttrReachable(s) {
		return // don't bother setting unreachable variable
	}
	bld := l.MakeSymbolUpdater(s)
	if bld.Type() == sym.SBSS {
		bld.SetType(sym.SDATA)
	}

	p := fmt.Sprintf("%s.str", name)
	sbld := l.CreateSymForUpdate(p, 0)
	sbld.Addstring(value)
	sbld.SetType(sym.SRODATA)

	// Don't reset the variable's size. String variable usually has size of
	// 2*PtrSize, but in ASAN build it can be larger due to red zone.
	// (See issue 56175.)
	bld.SetData(make([]byte, arch.PtrSize*2))
	bld.SetReadOnly(false)
	bld.ResetRelocs()
	bld.SetAddrPlus(arch, 0, sbld.Sym(), 0)
	bld.SetUint(arch, int64(arch.PtrSize), uint64(len(value)))
}

func (ctxt *Link) dostrdata() {
	for _, name := range strnames {
		addstrdata(ctxt.Arch, ctxt.loader, name, strdata[name])
	}
}

// addgostring adds str, as a Go string value, to s. symname is the name of the
// symbol used to define the string data and must be unique per linked object.
func addgostring(ctxt *Link, ldr *loader.Loader, s *loader.SymbolBuilder, symname, str string) {
	sdata := ldr.CreateSymForUpdate(symname, 0)
	if sdata.Type() != sym.Sxxx {
		ctxt.Errorf(s.Sym(), "duplicate symname in addgostring: %s", symname)
	}
	sdata.SetLocal(true)
	sdata.SetType(sym.SRODATA)
	sdata.SetSize(int64(len(str)))
	sdata.SetData([]byte(str))
	s.AddAddr(ctxt.Arch, sdata.Sym())
	s.AddUint(ctxt.Arch, uint64(len(str)))
}

func addinitarrdata(ctxt *Link, ldr *loader.Loader, s loader.Sym) {
	p := ldr.SymName(s) + ".ptr"
	sp := ldr.CreateSymForUpdate(p, 0)
	sp.SetType(sym.SINITARR)
	sp.SetSize(0)
	sp.SetDuplicateOK(true)
	sp.AddAddr(ctxt.Arch, s)
}

// symalign returns the required alignment for the given symbol s.
func symalign(ldr *loader.Loader, s loader.Sym) int32 {
	min := int32(thearch.Minalign)
	align := ldr.SymAlign(s)
	if align >= min {
		return align
	} else if align != 0 {
		return min
	}
	align = int32(thearch.Maxalign)
	ssz := ldr.SymSize(s)
	for int64(align) > ssz && align > min {
		align >>= 1
	}
	ldr.SetSymAlign(s, align)
	return align
}

func aligndatsize(state *dodataState, datsize int64, s loader.Sym) int64 {
	return Rnd(datsize, int64(symalign(state.ctxt.loader, s)))
}

const debugGCProg = false

type GCProg struct {
	ctxt *Link
	sym  *loader.SymbolBuilder
	w    gcprog.Writer
}

func (p *GCProg) Init(ctxt *Link, name string) {
	p.ctxt = ctxt
	p.sym = ctxt.loader.CreateSymForUpdate(name, 0)
	p.w.Init(p.writeByte())
	if debugGCProg {
		fmt.Fprintf(os.Stderr, "ld: start GCProg %s\n", name)
		p.w.Debug(os.Stderr)
	}
}

func (p *GCProg) writeByte() func(x byte) {
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

func (p *GCProg) AddSym(s loader.Sym) {
	ldr := p.ctxt.loader
	typ := ldr.SymGoType(s)

	// Things without pointers should be in sym.SNOPTRDATA or sym.SNOPTRBSS;
	// everything we see should have pointers and should therefore have a type.
	if typ == 0 {
		switch ldr.SymName(s) {
		case "runtime.data", "runtime.edata", "runtime.bss", "runtime.ebss":
			// Ignore special symbols that are sometimes laid out
			// as real symbols. See comment about dyld on darwin in
			// the address function.
			return
		}
		p.ctxt.Errorf(p.sym.Sym(), "missing Go type information for global symbol %s: size %d", ldr.SymName(s), ldr.SymSize(s))
		return
	}

	ptrsize := int64(p.ctxt.Arch.PtrSize)
	typData := ldr.Data(typ)
	nptr := decodetypePtrdata(p.ctxt.Arch, typData) / ptrsize

	if debugGCProg {
		fmt.Fprintf(os.Stderr, "gcprog sym: %s at %d (ptr=%d+%d)\n", ldr.SymName(s), ldr.SymValue(s), ldr.SymValue(s)/ptrsize, nptr)
	}

	sval := ldr.SymValue(s)
	if decodetypeUsegcprog(p.ctxt.Arch, typData) == 0 {
		// Copy pointers from mask into program.
		mask := decodetypeGcmask(p.ctxt, typ)
		for i := int64(0); i < nptr; i++ {
			if (mask[i/8]>>uint(i%8))&1 != 0 {
				p.w.Ptr(sval/ptrsize + i)
			}
		}
		return
	}

	// Copy program.
	prog := decodetypeGcprog(p.ctxt, typ)
	p.w.ZeroUntil(sval / ptrsize)
	p.w.Append(prog[4:], nptr)
}

// cutoff is the maximum data section size permitted by the linker
// (see issue #9862).
const cutoff = 2e9 // 2 GB (or so; looks better in errors than 2^31)

// check accumulated size of data sections
func (state *dodataState) checkdatsize(symn sym.SymKind) {
	if state.datsize > cutoff {
		Errorf(nil, "too much data, last section %v (%d, over %v bytes)", symn, state.datsize, cutoff)
	}
}

func checkSectSize(sect *sym.Section) {
	// TODO: consider using 4 GB size limit for DWARF sections, and
	// make sure we generate unsigned offset in relocations and check
	// for overflow.
	if sect.Length > cutoff {
		Errorf(nil, "too much data in section %s (%d, over %v bytes)", sect.Name, sect.Length, cutoff)
	}
}

// fixZeroSizedSymbols gives a few special symbols with zero size some space.
func fixZeroSizedSymbols(ctxt *Link) {
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
	//
	// On AIX, as all DATA sections are merged together, ld might not put
	// these symbols at the beginning of their respective section if there
	// aren't real symbols, their alignment might not match the
	// first symbol alignment. Therefore, there are explicitly put at the
	// beginning of their section with the same alignment.
	if !(ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin) && !(ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal) {
		return
	}

	ldr := ctxt.loader
	bss := ldr.CreateSymForUpdate("runtime.bss", 0)
	bss.SetSize(8)
	ldr.SetAttrSpecial(bss.Sym(), false)

	ebss := ldr.CreateSymForUpdate("runtime.ebss", 0)
	ldr.SetAttrSpecial(ebss.Sym(), false)

	data := ldr.CreateSymForUpdate("runtime.data", 0)
	data.SetSize(8)
	ldr.SetAttrSpecial(data.Sym(), false)

	edata := ldr.CreateSymForUpdate("runtime.edata", 0)
	ldr.SetAttrSpecial(edata.Sym(), false)

	if ctxt.HeadType == objabi.Haix {
		// XCOFFTOC symbols are part of .data section.
		edata.SetType(sym.SXCOFFTOC)
	}

	noptrbss := ldr.CreateSymForUpdate("runtime.noptrbss", 0)
	noptrbss.SetSize(8)
	ldr.SetAttrSpecial(noptrbss.Sym(), false)

	enoptrbss := ldr.CreateSymForUpdate("runtime.enoptrbss", 0)
	ldr.SetAttrSpecial(enoptrbss.Sym(), false)

	noptrdata := ldr.CreateSymForUpdate("runtime.noptrdata", 0)
	noptrdata.SetSize(8)
	ldr.SetAttrSpecial(noptrdata.Sym(), false)

	enoptrdata := ldr.CreateSymForUpdate("runtime.enoptrdata", 0)
	ldr.SetAttrSpecial(enoptrdata.Sym(), false)

	types := ldr.CreateSymForUpdate("runtime.types", 0)
	types.SetType(sym.STYPE)
	types.SetSize(8)
	ldr.SetAttrSpecial(types.Sym(), false)

	etypes := ldr.CreateSymForUpdate("runtime.etypes", 0)
	etypes.SetType(sym.SFUNCTAB)
	ldr.SetAttrSpecial(etypes.Sym(), false)

	if ctxt.HeadType == objabi.Haix {
		rodata := ldr.CreateSymForUpdate("runtime.rodata", 0)
		rodata.SetType(sym.SSTRING)
		rodata.SetSize(8)
		ldr.SetAttrSpecial(rodata.Sym(), false)

		erodata := ldr.CreateSymForUpdate("runtime.erodata", 0)
		ldr.SetAttrSpecial(erodata.Sym(), false)
	}
}

// makeRelroForSharedLib creates a section of readonly data if necessary.
func (state *dodataState) makeRelroForSharedLib(target *Link) {
	if !target.UseRelro() {
		return
	}

	// "read only" data with relocations needs to go in its own section
	// when building a shared library. We do this by boosting objects of
	// type SXXX with relocations to type SXXXRELRO.
	ldr := target.loader
	for _, symnro := range sym.ReadOnly {
		symnrelro := sym.RelROMap[symnro]

		ro := []loader.Sym{}
		relro := state.data[symnrelro]

		for _, s := range state.data[symnro] {
			relocs := ldr.Relocs(s)
			isRelro := relocs.Count() > 0
			switch state.symType(s) {
			case sym.STYPE, sym.STYPERELRO, sym.SGOFUNCRELRO:
				// Symbols are not sorted yet, so it is possible
				// that an Outer symbol has been changed to a
				// relro Type before it reaches here.
				isRelro = true
			case sym.SFUNCTAB:
				if ldr.SymName(s) == "runtime.etypes" {
					// runtime.etypes must be at the end of
					// the relro data.
					isRelro = true
				}
			case sym.SGOFUNC:
				// The only SGOFUNC symbols that contain relocations are .stkobj,
				// and their relocations are of type objabi.R_ADDROFF,
				// which always get resolved during linking.
				isRelro = false
			}
			if isRelro {
				state.setSymType(s, symnrelro)
				if outer := ldr.OuterSym(s); outer != 0 {
					state.setSymType(outer, symnrelro)
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
			if outer := ldr.OuterSym(s); outer != 0 {
				st := state.symType(s)
				ost := state.symType(outer)
				if st != ost {
					state.ctxt.Errorf(s, "inconsistent types for symbol and its Outer %s (%v != %v)",
						ldr.SymName(outer), st, ost)
				}
			}
		}

		state.data[symnro] = ro
		state.data[symnrelro] = relro
	}
}

// dodataState holds bits of state information needed by dodata() and the
// various helpers it calls. The lifetime of these items should not extend
// past the end of dodata().
type dodataState struct {
	// Link context
	ctxt *Link
	// Data symbols bucketed by type.
	data [sym.SXREF][]loader.Sym
	// Max alignment for each flavor of data symbol.
	dataMaxAlign [sym.SXREF]int32
	// Overridden sym type
	symGroupType []sym.SymKind
	// Current data size so far.
	datsize int64
}

// A note on symType/setSymType below:
//
// In the legacy linker, the types of symbols (notably data symbols) are
// changed during the symtab() phase so as to insure that similar symbols
// are bucketed together, then their types are changed back again during
// dodata. Symbol to section assignment also plays tricks along these lines
// in the case where a relro segment is needed.
//
// The value returned from setType() below reflects the effects of
// any overrides made by symtab and/or dodata.

// symType returns the (possibly overridden) type of 's'.
func (state *dodataState) symType(s loader.Sym) sym.SymKind {
	if int(s) < len(state.symGroupType) {
		if override := state.symGroupType[s]; override != 0 {
			return override
		}
	}
	return state.ctxt.loader.SymType(s)
}

// setSymType sets a new override type for 's'.
func (state *dodataState) setSymType(s loader.Sym, kind sym.SymKind) {
	if s == 0 {
		panic("bad")
	}
	if int(s) < len(state.symGroupType) {
		state.symGroupType[s] = kind
	} else {
		su := state.ctxt.loader.MakeSymbolUpdater(s)
		su.SetType(kind)
	}
}

func (ctxt *Link) dodata(symGroupType []sym.SymKind) {

	// Give zeros sized symbols space if necessary.
	fixZeroSizedSymbols(ctxt)

	// Collect data symbols by type into data.
	state := dodataState{ctxt: ctxt, symGroupType: symGroupType}
	ldr := ctxt.loader
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if !ldr.AttrReachable(s) || ldr.AttrSpecial(s) || ldr.AttrSubSymbol(s) ||
			!ldr.TopLevelSym(s) {
			continue
		}

		st := state.symType(s)

		if st <= sym.STEXT || st >= sym.SXREF {
			continue
		}
		state.data[st] = append(state.data[st], s)

		// Similarly with checking the onlist attr.
		if ldr.AttrOnList(s) {
			log.Fatalf("symbol %s listed multiple times", ldr.SymName(s))
		}
		ldr.SetAttrOnList(s, true)
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
	state.dynreloc(ctxt)

	// Move any RO data with relocations to a separate section.
	state.makeRelroForSharedLib(ctxt)

	// Set alignment for the symbol with the largest known index,
	// so as to trigger allocation of the loader's internal
	// alignment array. This will avoid data races in the parallel
	// section below.
	lastSym := loader.Sym(ldr.NSym() - 1)
	ldr.SetSymAlign(lastSym, ldr.SymAlign(lastSym))

	// Sort symbols.
	var wg sync.WaitGroup
	for symn := range state.data {
		symn := sym.SymKind(symn)
		wg.Add(1)
		go func() {
			state.data[symn], state.dataMaxAlign[symn] = state.dodataSect(ctxt, symn, state.data[symn])
			wg.Done()
		}()
	}
	wg.Wait()

	if ctxt.IsELF {
		// Make .rela and .rela.plt contiguous, the ELF ABI requires this
		// and Solaris actually cares.
		syms := state.data[sym.SELFROSECT]
		reli, plti := -1, -1
		for i, s := range syms {
			switch ldr.SymName(s) {
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
			ldr.SetSymAlign(rel, int32(ctxt.Arch.RegSize))
			ldr.SetSymAlign(plt, int32(ctxt.Arch.RegSize))
		}
		state.data[sym.SELFROSECT] = syms
	}

	if ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal {
		// These symbols must have the same alignment as their section.
		// Otherwise, ld might change the layout of Go sections.
		ldr.SetSymAlign(ldr.Lookup("runtime.data", 0), state.dataMaxAlign[sym.SDATA])
		ldr.SetSymAlign(ldr.Lookup("runtime.bss", 0), state.dataMaxAlign[sym.SBSS])
	}

	// Create *sym.Section objects and assign symbols to sections for
	// data/rodata (and related) symbols.
	state.allocateDataSections(ctxt)

	state.allocateSEHSections(ctxt)

	// Create *sym.Section objects and assign symbols to sections for
	// DWARF symbols.
	state.allocateDwarfSections(ctxt)

	/* number the sections */
	n := int16(1)

	for _, sect := range Segtext.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segrodata.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segrelrodata.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segdata.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segdwarf.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segpdata.Sections {
		sect.Extnum = n
		n++
	}
	for _, sect := range Segxdata.Sections {
		sect.Extnum = n
		n++
	}
}

// allocateDataSectionForSym creates a new sym.Section into which a
// single symbol will be placed. Here "seg" is the segment into which
// the section will go, "s" is the symbol to be placed into the new
// section, and "rwx" contains permissions for the section.
func (state *dodataState) allocateDataSectionForSym(seg *sym.Segment, s loader.Sym, rwx int) *sym.Section {
	ldr := state.ctxt.loader
	sname := ldr.SymName(s)
	if strings.HasPrefix(sname, "go:") {
		sname = ".go." + sname[len("go:"):]
	}
	sect := addsection(ldr, state.ctxt.Arch, seg, sname, rwx)
	sect.Align = symalign(ldr, s)
	state.datsize = Rnd(state.datsize, int64(sect.Align))
	sect.Vaddr = uint64(state.datsize)
	return sect
}

// allocateNamedDataSection creates a new sym.Section for a category
// of data symbols. Here "seg" is the segment into which the section
// will go, "sName" is the name to give to the section, "types" is a
// range of symbol types to be put into the section, and "rwx"
// contains permissions for the section.
func (state *dodataState) allocateNamedDataSection(seg *sym.Segment, sName string, types []sym.SymKind, rwx int) *sym.Section {
	sect := addsection(state.ctxt.loader, state.ctxt.Arch, seg, sName, rwx)
	if len(types) == 0 {
		sect.Align = 1
	} else if len(types) == 1 {
		sect.Align = state.dataMaxAlign[types[0]]
	} else {
		for _, symn := range types {
			align := state.dataMaxAlign[symn]
			if sect.Align < align {
				sect.Align = align
			}
		}
	}
	state.datsize = Rnd(state.datsize, int64(sect.Align))
	sect.Vaddr = uint64(state.datsize)
	return sect
}

// assignDsymsToSection assigns a collection of data symbols to a
// newly created section. "sect" is the section into which to place
// the symbols, "syms" holds the list of symbols to assign,
// "forceType" (if non-zero) contains a new sym type to apply to each
// sym during the assignment, and "aligner" is a hook to call to
// handle alignment during the assignment process.
func (state *dodataState) assignDsymsToSection(sect *sym.Section, syms []loader.Sym, forceType sym.SymKind, aligner func(state *dodataState, datsize int64, s loader.Sym) int64) {
	ldr := state.ctxt.loader
	for _, s := range syms {
		state.datsize = aligner(state, state.datsize, s)
		ldr.SetSymSect(s, sect)
		if forceType != sym.Sxxx {
			state.setSymType(s, forceType)
		}
		ldr.SetSymValue(s, int64(uint64(state.datsize)-sect.Vaddr))
		state.datsize += ldr.SymSize(s)
	}
	sect.Length = uint64(state.datsize) - sect.Vaddr
}

func (state *dodataState) assignToSection(sect *sym.Section, symn sym.SymKind, forceType sym.SymKind) {
	state.assignDsymsToSection(sect, state.data[symn], forceType, aligndatsize)
	state.checkdatsize(symn)
}

// allocateSingleSymSections walks through the bucketed data symbols
// with type 'symn', creates a new section for each sym, and assigns
// the sym to a newly created section. Section name is set from the
// symbol name. "Seg" is the segment into which to place the new
// section, "forceType" is the new sym.SymKind to assign to the symbol
// within the section, and "rwx" holds section permissions.
func (state *dodataState) allocateSingleSymSections(seg *sym.Segment, symn sym.SymKind, forceType sym.SymKind, rwx int) {
	ldr := state.ctxt.loader
	for _, s := range state.data[symn] {
		sect := state.allocateDataSectionForSym(seg, s, rwx)
		ldr.SetSymSect(s, sect)
		state.setSymType(s, forceType)
		ldr.SetSymValue(s, int64(uint64(state.datsize)-sect.Vaddr))
		state.datsize += ldr.SymSize(s)
		sect.Length = uint64(state.datsize) - sect.Vaddr
	}
	state.checkdatsize(symn)
}

// allocateNamedSectionAndAssignSyms creates a new section with the
// specified name, then walks through the bucketed data symbols with
// type 'symn' and assigns each of them to this new section. "Seg" is
// the segment into which to place the new section, "secName" is the
// name to give to the new section, "forceType" (if non-zero) contains
// a new sym type to apply to each sym during the assignment, and
// "rwx" holds section permissions.
func (state *dodataState) allocateNamedSectionAndAssignSyms(seg *sym.Segment, secName string, symn sym.SymKind, forceType sym.SymKind, rwx int) *sym.Section {

	sect := state.allocateNamedDataSection(seg, secName, []sym.SymKind{symn}, rwx)
	state.assignDsymsToSection(sect, state.data[symn], forceType, aligndatsize)
	return sect
}

// allocateDataSections allocates sym.Section objects for data/rodata
// (and related) symbols, and then assigns symbols to those sections.
func (state *dodataState) allocateDataSections(ctxt *Link) {
	// Allocate sections.
	// Data is processed before segtext, because we need
	// to see all symbols in the .data and .bss sections in order
	// to generate garbage collection information.

	// Writable data sections that do not need any specialized handling.
	writable := []sym.SymKind{
		sym.SBUILDINFO,
		sym.SELFSECT,
		sym.SMACHO,
		sym.SMACHOGOT,
		sym.SWINDOWS,
	}
	for _, symn := range writable {
		state.allocateSingleSymSections(&Segdata, symn, sym.SDATA, 06)
	}
	ldr := ctxt.loader

	// .got
	if len(state.data[sym.SELFGOT]) > 0 {
		state.allocateNamedSectionAndAssignSyms(&Segdata, ".got", sym.SELFGOT, sym.SDATA, 06)
	}

	/* pointer-free data */
	sect := state.allocateNamedSectionAndAssignSyms(&Segdata, ".noptrdata", sym.SNOPTRDATA, sym.SDATA, 06)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.noptrdata", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.enoptrdata", 0), sect)

	hasinitarr := ctxt.linkShared

	/* shared library initializer */
	switch ctxt.BuildMode {
	case BuildModeCArchive, BuildModeCShared, BuildModeShared, BuildModePlugin:
		hasinitarr = true
	}

	if ctxt.HeadType == objabi.Haix {
		if len(state.data[sym.SINITARR]) > 0 {
			Errorf(nil, "XCOFF format doesn't allow .init_array section")
		}
	}

	if hasinitarr && len(state.data[sym.SINITARR]) > 0 {
		state.allocateNamedSectionAndAssignSyms(&Segdata, ".init_array", sym.SINITARR, sym.Sxxx, 06)
	}

	/* data */
	sect = state.allocateNamedSectionAndAssignSyms(&Segdata, ".data", sym.SDATA, sym.SDATA, 06)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.data", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.edata", 0), sect)
	dataGcEnd := state.datsize - int64(sect.Vaddr)

	// On AIX, TOC entries must be the last of .data
	// These aren't part of gc as they won't change during the runtime.
	state.assignToSection(sect, sym.SXCOFFTOC, sym.SDATA)
	state.checkdatsize(sym.SDATA)
	sect.Length = uint64(state.datsize) - sect.Vaddr

	/* bss */
	sect = state.allocateNamedSectionAndAssignSyms(&Segdata, ".bss", sym.SBSS, sym.Sxxx, 06)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.bss", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.ebss", 0), sect)
	bssGcEnd := state.datsize - int64(sect.Vaddr)

	// Emit gcdata for bss symbols now that symbol values have been assigned.
	gcsToEmit := []struct {
		symName string
		symKind sym.SymKind
		gcEnd   int64
	}{
		{"runtime.gcdata", sym.SDATA, dataGcEnd},
		{"runtime.gcbss", sym.SBSS, bssGcEnd},
	}
	for _, g := range gcsToEmit {
		var gc GCProg
		gc.Init(ctxt, g.symName)
		for _, s := range state.data[g.symKind] {
			gc.AddSym(s)
		}
		gc.End(g.gcEnd)
	}

	/* pointer-free bss */
	sect = state.allocateNamedSectionAndAssignSyms(&Segdata, ".noptrbss", sym.SNOPTRBSS, sym.Sxxx, 06)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.noptrbss", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.enoptrbss", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.end", 0), sect)

	// Code coverage counters are assigned to the .noptrbss section.
	// We assign them in a separate pass so that they stay aggregated
	// together in a single blob (coverage runtime depends on this).
	covCounterDataStartOff = sect.Length
	state.assignToSection(sect, sym.SCOVERAGE_COUNTER, sym.SNOPTRBSS)
	covCounterDataLen = sect.Length - covCounterDataStartOff
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.covctrs", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.ecovctrs", 0), sect)

	// Coverage instrumentation counters for libfuzzer.
	if len(state.data[sym.SLIBFUZZER_8BIT_COUNTER]) > 0 {
		sect := state.allocateNamedSectionAndAssignSyms(&Segdata, ".go.fuzzcntrs", sym.SLIBFUZZER_8BIT_COUNTER, sym.Sxxx, 06)
		ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.__start___sancov_cntrs", 0), sect)
		ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.__stop___sancov_cntrs", 0), sect)
		ldr.SetSymSect(ldr.LookupOrCreateSym("internal/fuzz._counters", 0), sect)
		ldr.SetSymSect(ldr.LookupOrCreateSym("internal/fuzz._ecounters", 0), sect)
	}

	if len(state.data[sym.STLSBSS]) > 0 {
		var sect *sym.Section
		// FIXME: not clear why it is sometimes necessary to suppress .tbss section creation.
		if (ctxt.IsELF || ctxt.HeadType == objabi.Haix) && (ctxt.LinkMode == LinkExternal || !*FlagD) {
			sect = addsection(ldr, ctxt.Arch, &Segdata, ".tbss", 06)
			sect.Align = int32(ctxt.Arch.PtrSize)
			// FIXME: why does this need to be set to zero?
			sect.Vaddr = 0
		}
		state.datsize = 0

		for _, s := range state.data[sym.STLSBSS] {
			state.datsize = aligndatsize(state, state.datsize, s)
			if sect != nil {
				ldr.SetSymSect(s, sect)
			}
			ldr.SetSymValue(s, state.datsize)
			state.datsize += ldr.SymSize(s)
		}
		state.checkdatsize(sym.STLSBSS)

		if sect != nil {
			sect.Length = uint64(state.datsize)
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

	state.datsize = 0

	/* read-only executable ELF, Mach-O sections */
	if len(state.data[sym.STEXT]) != 0 {
		culprit := ldr.SymName(state.data[sym.STEXT][0])
		Errorf(nil, "dodata found an sym.STEXT symbol: %s", culprit)
	}
	state.allocateSingleSymSections(&Segtext, sym.SELFRXSECT, sym.SRODATA, 05)
	state.allocateSingleSymSections(&Segtext, sym.SMACHOPLT, sym.SRODATA, 05)

	/* read-only data */
	sect = state.allocateNamedDataSection(segro, ".rodata", sym.ReadOnly, 04)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.rodata", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.erodata", 0), sect)
	if !ctxt.UseRelro() {
		ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.types", 0), sect)
		ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.etypes", 0), sect)
	}
	for _, symn := range sym.ReadOnly {
		symnStartValue := state.datsize
		if len(state.data[symn]) != 0 {
			symnStartValue = aligndatsize(state, symnStartValue, state.data[symn][0])
		}
		state.assignToSection(sect, symn, sym.SRODATA)
		setCarrierSize(symn, state.datsize-symnStartValue)
		if ctxt.HeadType == objabi.Haix {
			// Read-only symbols might be wrapped inside their outer
			// symbol.
			// XCOFF symbol table needs to know the size of
			// these outer symbols.
			xcoffUpdateOuterSize(ctxt, state.datsize-symnStartValue, symn)
		}
	}

	/* read-only ELF, Mach-O sections */
	state.allocateSingleSymSections(segro, sym.SELFROSECT, sym.SRODATA, 04)

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
	const relroPerm = 06
	const fallbackPerm = 04
	relroSecPerm := fallbackPerm
	genrelrosecname := func(suffix string) string {
		if suffix == "" {
			return ".rodata"
		}
		return suffix
	}
	seg := segro

	if ctxt.UseRelro() {
		segrelro := &Segrelrodata
		if ctxt.LinkMode == LinkExternal && !ctxt.IsAIX() && !ctxt.IsDarwin() {
			// Using a separate segment with an external
			// linker results in some programs moving
			// their data sections unexpectedly, which
			// corrupts the moduledata. So we use the
			// rodata segment and let the external linker
			// sort out a rel.ro segment.
			segrelro = segro
		} else {
			// Reset datsize for new segment.
			state.datsize = 0
		}

		if !ctxt.IsDarwin() { // We don't need the special names on darwin.
			genrelrosecname = func(suffix string) string {
				return ".data.rel.ro" + suffix
			}
		}

		relroReadOnly := []sym.SymKind{}
		for _, symnro := range sym.ReadOnly {
			symn := sym.RelROMap[symnro]
			relroReadOnly = append(relroReadOnly, symn)
		}
		seg = segrelro
		relroSecPerm = relroPerm

		/* data only written by relocations */
		sect = state.allocateNamedDataSection(segrelro, genrelrosecname(""), relroReadOnly, relroSecPerm)

		ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.types", 0), sect)
		ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.etypes", 0), sect)

		for i, symnro := range sym.ReadOnly {
			if i == 0 && symnro == sym.STYPE && ctxt.HeadType != objabi.Haix {
				// Skip forward so that no type
				// reference uses a zero offset.
				// This is unlikely but possible in small
				// programs with no other read-only data.
				state.datsize++
			}

			symn := sym.RelROMap[symnro]
			symnStartValue := state.datsize
			if len(state.data[symn]) != 0 {
				symnStartValue = aligndatsize(state, symnStartValue, state.data[symn][0])
			}

			for _, s := range state.data[symn] {
				outer := ldr.OuterSym(s)
				if s != 0 && ldr.SymSect(outer) != nil && ldr.SymSect(outer) != sect {
					ctxt.Errorf(s, "s.Outer (%s) in different section from s, %s != %s", ldr.SymName(outer), ldr.SymSect(outer).Name, sect.Name)
				}
			}
			state.assignToSection(sect, symn, sym.SRODATA)
			setCarrierSize(symn, state.datsize-symnStartValue)
			if ctxt.HeadType == objabi.Haix {
				// Read-only symbols might be wrapped inside their outer
				// symbol.
				// XCOFF symbol table needs to know the size of
				// these outer symbols.
				xcoffUpdateOuterSize(ctxt, state.datsize-symnStartValue, symn)
			}
		}

		sect.Length = uint64(state.datsize) - sect.Vaddr
	}

	/* typelink */
	sect = state.allocateNamedDataSection(seg, genrelrosecname(".typelink"), []sym.SymKind{sym.STYPELINK}, relroSecPerm)

	typelink := ldr.CreateSymForUpdate("runtime.typelink", 0)
	ldr.SetSymSect(typelink.Sym(), sect)
	typelink.SetType(sym.SRODATA)
	state.datsize += typelink.Size()
	state.checkdatsize(sym.STYPELINK)
	sect.Length = uint64(state.datsize) - sect.Vaddr

	/* itablink */
	sect = state.allocateNamedDataSection(seg, genrelrosecname(".itablink"), []sym.SymKind{sym.SITABLINK}, relroSecPerm)

	itablink := ldr.CreateSymForUpdate("runtime.itablink", 0)
	ldr.SetSymSect(itablink.Sym(), sect)
	itablink.SetType(sym.SRODATA)
	state.datsize += itablink.Size()
	state.checkdatsize(sym.SITABLINK)
	sect.Length = uint64(state.datsize) - sect.Vaddr

	/* gosymtab */
	sect = state.allocateNamedSectionAndAssignSyms(seg, genrelrosecname(".gosymtab"), sym.SSYMTAB, sym.SRODATA, relroSecPerm)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.symtab", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.esymtab", 0), sect)

	/* gopclntab */
	sect = state.allocateNamedSectionAndAssignSyms(seg, genrelrosecname(".gopclntab"), sym.SPCLNTAB, sym.SRODATA, relroSecPerm)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.pclntab", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.pcheader", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.funcnametab", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.cutab", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.filetab", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.pctab", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.functab", 0), sect)
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.epclntab", 0), sect)
	setCarrierSize(sym.SPCLNTAB, int64(sect.Length))
	if ctxt.HeadType == objabi.Haix {
		xcoffUpdateOuterSize(ctxt, int64(sect.Length), sym.SPCLNTAB)
	}

	// 6g uses 4-byte relocation offsets, so the entire segment must fit in 32 bits.
	if state.datsize != int64(uint32(state.datsize)) {
		Errorf(nil, "read-only data segment too large: %d", state.datsize)
	}

	siz := 0
	for symn := sym.SELFRXSECT; symn < sym.SXREF; symn++ {
		siz += len(state.data[symn])
	}
	ctxt.datap = make([]loader.Sym, 0, siz)
	for symn := sym.SELFRXSECT; symn < sym.SXREF; symn++ {
		ctxt.datap = append(ctxt.datap, state.data[symn]...)
	}
}

// allocateDwarfSections allocates sym.Section objects for DWARF
// symbols, and assigns symbols to sections.
func (state *dodataState) allocateDwarfSections(ctxt *Link) {

	alignOne := func(state *dodataState, datsize int64, s loader.Sym) int64 { return datsize }

	ldr := ctxt.loader
	for i := 0; i < len(dwarfp); i++ {
		// First the section symbol.
		s := dwarfp[i].secSym()
		sect := state.allocateNamedDataSection(&Segdwarf, ldr.SymName(s), []sym.SymKind{}, 04)
		ldr.SetSymSect(s, sect)
		sect.Sym = sym.LoaderSym(s)
		curType := ldr.SymType(s)
		state.setSymType(s, sym.SRODATA)
		ldr.SetSymValue(s, int64(uint64(state.datsize)-sect.Vaddr))
		state.datsize += ldr.SymSize(s)

		// Then any sub-symbols for the section symbol.
		subSyms := dwarfp[i].subSyms()
		state.assignDsymsToSection(sect, subSyms, sym.SRODATA, alignOne)

		for j := 0; j < len(subSyms); j++ {
			s := subSyms[j]
			if ctxt.HeadType == objabi.Haix && curType == sym.SDWARFLOC {
				// Update the size of .debug_loc for this symbol's
				// package.
				addDwsectCUSize(".debug_loc", ldr.SymPkg(s), uint64(ldr.SymSize(s)))
			}
		}
		sect.Length = uint64(state.datsize) - sect.Vaddr
		checkSectSize(sect)
	}
}

// allocateSEHSections allocate a sym.Section object for SEH
// symbols, and assigns symbols to sections.
func (state *dodataState) allocateSEHSections(ctxt *Link) {
	if len(sehp.pdata) > 0 {
		sect := state.allocateNamedDataSection(&Segpdata, ".pdata", []sym.SymKind{}, 04)
		state.assignDsymsToSection(sect, sehp.pdata, sym.SRODATA, aligndatsize)
		state.checkdatsize(sym.SSEHSECT)
	}
	if len(sehp.xdata) > 0 {
		sect := state.allocateNamedDataSection(&Segxdata, ".xdata", []sym.SymKind{}, 04)
		state.assignDsymsToSection(sect, sehp.xdata, sym.SRODATA, aligndatsize)
		state.checkdatsize(sym.SSEHSECT)
	}
}

type symNameSize struct {
	name string
	sz   int64
	val  int64
	sym  loader.Sym
}

func (state *dodataState) dodataSect(ctxt *Link, symn sym.SymKind, syms []loader.Sym) (result []loader.Sym, maxAlign int32) {
	var head, tail, zerobase loader.Sym
	ldr := ctxt.loader
	sl := make([]symNameSize, len(syms))

	// For ppc64, we want to interleave the .got and .toc sections
	// from input files. Both are type sym.SELFGOT, so in that case
	// we skip size comparison and do the name comparison instead
	// (conveniently, .got sorts before .toc).
	checkSize := symn != sym.SELFGOT

	for k, s := range syms {
		ss := ldr.SymSize(s)
		sl[k] = symNameSize{sz: ss, sym: s}
		if !checkSize {
			sl[k].name = ldr.SymName(s)
		}
		ds := int64(len(ldr.Data(s)))
		switch {
		case ss < ds:
			ctxt.Errorf(s, "initialize bounds (%d < %d)", ss, ds)
		case ss < 0:
			ctxt.Errorf(s, "negative size (%d bytes)", ss)
		case ss > cutoff:
			ctxt.Errorf(s, "symbol too large (%d bytes)", ss)
		}

		// If the usually-special section-marker symbols are being laid
		// out as regular symbols, put them either at the beginning or
		// end of their section.
		if (ctxt.DynlinkingGo() && ctxt.HeadType == objabi.Hdarwin) || (ctxt.HeadType == objabi.Haix && ctxt.LinkMode == LinkExternal) {
			switch ldr.SymName(s) {
			case "runtime.text", "runtime.bss", "runtime.data", "runtime.types", "runtime.rodata",
				"runtime.noptrdata", "runtime.noptrbss":
				head = s
				continue
			case "runtime.etext", "runtime.ebss", "runtime.edata", "runtime.etypes", "runtime.erodata",
				"runtime.enoptrdata", "runtime.enoptrbss":
				tail = s
				continue
			}
		}
	}
	zerobase = ldr.Lookup("runtime.zerobase", 0)

	// Perform the sort.
	if symn != sym.SPCLNTAB {
		sort.Slice(sl, func(i, j int) bool {
			si, sj := sl[i].sym, sl[j].sym
			isz, jsz := sl[i].sz, sl[j].sz
			switch {
			case si == head, sj == tail:
				return true
			case sj == head, si == tail:
				return false
			// put zerobase right after all the zero-sized symbols,
			// so zero-sized symbols have the same address as zerobase.
			case si == zerobase:
				return jsz != 0 // zerobase < nonzero-sized
			case sj == zerobase:
				return isz == 0 // 0-sized < zerobase
			}
			if checkSize {
				if isz != jsz {
					return isz < jsz
				}
			} else {
				iname := sl[i].name
				jname := sl[j].name
				if iname != jname {
					return iname < jname
				}
			}
			return si < sj
		})
	} else {
		// PCLNTAB was built internally, and already has the proper order.
	}

	// Set alignment, construct result
	syms = syms[:0]
	for k := range sl {
		s := sl[k].sym
		if s != head && s != tail {
			align := symalign(ldr, s)
			if maxAlign < align {
				maxAlign = align
			}
		}
		syms = append(syms, s)
	}

	return syms, maxAlign
}

// Add buildid to beginning of text segment, on non-ELF systems.
// Non-ELF binary formats are not always flexible enough to
// give us a place to put the Go build ID. On those systems, we put it
// at the very beginning of the text segment.
// This “header” is read by cmd/go.
func (ctxt *Link) textbuildid() {
	if ctxt.IsELF || *flagBuildid == "" {
		return
	}

	ldr := ctxt.loader
	s := ldr.CreateSymForUpdate("go:buildid", 0)
	// The \xff is invalid UTF-8, meant to make it less likely
	// to find one of these accidentally.
	data := "\xff Go build ID: " + strconv.Quote(*flagBuildid) + "\n \xff"
	s.SetType(sym.STEXT)
	s.SetData([]byte(data))
	s.SetSize(int64(len(data)))

	ctxt.Textp = append(ctxt.Textp, 0)
	copy(ctxt.Textp[1:], ctxt.Textp)
	ctxt.Textp[0] = s.Sym()
}

func (ctxt *Link) buildinfo() {
	// Write the buildinfo symbol, which go version looks for.
	// The code reading this data is in package debug/buildinfo.
	ldr := ctxt.loader
	s := ldr.CreateSymForUpdate("go:buildinfo", 0)
	s.SetType(sym.SBUILDINFO)
	s.SetAlign(16)
	// The \xff is invalid UTF-8, meant to make it less likely
	// to find one of these accidentally.
	const prefix = "\xff Go buildinf:" // 14 bytes, plus 2 data bytes filled in below
	data := make([]byte, 32)
	copy(data, prefix)
	data[len(prefix)] = byte(ctxt.Arch.PtrSize)
	data[len(prefix)+1] = 0
	if ctxt.Arch.ByteOrder == binary.BigEndian {
		data[len(prefix)+1] = 1
	}
	data[len(prefix)+1] |= 2 // signals new pointer-free format
	data = appendString(data, strdata["runtime.buildVersion"])
	data = appendString(data, strdata["runtime.modinfo"])
	// MacOS linker gets very upset if the size os not a multiple of alignment.
	for len(data)%16 != 0 {
		data = append(data, 0)
	}
	s.SetData(data)
	s.SetSize(int64(len(data)))

	// Add reference to go:buildinfo from the rodata section,
	// so that external linking with -Wl,--gc-sections does not
	// delete the build info.
	sr := ldr.CreateSymForUpdate("go:buildinfo.ref", 0)
	sr.SetType(sym.SRODATA)
	sr.SetAlign(int32(ctxt.Arch.PtrSize))
	sr.AddAddr(ctxt.Arch, s.Sym())
}

// appendString appends s to data, prefixed by its varint-encoded length.
func appendString(data []byte, s string) []byte {
	var v [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(v[:], uint64(len(s)))
	data = append(data, v[:n]...)
	data = append(data, s...)
	return data
}

// assign addresses to text
func (ctxt *Link) textaddress() {
	addsection(ctxt.loader, ctxt.Arch, &Segtext, ".text", 05)

	// Assign PCs in text segment.
	// Could parallelize, by assigning to text
	// and then letting threads copy down, but probably not worth it.
	sect := Segtext.Sections[0]

	sect.Align = int32(Funcalign)

	ldr := ctxt.loader

	text := ctxt.xdefine("runtime.text", sym.STEXT, 0)
	etext := ctxt.xdefine("runtime.etext", sym.STEXT, 0)
	ldr.SetSymSect(text, sect)
	if ctxt.IsAIX() && ctxt.IsExternal() {
		// Setting runtime.text has a real symbol prevents ld to
		// change its base address resulting in wrong offsets for
		// reflect methods.
		u := ldr.MakeSymbolUpdater(text)
		u.SetAlign(sect.Align)
		u.SetSize(8)
	}

	if (ctxt.DynlinkingGo() && ctxt.IsDarwin()) || (ctxt.IsAIX() && ctxt.IsExternal()) {
		ldr.SetSymSect(etext, sect)
		ctxt.Textp = append(ctxt.Textp, etext, 0)
		copy(ctxt.Textp[1:], ctxt.Textp)
		ctxt.Textp[0] = text
	}

	start := uint64(Rnd(*FlagTextAddr, int64(Funcalign)))
	va := start
	n := 1
	sect.Vaddr = va

	limit := thearch.TrampLimit
	if limit == 0 {
		limit = 1 << 63 // unlimited
	}
	if *FlagDebugTextSize != 0 {
		limit = uint64(*FlagDebugTextSize)
	}
	if *FlagDebugTramp > 1 {
		limit = 1 // debug mode, force generating trampolines for everything
	}

	if ctxt.IsAIX() && ctxt.IsExternal() {
		// On AIX, normally we won't generate direct calls to external symbols,
		// except in one test, cmd/go/testdata/script/link_syso_issue33139.txt.
		// That test doesn't make much sense, and I'm not sure it ever works.
		// Just generate trampoline for now (which will turn a direct call to
		// an indirect call, which at least builds).
		limit = 1
	}

	// First pass: assign addresses assuming the program is small and will
	// not require trampoline generation.
	big := false
	for _, s := range ctxt.Textp {
		sect, n, va = assignAddress(ctxt, sect, n, s, va, false, big)
		if va-start >= limit {
			big = true
			break
		}
	}

	// Second pass: only if it is too big, insert trampolines for too-far
	// jumps and targets with unknown addresses.
	if big {
		// reset addresses
		for _, s := range ctxt.Textp {
			if s != text {
				resetAddress(ctxt, s)
			}
		}
		va = start

		ntramps := 0
		var curPkg string
		for i, s := range ctxt.Textp {
			// When we find the first symbol in a package, perform a
			// single iteration that assigns temporary addresses to all
			// of the text in the same package, using the maximum possible
			// number of trampolines. This allows for better decisions to
			// be made regarding reachability and the need for trampolines.
			if symPkg := ldr.SymPkg(s); symPkg != "" && curPkg != symPkg {
				curPkg = symPkg
				vaTmp := va
				for j := i; j < len(ctxt.Textp); j++ {
					curSym := ctxt.Textp[j]
					if symPkg := ldr.SymPkg(curSym); symPkg == "" || curPkg != symPkg {
						break
					}
					// We do not pass big to assignAddress here, as this
					// can result in side effects such as section splitting.
					sect, n, vaTmp = assignAddress(ctxt, sect, n, curSym, vaTmp, false, false)
					vaTmp += maxSizeTrampolines(ctxt, ldr, curSym, false)
				}
			}

			// Reset address for current symbol.
			if s != text {
				resetAddress(ctxt, s)
			}

			// Assign actual address for current symbol.
			sect, n, va = assignAddress(ctxt, sect, n, s, va, false, big)

			// Resolve jumps, adding trampolines if they are needed.
			trampoline(ctxt, s)

			// lay down trampolines after each function
			for ; ntramps < len(ctxt.tramps); ntramps++ {
				tramp := ctxt.tramps[ntramps]
				if ctxt.IsAIX() && strings.HasPrefix(ldr.SymName(tramp), "runtime.text.") {
					// Already set in assignAddress
					continue
				}
				sect, n, va = assignAddress(ctxt, sect, n, tramp, va, true, big)
			}
		}

		// merge tramps into Textp, keeping Textp in address order
		if ntramps != 0 {
			newtextp := make([]loader.Sym, 0, len(ctxt.Textp)+ntramps)
			i := 0
			for _, s := range ctxt.Textp {
				for ; i < ntramps && ldr.SymValue(ctxt.tramps[i]) < ldr.SymValue(s); i++ {
					newtextp = append(newtextp, ctxt.tramps[i])
				}
				newtextp = append(newtextp, s)
			}
			newtextp = append(newtextp, ctxt.tramps[i:ntramps]...)

			ctxt.Textp = newtextp
		}
	}

	// Add MinLC size after etext, so it won't collide with the next symbol
	// (which may confuse some symbolizer).
	sect.Length = va - sect.Vaddr + uint64(ctxt.Arch.MinLC)
	ldr.SetSymSect(etext, sect)
	if ldr.SymValue(etext) == 0 {
		// Set the address of the start/end symbols, if not already
		// (i.e. not darwin+dynlink or AIX+external, see above).
		ldr.SetSymValue(etext, int64(va))
		ldr.SetSymValue(text, int64(Segtext.Sections[0].Vaddr))
	}
}

// assigns address for a text symbol, returns (possibly new) section, its number, and the address.
func assignAddress(ctxt *Link, sect *sym.Section, n int, s loader.Sym, va uint64, isTramp, big bool) (*sym.Section, int, uint64) {
	ldr := ctxt.loader
	if thearch.AssignAddress != nil {
		return thearch.AssignAddress(ldr, sect, n, s, va, isTramp)
	}

	ldr.SetSymSect(s, sect)
	if ldr.AttrSubSymbol(s) {
		return sect, n, va
	}

	align := ldr.SymAlign(s)
	if align == 0 {
		align = int32(Funcalign)
	}
	va = uint64(Rnd(int64(va), int64(align)))
	if sect.Align < align {
		sect.Align = align
	}

	funcsize := uint64(MINFUNC) // spacing required for findfunctab
	if ldr.SymSize(s) > MINFUNC {
		funcsize = uint64(ldr.SymSize(s))
	}

	// If we need to split text sections, and this function doesn't fit in the current
	// section, then create a new one.
	//
	// Only break at outermost syms.
	if big && splitTextSections(ctxt) && ldr.OuterSym(s) == 0 {
		// For debugging purposes, allow text size limit to be cranked down,
		// so as to stress test the code that handles multiple text sections.
		var textSizelimit uint64 = thearch.TrampLimit
		if *FlagDebugTextSize != 0 {
			textSizelimit = uint64(*FlagDebugTextSize)
		}

		// Sanity check: make sure the limit is larger than any
		// individual text symbol.
		if funcsize > textSizelimit {
			panic(fmt.Sprintf("error: text size limit %d less than text symbol %s size of %d", textSizelimit, ldr.SymName(s), funcsize))
		}

		if va-sect.Vaddr+funcsize+maxSizeTrampolines(ctxt, ldr, s, isTramp) > textSizelimit {
			sectAlign := int32(thearch.Funcalign)
			if ctxt.IsPPC64() {
				// Align the next text section to the worst case function alignment likely
				// to be encountered when processing function symbols. The start address
				// is rounded against the final alignment of the text section later on in
				// (*Link).address. This may happen due to usage of PCALIGN directives
				// larger than Funcalign, or usage of ISA 3.1 prefixed instructions
				// (see ISA 3.1 Book I 1.9).
				const ppc64maxFuncalign = 64
				sectAlign = ppc64maxFuncalign
				va = uint64(Rnd(int64(va), ppc64maxFuncalign))
			}

			// Set the length for the previous text section
			sect.Length = va - sect.Vaddr

			// Create new section, set the starting Vaddr
			sect = addsection(ctxt.loader, ctxt.Arch, &Segtext, ".text", 05)

			sect.Vaddr = va
			sect.Align = sectAlign
			ldr.SetSymSect(s, sect)

			// Create a symbol for the start of the secondary text sections
			ntext := ldr.CreateSymForUpdate(fmt.Sprintf("runtime.text.%d", n), 0)
			ntext.SetSect(sect)
			if ctxt.IsAIX() {
				// runtime.text.X must be a real symbol on AIX.
				// Assign its address directly in order to be the
				// first symbol of this new section.
				ntext.SetType(sym.STEXT)
				ntext.SetSize(int64(MINFUNC))
				ntext.SetOnList(true)
				ntext.SetAlign(sectAlign)
				ctxt.tramps = append(ctxt.tramps, ntext.Sym())

				ntext.SetValue(int64(va))
				va += uint64(ntext.Size())

				if align := ldr.SymAlign(s); align != 0 {
					va = uint64(Rnd(int64(va), int64(align)))
				} else {
					va = uint64(Rnd(int64(va), int64(Funcalign)))
				}
			}
			n++
		}
	}

	ldr.SetSymValue(s, 0)
	for sub := s; sub != 0; sub = ldr.SubSym(sub) {
		ldr.SetSymValue(sub, ldr.SymValue(sub)+int64(va))
		if ctxt.Debugvlog > 2 {
			fmt.Println("assign text address:", ldr.SymName(sub), ldr.SymValue(sub))
		}
	}

	va += funcsize

	return sect, n, va
}

func resetAddress(ctxt *Link, s loader.Sym) {
	ldr := ctxt.loader
	if ldr.OuterSym(s) != 0 {
		return
	}
	oldv := ldr.SymValue(s)
	for sub := s; sub != 0; sub = ldr.SubSym(sub) {
		ldr.SetSymValue(sub, ldr.SymValue(sub)-oldv)
	}
}

// Return whether we may need to split text sections.
//
// On PPC64x, when external linking, a text section should not be
// larger than 2^25 bytes due to the size of call target offset field
// in the 'bl' instruction. Splitting into smaller text sections
// smaller than this limit allows the system linker to modify the long
// calls appropriately. The limit allows for the space needed for
// tables inserted by the linker.
//
// The same applies to Darwin/ARM64, with 2^27 byte threshold.
//
// Similarly for ARM, we split sections (at 2^25 bytes) to avoid
// inconsistencies between the Go linker's reachability calculations
// (e.g. will direct call from X to Y need a trampoline) and similar
// machinery in the external linker; see #58425 for more on the
// history here.
func splitTextSections(ctxt *Link) bool {
	return (ctxt.IsARM() || ctxt.IsPPC64() || (ctxt.IsARM64() && ctxt.IsDarwin())) && ctxt.IsExternal()
}

// On Wasm, we reserve 4096 bytes for zero page, then 8192 bytes for wasm_exec.js
// to store command line args and environment variables.
// Data sections starts from at least address 12288.
// Keep in sync with wasm_exec.js.
const wasmMinDataAddr = 4096 + 8192

// address assigns virtual addresses to all segments and sections and
// returns all segments in file order.
func (ctxt *Link) address() []*sym.Segment {
	var order []*sym.Segment // Layout order

	va := uint64(*FlagTextAddr)
	order = append(order, &Segtext)
	Segtext.Rwx = 05
	Segtext.Vaddr = va
	for i, s := range Segtext.Sections {
		va = uint64(Rnd(int64(va), int64(s.Align)))
		s.Vaddr = va
		va += s.Length

		if ctxt.IsWasm() && i == 0 && va < wasmMinDataAddr {
			va = wasmMinDataAddr
		}
	}

	Segtext.Length = va - uint64(*FlagTextAddr)

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
		va = uint64(Rnd(int64(va), *FlagRound))

		order = append(order, &Segrodata)
		Segrodata.Rwx = 04
		Segrodata.Vaddr = va
		for _, s := range Segrodata.Sections {
			va = uint64(Rnd(int64(va), int64(s.Align)))
			s.Vaddr = va
			va += s.Length
		}

		Segrodata.Length = va - Segrodata.Vaddr
	}
	if len(Segrelrodata.Sections) > 0 {
		// align to page boundary so as not to mix
		// rodata, rel-ro data, and executable text.
		va = uint64(Rnd(int64(va), *FlagRound))
		if ctxt.HeadType == objabi.Haix {
			// Relro data are inside data segment on AIX.
			va += uint64(XCOFFDATABASE) - uint64(XCOFFTEXTBASE)
		}

		order = append(order, &Segrelrodata)
		Segrelrodata.Rwx = 06
		Segrelrodata.Vaddr = va
		for _, s := range Segrelrodata.Sections {
			va = uint64(Rnd(int64(va), int64(s.Align)))
			s.Vaddr = va
			va += s.Length
		}

		Segrelrodata.Length = va - Segrelrodata.Vaddr
	}

	va = uint64(Rnd(int64(va), *FlagRound))
	if ctxt.HeadType == objabi.Haix && len(Segrelrodata.Sections) == 0 {
		// Data sections are moved to an unreachable segment
		// to ensure that they are position-independent.
		// Already done if relro sections exist.
		va += uint64(XCOFFDATABASE) - uint64(XCOFFTEXTBASE)
	}
	order = append(order, &Segdata)
	Segdata.Rwx = 06
	Segdata.Vaddr = va
	var data *sym.Section
	var noptr *sym.Section
	var bss *sym.Section
	var noptrbss *sym.Section
	var fuzzCounters *sym.Section
	for i, s := range Segdata.Sections {
		if (ctxt.IsELF || ctxt.HeadType == objabi.Haix) && s.Name == ".tbss" {
			continue
		}
		vlen := int64(s.Length)
		if i+1 < len(Segdata.Sections) && !((ctxt.IsELF || ctxt.HeadType == objabi.Haix) && Segdata.Sections[i+1].Name == ".tbss") {
			vlen = int64(Segdata.Sections[i+1].Vaddr - s.Vaddr)
		}
		s.Vaddr = va
		va += uint64(vlen)
		Segdata.Length = va - Segdata.Vaddr
		switch s.Name {
		case ".data":
			data = s
		case ".noptrdata":
			noptr = s
		case ".bss":
			bss = s
		case ".noptrbss":
			noptrbss = s
		case ".go.fuzzcntrs":
			fuzzCounters = s
		}
	}

	// Assign Segdata's Filelen omitting the BSS. We do this here
	// simply because right now we know where the BSS starts.
	Segdata.Filelen = bss.Vaddr - Segdata.Vaddr

	if len(Segpdata.Sections) > 0 {
		va = uint64(Rnd(int64(va), *FlagRound))
		order = append(order, &Segpdata)
		Segpdata.Rwx = 04
		Segpdata.Vaddr = va
		// Segpdata.Sections is intended to contain just one section.
		// Loop through the slice anyway for consistency.
		for _, s := range Segpdata.Sections {
			va = uint64(Rnd(int64(va), int64(s.Align)))
			s.Vaddr = va
			va += s.Length
		}
		Segpdata.Length = va - Segpdata.Vaddr
	}

	if len(Segxdata.Sections) > 0 {
		va = uint64(Rnd(int64(va), *FlagRound))
		order = append(order, &Segxdata)
		Segxdata.Rwx = 04
		Segxdata.Vaddr = va
		// Segxdata.Sections is intended to contain just one section.
		// Loop through the slice anyway for consistency.
		for _, s := range Segxdata.Sections {
			va = uint64(Rnd(int64(va), int64(s.Align)))
			s.Vaddr = va
			va += s.Length
		}
		Segxdata.Length = va - Segxdata.Vaddr
	}

	va = uint64(Rnd(int64(va), *FlagRound))
	order = append(order, &Segdwarf)
	Segdwarf.Rwx = 06
	Segdwarf.Vaddr = va
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

	ldr := ctxt.loader
	var (
		rodata  = ldr.SymSect(ldr.LookupOrCreateSym("runtime.rodata", 0))
		symtab  = ldr.SymSect(ldr.LookupOrCreateSym("runtime.symtab", 0))
		pclntab = ldr.SymSect(ldr.LookupOrCreateSym("runtime.pclntab", 0))
		types   = ldr.SymSect(ldr.LookupOrCreateSym("runtime.types", 0))
	)

	for _, s := range ctxt.datap {
		if sect := ldr.SymSect(s); sect != nil {
			ldr.AddToSymValue(s, int64(sect.Vaddr))
		}
		v := ldr.SymValue(s)
		for sub := ldr.SubSym(s); sub != 0; sub = ldr.SubSym(sub) {
			ldr.AddToSymValue(sub, v)
		}
	}

	for _, si := range dwarfp {
		for _, s := range si.syms {
			if sect := ldr.SymSect(s); sect != nil {
				ldr.AddToSymValue(s, int64(sect.Vaddr))
			}
			sub := ldr.SubSym(s)
			if sub != 0 {
				panic(fmt.Sprintf("unexpected sub-sym for %s %s", ldr.SymName(s), ldr.SymType(s).String()))
			}
			v := ldr.SymValue(s)
			for ; sub != 0; sub = ldr.SubSym(sub) {
				ldr.AddToSymValue(s, v)
			}
		}
	}

	for _, s := range sehp.pdata {
		if sect := ldr.SymSect(s); sect != nil {
			ldr.AddToSymValue(s, int64(sect.Vaddr))
		}
	}
	for _, s := range sehp.xdata {
		if sect := ldr.SymSect(s); sect != nil {
			ldr.AddToSymValue(s, int64(sect.Vaddr))
		}
	}

	if ctxt.BuildMode == BuildModeShared {
		s := ldr.LookupOrCreateSym("go:link.abihashbytes", 0)
		sect := ldr.SymSect(ldr.LookupOrCreateSym(".note.go.abihash", 0))
		ldr.SetSymSect(s, sect)
		ldr.SetSymValue(s, int64(sect.Vaddr+16))
	}

	// If there are multiple text sections, create runtime.text.n for
	// their section Vaddr, using n for index
	n := 1
	for _, sect := range Segtext.Sections[1:] {
		if sect.Name != ".text" {
			break
		}
		symname := fmt.Sprintf("runtime.text.%d", n)
		if ctxt.HeadType != objabi.Haix || ctxt.LinkMode != LinkExternal {
			// Addresses are already set on AIX with external linker
			// because these symbols are part of their sections.
			ctxt.xdefine(symname, sym.STEXT, int64(sect.Vaddr))
		}
		n++
	}

	ctxt.xdefine("runtime.rodata", sym.SRODATA, int64(rodata.Vaddr))
	ctxt.xdefine("runtime.erodata", sym.SRODATA, int64(rodata.Vaddr+rodata.Length))
	ctxt.xdefine("runtime.types", sym.SRODATA, int64(types.Vaddr))
	ctxt.xdefine("runtime.etypes", sym.SRODATA, int64(types.Vaddr+types.Length))

	s := ldr.Lookup("runtime.gcdata", 0)
	ldr.SetAttrLocal(s, true)
	ctxt.xdefine("runtime.egcdata", sym.SRODATA, ldr.SymAddr(s)+ldr.SymSize(s))
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.egcdata", 0), ldr.SymSect(s))

	s = ldr.LookupOrCreateSym("runtime.gcbss", 0)
	ldr.SetAttrLocal(s, true)
	ctxt.xdefine("runtime.egcbss", sym.SRODATA, ldr.SymAddr(s)+ldr.SymSize(s))
	ldr.SetSymSect(ldr.LookupOrCreateSym("runtime.egcbss", 0), ldr.SymSect(s))

	ctxt.xdefine("runtime.symtab", sym.SRODATA, int64(symtab.Vaddr))
	ctxt.xdefine("runtime.esymtab", sym.SRODATA, int64(symtab.Vaddr+symtab.Length))
	ctxt.xdefine("runtime.pclntab", sym.SRODATA, int64(pclntab.Vaddr))
	ctxt.defineInternal("runtime.pcheader", sym.SRODATA)
	ctxt.defineInternal("runtime.funcnametab", sym.SRODATA)
	ctxt.defineInternal("runtime.cutab", sym.SRODATA)
	ctxt.defineInternal("runtime.filetab", sym.SRODATA)
	ctxt.defineInternal("runtime.pctab", sym.SRODATA)
	ctxt.defineInternal("runtime.functab", sym.SRODATA)
	ctxt.xdefine("runtime.epclntab", sym.SRODATA, int64(pclntab.Vaddr+pclntab.Length))
	ctxt.xdefine("runtime.noptrdata", sym.SNOPTRDATA, int64(noptr.Vaddr))
	ctxt.xdefine("runtime.enoptrdata", sym.SNOPTRDATA, int64(noptr.Vaddr+noptr.Length))
	ctxt.xdefine("runtime.bss", sym.SBSS, int64(bss.Vaddr))
	ctxt.xdefine("runtime.ebss", sym.SBSS, int64(bss.Vaddr+bss.Length))
	ctxt.xdefine("runtime.data", sym.SDATA, int64(data.Vaddr))
	ctxt.xdefine("runtime.edata", sym.SDATA, int64(data.Vaddr+data.Length))
	ctxt.xdefine("runtime.noptrbss", sym.SNOPTRBSS, int64(noptrbss.Vaddr))
	ctxt.xdefine("runtime.enoptrbss", sym.SNOPTRBSS, int64(noptrbss.Vaddr+noptrbss.Length))
	ctxt.xdefine("runtime.covctrs", sym.SCOVERAGE_COUNTER, int64(noptrbss.Vaddr+covCounterDataStartOff))
	ctxt.xdefine("runtime.ecovctrs", sym.SCOVERAGE_COUNTER, int64(noptrbss.Vaddr+covCounterDataStartOff+covCounterDataLen))
	ctxt.xdefine("runtime.end", sym.SBSS, int64(Segdata.Vaddr+Segdata.Length))

	if fuzzCounters != nil {
		ctxt.xdefine("runtime.__start___sancov_cntrs", sym.SLIBFUZZER_8BIT_COUNTER, int64(fuzzCounters.Vaddr))
		ctxt.xdefine("runtime.__stop___sancov_cntrs", sym.SLIBFUZZER_8BIT_COUNTER, int64(fuzzCounters.Vaddr+fuzzCounters.Length))
		ctxt.xdefine("internal/fuzz._counters", sym.SLIBFUZZER_8BIT_COUNTER, int64(fuzzCounters.Vaddr))
		ctxt.xdefine("internal/fuzz._ecounters", sym.SLIBFUZZER_8BIT_COUNTER, int64(fuzzCounters.Vaddr+fuzzCounters.Length))
	}

	if ctxt.IsSolaris() {
		// On Solaris, in the runtime it sets the external names of the
		// end symbols. Unset them and define separate symbols, so we
		// keep both.
		etext := ldr.Lookup("runtime.etext", 0)
		edata := ldr.Lookup("runtime.edata", 0)
		end := ldr.Lookup("runtime.end", 0)
		ldr.SetSymExtname(etext, "runtime.etext")
		ldr.SetSymExtname(edata, "runtime.edata")
		ldr.SetSymExtname(end, "runtime.end")
		ctxt.xdefine("_etext", ldr.SymType(etext), ldr.SymValue(etext))
		ctxt.xdefine("_edata", ldr.SymType(edata), ldr.SymValue(edata))
		ctxt.xdefine("_end", ldr.SymType(end), ldr.SymValue(end))
		ldr.SetSymSect(ldr.Lookup("_etext", 0), ldr.SymSect(etext))
		ldr.SetSymSect(ldr.Lookup("_edata", 0), ldr.SymSect(edata))
		ldr.SetSymSect(ldr.Lookup("_end", 0), ldr.SymSect(end))
	}

	if ctxt.IsPPC64() && ctxt.IsElf() {
		// Resolve .TOC. symbols for all objects. Only one TOC region is supported. If a
		// GOT section is present, compute it as suggested by the ELFv2 ABI. Otherwise,
		// choose a similar offset from the start of the data segment.
		tocAddr := int64(Segdata.Vaddr) + 0x8000
		if gotAddr := ldr.SymValue(ctxt.GOT); gotAddr != 0 {
			tocAddr = gotAddr + 0x8000
		}
		for i := range ctxt.DotTOC {
			if i >= sym.SymVerABICount && i < sym.SymVerStatic { // these versions are not used currently
				continue
			}
			if toc := ldr.Lookup(".TOC.", i); toc != 0 {
				ldr.SetSymValue(toc, tocAddr)
			}
		}
	}

	return order
}

// layout assigns file offsets and lengths to the segments in order.
// Returns the file size containing all the segments.
func (ctxt *Link) layout(order []*sym.Segment) uint64 {
	var prev *sym.Segment
	for _, seg := range order {
		if prev == nil {
			seg.Fileoff = uint64(HEADR)
		} else {
			switch ctxt.HeadType {
			default:
				// Assuming the previous segment was
				// aligned, the following rounding
				// should ensure that this segment's
				// VA ≡ Fileoff mod FlagRound.
				seg.Fileoff = uint64(Rnd(int64(prev.Fileoff+prev.Filelen), *FlagRound))
				if seg.Vaddr%uint64(*FlagRound) != seg.Fileoff%uint64(*FlagRound) {
					Exitf("bad segment rounding (Vaddr=%#x Fileoff=%#x FlagRound=%#x)", seg.Vaddr, seg.Fileoff, *FlagRound)
				}
			case objabi.Hwindows:
				seg.Fileoff = prev.Fileoff + uint64(Rnd(int64(prev.Filelen), PEFILEALIGN))
			case objabi.Hplan9:
				seg.Fileoff = prev.Fileoff + prev.Filelen
			}
		}
		if seg != &Segdata {
			// Link.address already set Segdata.Filelen to
			// account for BSS.
			seg.Filelen = seg.Length
		}
		prev = seg
	}
	return prev.Fileoff + prev.Filelen
}

// add a trampoline with symbol s (to be laid down after the current function)
func (ctxt *Link) AddTramp(s *loader.SymbolBuilder) {
	s.SetType(sym.STEXT)
	s.SetReachable(true)
	s.SetOnList(true)
	ctxt.tramps = append(ctxt.tramps, s.Sym())
	if *FlagDebugTramp > 0 && ctxt.Debugvlog > 0 {
		ctxt.Logf("trampoline %s inserted\n", s.Name())
	}
}

// compressSyms compresses syms and returns the contents of the
// compressed section. If the section would get larger, it returns nil.
func compressSyms(ctxt *Link, syms []loader.Sym) []byte {
	ldr := ctxt.loader
	var total int64
	for _, sym := range syms {
		total += ldr.SymSize(sym)
	}

	var buf bytes.Buffer
	if ctxt.IsELF {
		switch ctxt.Arch.PtrSize {
		case 8:
			binary.Write(&buf, ctxt.Arch.ByteOrder, elf.Chdr64{
				Type:      uint32(elf.COMPRESS_ZLIB),
				Size:      uint64(total),
				Addralign: uint64(ctxt.Arch.Alignment),
			})
		case 4:
			binary.Write(&buf, ctxt.Arch.ByteOrder, elf.Chdr32{
				Type:      uint32(elf.COMPRESS_ZLIB),
				Size:      uint32(total),
				Addralign: uint32(ctxt.Arch.Alignment),
			})
		default:
			log.Fatalf("can't compress header size:%d", ctxt.Arch.PtrSize)
		}
	} else {
		buf.Write([]byte("ZLIB"))
		var sizeBytes [8]byte
		binary.BigEndian.PutUint64(sizeBytes[:], uint64(total))
		buf.Write(sizeBytes[:])
	}

	var relocbuf []byte // temporary buffer for applying relocations

	// Using zlib.BestSpeed achieves very nearly the same
	// compression levels of zlib.DefaultCompression, but takes
	// substantially less time. This is important because DWARF
	// compression can be a significant fraction of link time.
	z, err := zlib.NewWriterLevel(&buf, zlib.BestSpeed)
	if err != nil {
		log.Fatalf("NewWriterLevel failed: %s", err)
	}
	st := ctxt.makeRelocSymState()
	for _, s := range syms {
		// Symbol data may be read-only. Apply relocations in a
		// temporary buffer, and immediately write it out.
		P := ldr.Data(s)
		relocs := ldr.Relocs(s)
		if relocs.Count() != 0 {
			relocbuf = append(relocbuf[:0], P...)
			P = relocbuf
			st.relocsym(s, P)
		}
		if _, err := z.Write(P); err != nil {
			log.Fatalf("compression failed: %s", err)
		}
		for i := ldr.SymSize(s) - int64(len(P)); i > 0; {
			b := zeros[:]
			if i < int64(len(b)) {
				b = b[:i]
			}
			n, err := z.Write(b)
			if err != nil {
				log.Fatalf("compression failed: %s", err)
			}
			i -= int64(n)
		}
	}
	if err := z.Close(); err != nil {
		log.Fatalf("compression failed: %s", err)
	}
	if int64(buf.Len()) >= total {
		// Compression didn't save any space.
		return nil
	}
	return buf.Bytes()
}
