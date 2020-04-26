// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"log"
	"strings"
	"sync"
)

// Temporary dumping around for sym.Symbol version of helper
// functions in dodata(), still being used for some archs/oses.
// FIXME: get rid of this file when dodata() is completely
// converted.

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
	s.Align = align
	return align
}

func relocsym2(target *Target, ldr *loader.Loader, err *ErrorReporter, syms *ArchSyms, s *sym.Symbol) {
	if len(s.R) == 0 {
		return
	}
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
			if target.IsShared() || target.IsPlugin() {
				if r.Sym.Name == "main.main" || (!target.IsPlugin() && r.Sym.Name == "main..inittask") {
					r.Sym.Type = sym.SDYNIMPORT
				} else if strings.HasPrefix(r.Sym.Name, "go.info.") {
					// Skip go.info symbols. They are only needed to communicate
					// DWARF info between the compiler and linker.
					continue
				}
			} else {
				err.errorUnresolved2(s, r)
				continue
			}
		}

		if r.Type >= objabi.ElfRelocOffset {
			continue
		}
		if r.Siz == 0 { // informational relocation - no work to do
			continue
		}

		// We need to be able to reference dynimport symbols when linking against
		// shared libraries, and Solaris, Darwin and AIX need it always
		if !target.IsSolaris() && !target.IsDarwin() && !target.IsAIX() && r.Sym != nil && r.Sym.Type == sym.SDYNIMPORT && !target.IsDynlinkingGo() && !r.Sym.Attr.SubSymbol() {
			if !(target.IsPPC64() && target.IsExternal() && r.Sym.Name == ".TOC.") {
				Errorf(s, "unhandled relocation for %s (type %d (%s) rtype %d (%s))", r.Sym.Name, r.Sym.Type, r.Sym.Type, r.Type, sym.RelocName(target.Arch, r.Type))
			}
		}
		if r.Sym != nil && r.Sym.Type != sym.STLSBSS && r.Type != objabi.R_WEAKADDROFF && !r.Sym.Attr.Reachable() {
			Errorf(s, "unreachable sym in relocation: %s", r.Sym.Name)
		}

		if target.IsExternal() {
			r.InitExt()
		}

		// TODO(mundaym): remove this special case - see issue 14218.
		if target.IsS390X() {
			switch r.Type {
			case objabi.R_PCRELDBL:
				r.InitExt()
				r.Type = objabi.R_PCREL
				r.Variant = sym.RV_390_DBL
			case objabi.R_CALL:
				r.InitExt()
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
				o = int64(target.Arch.ByteOrder.Uint16(s.P[off:]))
			case 4:
				o = int64(target.Arch.ByteOrder.Uint32(s.P[off:]))
			case 8:
				o = int64(target.Arch.ByteOrder.Uint64(s.P[off:]))
			}
			if offset, ok := thearch.Archreloc(target, syms, r, s, o); ok {
				o = offset
			} else {
				Errorf(s, "unknown reloc to %v: %d (%s)", r.Sym.Name, r.Type, sym.RelocName(target.Arch, r.Type))
			}
		case objabi.R_TLS_LE:
			if target.IsExternal() && target.IsElf() {
				r.Done = false
				if r.Sym == nil {
					r.Sym = syms.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if !target.IsAMD64() {
					o = r.Add
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
				o = 8 + r.Sym.Value
			} else if target.IsElf() || target.IsPlan9() || target.IsDarwin() {
				o = int64(syms.Tlsoffset) + r.Add
			} else if target.IsWindows() {
				o = r.Add
			} else {
				log.Fatalf("unexpected R_TLS_LE relocation for %v", target.HeadType)
			}
		case objabi.R_TLS_IE:
			if target.IsExternal() && target.IsElf() {
				r.Done = false
				if r.Sym == nil {
					r.Sym = syms.Tlsg
				}
				r.Xsym = r.Sym
				r.Xadd = r.Add
				o = 0
				if !target.IsAMD64() {
					o = r.Add
				}
				break
			}
			if target.IsPIE() && target.IsElf() {
				// We are linking the final executable, so we
				// can optimize any TLS IE relocation to LE.
				if thearch.TLSIEtoLE == nil {
					log.Fatalf("internal linking of TLS IE not supported on %v", target.Arch.Family)
				}
				thearch.TLSIEtoLE(s.P, int(off), int(r.Siz))
				o = int64(syms.Tlsoffset)
				// TODO: o += r.Add when !target.IsAmd64()?
				// Why do we treat r.Add differently on AMD64?
				// Is the external linker using Xadd at all?
			} else {
				log.Fatalf("cannot handle R_TLS_IE (sym %s) when linking internally", s.Name)
			}
		case objabi.R_ADDR:
			if target.IsExternal() && r.Sym.Type != sym.SCONST {
				r.Done = false

				// set up addend for eventual relocation via outer symbol.
				rs := r.Sym

				r.Xadd = r.Add
				for rs.Outer != nil {
					r.Xadd += Symaddr(rs) - Symaddr(rs.Outer)
					rs = rs.Outer
				}

				if rs.Type != sym.SHOSTOBJ && rs.Type != sym.SDYNIMPORT && rs.Type != sym.SUNDEFEXT && rs.Sect == nil {
					Errorf(s, "missing section for relocation target %s", rs.Name)
				}
				r.Xsym = rs

				o = r.Xadd
				if target.IsElf() {
					if target.IsAMD64() {
						o = 0
					}
				} else if target.IsDarwin() {
					if rs.Type != sym.SHOSTOBJ {
						o += Symaddr(rs)
					}
				} else if target.IsWindows() {
					// nothing to do
				} else if target.IsAIX() {
					o = Symaddr(r.Sym) + r.Add
				} else {
					Errorf(s, "unhandled pcrel relocation to %s on %v", rs.Name, target.HeadType)
				}

				break
			}

			// On AIX, a second relocation must be done by the loader,
			// as section addresses can change once loaded.
			// The "default" symbol address is still needed by the loader so
			// the current relocation can't be skipped.
			if target.IsAIX() && r.Sym.Type != sym.SDYNIMPORT {
				// It's not possible to make a loader relocation in a
				// symbol which is not inside .data section.
				// FIXME: It should be forbidden to have R_ADDR from a
				// symbol which isn't in .data. However, as .text has the
				// same address once loaded, this is possible.
				if s.Sect.Seg == &Segdata {
					Xcoffadddynrel(target, ldr, s, r)
				}
			}

			o = Symaddr(r.Sym) + r.Add

			// On amd64, 4-byte offsets will be sign-extended, so it is impossible to
			// access more than 2GB of static data; fail at link time is better than
			// fail at runtime. See https://golang.org/issue/7980.
			// Instead of special casing only amd64, we treat this as an error on all
			// 64-bit architectures so as to be future-proof.
			if int32(o) < 0 && target.Arch.PtrSize > 4 && siz == 4 {
				Errorf(s, "non-pc-relative relocation address for %s is too big: %#x (%#x + %#x)", r.Sym.Name, uint64(o), Symaddr(r.Sym), r.Add)
				errorexit()
			}
		case objabi.R_DWARFSECREF:
			if r.Sym.Sect == nil {
				Errorf(s, "missing DWARF section for relocation target %s", r.Sym.Name)
			}

			if target.IsExternal() {
				r.Done = false

				// On most platforms, the external linker needs to adjust DWARF references
				// as it combines DWARF sections. However, on Darwin, dsymutil does the
				// DWARF linking, and it understands how to follow section offsets.
				// Leaving in the relocation records confuses it (see
				// https://golang.org/issue/22068) so drop them for Darwin.
				if target.IsDarwin() {
					r.Done = true
				}

				// PE code emits IMAGE_REL_I386_SECREL and IMAGE_REL_AMD64_SECREL
				// for R_DWARFSECREF relocations, while R_ADDR is replaced with
				// IMAGE_REL_I386_DIR32, IMAGE_REL_AMD64_ADDR64 and IMAGE_REL_AMD64_ADDR32.
				// Do not replace R_DWARFSECREF with R_ADDR for windows -
				// let PE code emit correct relocations.
				if !target.IsWindows() {
					r.Type = objabi.R_ADDR
				}

				r.Xsym = r.Sym.Sect.Sym
				r.Xadd = r.Add + Symaddr(r.Sym) - int64(r.Sym.Sect.Vaddr)

				o = r.Xadd
				if target.IsElf() && target.IsAMD64() {
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
			u := ldr.SymUnit(loader.Sym(r.Sym.SymIdx))
			o = Symaddr(r.Sym) + r.Add - Symaddr(ldr.Syms[u.Textp2[0]])

			// r->sym can be null when CALL $(constant) is transformed from absolute PC to relative PC call.
		case objabi.R_GOTPCREL:
			if target.IsDynlinkingGo() && target.IsDarwin() && r.Sym != nil && r.Sym.Type != sym.SCONST {
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
			if target.IsExternal() && r.Sym != nil && r.Sym.Type == sym.SUNDEFEXT {
				// pass through to the external linker.
				r.Done = false
				r.Xadd = 0
				if target.IsElf() {
					r.Xadd -= int64(r.Siz)
				}
				r.Xsym = r.Sym
				o = 0
				break
			}
			if target.IsExternal() && r.Sym != nil && r.Sym.Type != sym.SCONST && (r.Sym.Sect != s.Sect || r.Type == objabi.R_GOTPCREL) {
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
				if target.IsElf() {
					if target.IsAMD64() {
						o = 0
					}
				} else if target.IsDarwin() {
					if r.Type == objabi.R_CALL {
						if target.IsExternal() && rs.Type == sym.SDYNIMPORT {
							if target.IsAMD64() {
								// AMD64 dynamic relocations are relative to the end of the relocation.
								o += int64(r.Siz)
							}
						} else {
							if rs.Type != sym.SHOSTOBJ {
								o += int64(uint64(Symaddr(rs)) - rs.Sect.Vaddr)
							}
							o -= int64(r.Off) // relative to section offset, not symbol
						}
					} else {
						o += int64(r.Siz)
					}
				} else if target.IsWindows() && target.IsAMD64() { // only amd64 needs PCREL
					// PE/COFF's PC32 relocation uses the address after the relocated
					// bytes as the base. Compensate by skewing the addend.
					o += int64(r.Siz)
				} else {
					Errorf(s, "unhandled pcrel relocation to %s on %v", rs.Name, target.HeadType)
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

		case objabi.R_XCOFFREF:
			if !target.IsAIX() {
				Errorf(s, "find XCOFF R_REF on non-XCOFF files")
			}
			if !target.IsExternal() {
				Errorf(s, "find XCOFF R_REF with internal linking")
			}
			r.Xsym = r.Sym
			r.Xadd = r.Add
			r.Done = false

			// This isn't a real relocation so it must not update
			// its offset value.
			continue

		case objabi.R_DWARFFILEREF:
			// The final file index is saved in r.Add in dwarf.go:writelines.
			o = r.Add
		}

		if target.IsPPC64() || target.IsS390X() {
			r.InitExt()
			if r.Variant != sym.RV_NONE {
				o = thearch.Archrelocvariant(target, syms, r, s, o)
			}
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
			fmt.Printf("relocate %s %#x (%#x+%#x, size %d) => %s %#x +%#x (xsym: %s +%#x) [type %d (%s)/%d, %x]\n", s.Name, s.Value+int64(off), s.Value, r.Off, r.Siz, nam, addr, r.Add, xnam, r.Xadd, r.Type, sym.RelocName(target.Arch, r.Type), r.Variant, o)
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
			target.Arch.ByteOrder.PutUint16(s.P[off:], uint16(i16))
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
			target.Arch.ByteOrder.PutUint32(s.P[off:], uint32(fl))
		case 8:
			target.Arch.ByteOrder.PutUint64(s.P[off:], uint64(o))
		}
	}
}

func (ctxt *Link) reloc2() {
	var wg sync.WaitGroup
	target := &ctxt.Target
	ldr := ctxt.loader
	reporter := &ctxt.ErrorReporter
	syms := &ctxt.ArchSyms
	wg.Add(3)
	go func() {
		if !ctxt.IsWasm() { // On Wasm, text relocations are applied in Asmb2.
			for _, s := range ctxt.Textp {
				relocsym2(target, ldr, reporter, syms, s)
			}
		}
		wg.Done()
	}()
	go func() {
		for _, s := range ctxt.datap {
			relocsym2(target, ldr, reporter, syms, s)
		}
		wg.Done()
	}()
	go func() {
		for _, si := range dwarfp {
			for _, s := range si.syms {
				relocsym2(target, ldr, reporter, syms, s)
			}
		}
		wg.Done()
	}()
	wg.Wait()
}
