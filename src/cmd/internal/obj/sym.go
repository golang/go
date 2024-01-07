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

package obj

import (
	"cmd/internal/goobj"
	"cmd/internal/notsha256"
	"cmd/internal/objabi"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"internal/buildcfg"
	"log"
	"math"
	"sort"
)

func Linknew(arch *LinkArch) *Link {
	ctxt := new(Link)
	ctxt.hash = make(map[string]*LSym)
	ctxt.funchash = make(map[string]*LSym)
	ctxt.statichash = make(map[string]*LSym)
	ctxt.Arch = arch
	ctxt.Pathname = objabi.WorkingDir()

	if err := ctxt.Headtype.Set(buildcfg.GOOS); err != nil {
		log.Fatalf("unknown goos %s", buildcfg.GOOS)
	}

	ctxt.Flag_optimize = true
	return ctxt
}

// LookupDerived looks up or creates the symbol with name derived from symbol s.
// The resulting symbol will be static iff s is.
func (ctxt *Link) LookupDerived(s *LSym, name string) *LSym {
	if s.Static() {
		return ctxt.LookupStatic(name)
	}
	return ctxt.Lookup(name)
}

// LookupStatic looks up the static symbol with name name.
// If it does not exist, it creates it.
func (ctxt *Link) LookupStatic(name string) *LSym {
	s := ctxt.statichash[name]
	if s == nil {
		s = &LSym{Name: name, Attribute: AttrStatic}
		ctxt.statichash[name] = s
	}
	return s
}

// LookupABI looks up a symbol with the given ABI.
// If it does not exist, it creates it.
func (ctxt *Link) LookupABI(name string, abi ABI) *LSym {
	return ctxt.LookupABIInit(name, abi, nil)
}

// LookupABIInit looks up a symbol with the given ABI.
// If it does not exist, it creates it and
// passes it to init for one-time initialization.
func (ctxt *Link) LookupABIInit(name string, abi ABI, init func(s *LSym)) *LSym {
	var hash map[string]*LSym
	switch abi {
	case ABI0:
		hash = ctxt.hash
	case ABIInternal:
		hash = ctxt.funchash
	default:
		panic("unknown ABI")
	}

	ctxt.hashmu.Lock()
	s := hash[name]
	if s == nil {
		s = &LSym{Name: name}
		s.SetABI(abi)
		hash[name] = s
		if init != nil {
			init(s)
		}
	}
	ctxt.hashmu.Unlock()
	return s
}

// Lookup looks up the symbol with name name.
// If it does not exist, it creates it.
func (ctxt *Link) Lookup(name string) *LSym {
	return ctxt.LookupInit(name, nil)
}

// LookupInit looks up the symbol with name name.
// If it does not exist, it creates it and
// passes it to init for one-time initialization.
func (ctxt *Link) LookupInit(name string, init func(s *LSym)) *LSym {
	ctxt.hashmu.Lock()
	s := ctxt.hash[name]
	if s == nil {
		s = &LSym{Name: name}
		ctxt.hash[name] = s
		if init != nil {
			init(s)
		}
	}
	ctxt.hashmu.Unlock()
	return s
}

func (ctxt *Link) Float32Sym(f float32) *LSym {
	i := math.Float32bits(f)
	name := fmt.Sprintf("$f32.%08x", i)
	return ctxt.LookupInit(name, func(s *LSym) {
		s.Size = 4
		s.WriteFloat32(ctxt, 0, f)
		s.Type = objabi.SRODATA
		s.Set(AttrLocal, true)
		s.Set(AttrContentAddressable, true)
		ctxt.constSyms = append(ctxt.constSyms, s)
	})
}

func (ctxt *Link) Float64Sym(f float64) *LSym {
	i := math.Float64bits(f)
	name := fmt.Sprintf("$f64.%016x", i)
	return ctxt.LookupInit(name, func(s *LSym) {
		s.Size = 8
		s.WriteFloat64(ctxt, 0, f)
		s.Type = objabi.SRODATA
		s.Set(AttrLocal, true)
		s.Set(AttrContentAddressable, true)
		ctxt.constSyms = append(ctxt.constSyms, s)
	})
}

func (ctxt *Link) Int32Sym(i int64) *LSym {
	name := fmt.Sprintf("$i32.%08x", uint64(i))
	return ctxt.LookupInit(name, func(s *LSym) {
		s.Size = 4
		s.WriteInt(ctxt, 0, 4, i)
		s.Type = objabi.SRODATA
		s.Set(AttrLocal, true)
		s.Set(AttrContentAddressable, true)
		ctxt.constSyms = append(ctxt.constSyms, s)
	})
}

func (ctxt *Link) Int64Sym(i int64) *LSym {
	name := fmt.Sprintf("$i64.%016x", uint64(i))
	return ctxt.LookupInit(name, func(s *LSym) {
		s.Size = 8
		s.WriteInt(ctxt, 0, 8, i)
		s.Type = objabi.SRODATA
		s.Set(AttrLocal, true)
		s.Set(AttrContentAddressable, true)
		ctxt.constSyms = append(ctxt.constSyms, s)
	})
}

func (ctxt *Link) Int128Sym(hi, lo int64) *LSym {
	name := fmt.Sprintf("$i128.%016x%016x", uint64(hi), uint64(lo))
	return ctxt.LookupInit(name, func(s *LSym) {
		s.Size = 16
		if ctxt.Arch.ByteOrder == binary.LittleEndian {
			s.WriteInt(ctxt, 0, 8, lo)
			s.WriteInt(ctxt, 8, 8, hi)
		} else {
			s.WriteInt(ctxt, 0, 8, hi)
			s.WriteInt(ctxt, 8, 8, lo)
		}
		s.Type = objabi.SRODATA
		s.Set(AttrLocal, true)
		s.Set(AttrContentAddressable, true)
		ctxt.constSyms = append(ctxt.constSyms, s)
	})
}

// GCLocalsSym generates a content-addressable sym containing data.
func (ctxt *Link) GCLocalsSym(data []byte) *LSym {
	sum := notsha256.Sum256(data)
	str := base64.StdEncoding.EncodeToString(sum[:16])
	return ctxt.LookupInit(fmt.Sprintf("gclocals·%s", str), func(lsym *LSym) {
		lsym.P = data
		lsym.Set(AttrContentAddressable, true)
	})
}

// Assign index to symbols.
// asm is set to true if this is called by the assembler (i.e. not the compiler),
// in which case all the symbols are non-package (for now).
func (ctxt *Link) NumberSyms() {
	if ctxt.Pkgpath == "" {
		panic("NumberSyms called without package path")
	}

	if ctxt.Headtype == objabi.Haix {
		// Data must be in a reliable order for reproducible builds.
		// The original entries are in a reliable order, but the TOC symbols
		// that are added in Progedit are added by different goroutines
		// that can be scheduled independently. We need to reorder those
		// symbols reliably. Sort by name but use a stable sort, so that
		// any original entries with the same name (all DWARFVAR symbols
		// have empty names but different relocation sets) are not shuffled.
		// TODO: Find a better place and optimize to only sort TOC symbols.
		sort.SliceStable(ctxt.Data, func(i, j int) bool {
			return ctxt.Data[i].Name < ctxt.Data[j].Name
		})
	}

	// Constant symbols are created late in the concurrent phase. Sort them
	// to ensure a deterministic order.
	sort.Slice(ctxt.constSyms, func(i, j int) bool {
		return ctxt.constSyms[i].Name < ctxt.constSyms[j].Name
	})
	ctxt.Data = append(ctxt.Data, ctxt.constSyms...)
	ctxt.constSyms = nil

	ctxt.pkgIdx = make(map[string]int32)
	ctxt.defs = []*LSym{}
	ctxt.hashed64defs = []*LSym{}
	ctxt.hasheddefs = []*LSym{}
	ctxt.nonpkgdefs = []*LSym{}

	var idx, hashedidx, hashed64idx, nonpkgidx int32
	ctxt.traverseSyms(traverseDefs|traversePcdata, func(s *LSym) {
		if s.ContentAddressable() {
			if s.Size <= 8 && len(s.R) == 0 && contentHashSection(s) == 0 {
				// We can use short hash only for symbols without relocations.
				// Don't use short hash for symbols that belong in a particular section
				// or require special handling (such as type symbols).
				s.PkgIdx = goobj.PkgIdxHashed64
				s.SymIdx = hashed64idx
				if hashed64idx != int32(len(ctxt.hashed64defs)) {
					panic("bad index")
				}
				ctxt.hashed64defs = append(ctxt.hashed64defs, s)
				hashed64idx++
			} else {
				s.PkgIdx = goobj.PkgIdxHashed
				s.SymIdx = hashedidx
				if hashedidx != int32(len(ctxt.hasheddefs)) {
					panic("bad index")
				}
				ctxt.hasheddefs = append(ctxt.hasheddefs, s)
				hashedidx++
			}
		} else if isNonPkgSym(ctxt, s) {
			s.PkgIdx = goobj.PkgIdxNone
			s.SymIdx = nonpkgidx
			if nonpkgidx != int32(len(ctxt.nonpkgdefs)) {
				panic("bad index")
			}
			ctxt.nonpkgdefs = append(ctxt.nonpkgdefs, s)
			nonpkgidx++
		} else {
			s.PkgIdx = goobj.PkgIdxSelf
			s.SymIdx = idx
			if idx != int32(len(ctxt.defs)) {
				panic("bad index")
			}
			ctxt.defs = append(ctxt.defs, s)
			idx++
		}
		s.Set(AttrIndexed, true)
	})

	ipkg := int32(1) // 0 is invalid index
	nonpkgdef := nonpkgidx
	ctxt.traverseSyms(traverseRefs|traverseAux, func(rs *LSym) {
		if rs.PkgIdx != goobj.PkgIdxInvalid {
			return
		}
		if !ctxt.Flag_linkshared {
			// Assign special index for builtin symbols.
			// Don't do it when linking against shared libraries, as the runtime
			// may be in a different library.
			if i := goobj.BuiltinIdx(rs.Name, int(rs.ABI())); i != -1 {
				rs.PkgIdx = goobj.PkgIdxBuiltin
				rs.SymIdx = int32(i)
				rs.Set(AttrIndexed, true)
				return
			}
		}
		pkg := rs.Pkg
		if rs.ContentAddressable() {
			// for now, only support content-addressable symbols that are always locally defined.
			panic("hashed refs unsupported for now")
		}
		if pkg == "" || pkg == "\"\"" || pkg == "_" || !rs.Indexed() {
			rs.PkgIdx = goobj.PkgIdxNone
			rs.SymIdx = nonpkgidx
			rs.Set(AttrIndexed, true)
			if nonpkgidx != nonpkgdef+int32(len(ctxt.nonpkgrefs)) {
				panic("bad index")
			}
			ctxt.nonpkgrefs = append(ctxt.nonpkgrefs, rs)
			nonpkgidx++
			return
		}
		if k, ok := ctxt.pkgIdx[pkg]; ok {
			rs.PkgIdx = k
			return
		}
		rs.PkgIdx = ipkg
		ctxt.pkgIdx[pkg] = ipkg
		ipkg++
	})
}

// Returns whether s is a non-package symbol, which needs to be referenced
// by name instead of by index.
func isNonPkgSym(ctxt *Link, s *LSym) bool {
	if ctxt.IsAsm && !s.Static() {
		// asm symbols are referenced by name only, except static symbols
		// which are file-local and can be referenced by index.
		return true
	}
	if ctxt.Flag_linkshared {
		// The referenced symbol may be in a different shared library so
		// the linker cannot see its index.
		return true
	}
	if s.Pkg == "_" {
		// The frontend uses package "_" to mark symbols that should not
		// be referenced by index, e.g. linkname'd symbols.
		return true
	}
	if s.DuplicateOK() {
		// Dupok symbol needs to be dedup'd by name.
		return true
	}
	return false
}

// StaticNamePref is the prefix the front end applies to static temporary
// variables. When turned into LSyms, these can be tagged as static so
// as to avoid inserting them into the linker's name lookup tables.
const StaticNamePref = ".stmp_"

type traverseFlag uint32

const (
	traverseDefs traverseFlag = 1 << iota
	traverseRefs
	traverseAux
	traversePcdata

	traverseAll = traverseDefs | traverseRefs | traverseAux | traversePcdata
)

// Traverse symbols based on flag, call fn for each symbol.
func (ctxt *Link) traverseSyms(flag traverseFlag, fn func(*LSym)) {
	fnNoNil := func(s *LSym) {
		if s != nil {
			fn(s)
		}
	}
	lists := [][]*LSym{ctxt.Text, ctxt.Data}
	files := ctxt.PosTable.FileTable()
	for _, list := range lists {
		for _, s := range list {
			if flag&traverseDefs != 0 {
				fn(s)
			}
			if flag&traverseRefs != 0 {
				for _, r := range s.R {
					fnNoNil(r.Sym)
				}
			}
			if flag&traverseAux != 0 {
				fnNoNil(s.Gotype)
				if s.Type == objabi.STEXT {
					f := func(parent *LSym, aux *LSym) {
						fn(aux)
					}
					ctxt.traverseFuncAux(flag, s, f, files)
				} else if v := s.VarInfo(); v != nil {
					fnNoNil(v.dwarfInfoSym)
				}
			}
			if flag&traversePcdata != 0 && s.Type == objabi.STEXT {
				fi := s.Func().Pcln
				fnNoNil(fi.Pcsp)
				fnNoNil(fi.Pcfile)
				fnNoNil(fi.Pcline)
				fnNoNil(fi.Pcinline)
				for _, d := range fi.Pcdata {
					fnNoNil(d)
				}
			}
		}
	}
}

func (ctxt *Link) traverseFuncAux(flag traverseFlag, fsym *LSym, fn func(parent *LSym, aux *LSym), files []string) {
	fninfo := fsym.Func()
	pc := &fninfo.Pcln
	if flag&traverseAux == 0 {
		// NB: should it become necessary to walk aux sym reloc references
		// without walking the aux syms themselves, this can be changed.
		panic("should not be here")
	}
	for _, d := range pc.Funcdata {
		if d != nil {
			fn(fsym, d)
		}
	}
	usedFiles := make([]goobj.CUFileIndex, 0, len(pc.UsedFiles))
	for f := range pc.UsedFiles {
		usedFiles = append(usedFiles, f)
	}
	sort.Slice(usedFiles, func(i, j int) bool { return usedFiles[i] < usedFiles[j] })
	for _, f := range usedFiles {
		if filesym := ctxt.Lookup(files[f]); filesym != nil {
			fn(fsym, filesym)
		}
	}
	for _, call := range pc.InlTree.nodes {
		if call.Func != nil {
			fn(fsym, call.Func)
		}
	}

	auxsyms := []*LSym{fninfo.dwarfRangesSym, fninfo.dwarfLocSym, fninfo.dwarfDebugLinesSym, fninfo.dwarfInfoSym, fninfo.WasmImportSym, fninfo.sehUnwindInfoSym}
	for _, s := range auxsyms {
		if s == nil || s.Size == 0 {
			continue
		}
		fn(fsym, s)
		if flag&traverseRefs != 0 {
			for _, r := range s.R {
				if r.Sym != nil {
					fn(s, r.Sym)
				}
			}
		}
	}
}

// Traverse aux symbols, calling fn for each sym/aux pair.
func (ctxt *Link) traverseAuxSyms(flag traverseFlag, fn func(parent *LSym, aux *LSym)) {
	lists := [][]*LSym{ctxt.Text, ctxt.Data}
	files := ctxt.PosTable.FileTable()
	for _, list := range lists {
		for _, s := range list {
			if s.Gotype != nil {
				if flag&traverseDefs != 0 {
					fn(s, s.Gotype)
				}
			}
			if s.Type == objabi.STEXT {
				ctxt.traverseFuncAux(flag, s, fn, files)
			} else if v := s.VarInfo(); v != nil && v.dwarfInfoSym != nil {
				fn(s, v.dwarfInfoSym)
			}
		}
	}
}
