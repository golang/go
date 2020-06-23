// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj

import (
	"cmd/internal/goobj2"
	"cmd/internal/objabi"
	"strings"
)

// Read object file in new format. For now we still fill
// the data to the current goobj API.
func (r *objReader) readNew() {
	start := uint32(r.offset)

	length := r.limit - r.offset
	objbytes := make([]byte, length)
	r.readFull(objbytes)
	rr := goobj2.NewReaderFromBytes(objbytes, false)
	if rr == nil {
		panic("cannot read object file")
	}

	// Imports
	autolib := rr.Autolib()
	for _, p := range autolib {
		r.p.Imports = append(r.p.Imports, p.Pkg)
		// Ignore fingerprint (for tools like objdump which only reads one object).
	}

	// Name of referenced indexed symbols.
	nrefName := rr.NRefName()
	refNames := make(map[goobj2.SymRef]string, nrefName)
	for i := 0; i < nrefName; i++ {
		rn := rr.RefName(i)
		refNames[rn.Sym()] = rn.Name(rr)
	}

	abiToVer := func(abi uint16) int64 {
		var vers int64
		if abi == goobj2.SymABIstatic {
			// Static symbol
			vers = r.p.MaxVersion
		}
		return vers
	}

	resolveSymRef := func(s goobj2.SymRef) SymID {
		var i int
		switch p := s.PkgIdx; p {
		case goobj2.PkgIdxInvalid:
			if s.SymIdx != 0 {
				panic("bad sym ref")
			}
			return SymID{}
		case goobj2.PkgIdxNone:
			i = int(s.SymIdx) + rr.NSym()
		case goobj2.PkgIdxBuiltin:
			name, abi := goobj2.BuiltinName(int(s.SymIdx))
			return SymID{name, int64(abi)}
		case goobj2.PkgIdxSelf:
			i = int(s.SymIdx)
		default:
			return SymID{refNames[s], 0}
		}
		sym := rr.Sym(i)
		return SymID{sym.Name(rr), abiToVer(sym.ABI())}
	}

	// Read things for the current goobj API for now.

	// Symbols
	pcdataBase := start + rr.PcdataBase()
	n := rr.NSym() + rr.NNonpkgdef() + rr.NNonpkgref()
	ndef := rr.NSym() + rr.NNonpkgdef()
	for i := 0; i < n; i++ {
		osym := rr.Sym(i)
		if osym.Name(rr) == "" {
			continue // not a real symbol
		}
		// In a symbol name in an object file, "". denotes the
		// prefix for the package in which the object file has been found.
		// Expand it.
		name := strings.ReplaceAll(osym.Name(rr), `"".`, r.pkgprefix)
		symID := SymID{Name: name, Version: abiToVer(osym.ABI())}
		r.p.SymRefs = append(r.p.SymRefs, symID)

		if i >= ndef {
			continue // not a defined symbol from here
		}

		// Symbol data
		dataOff := rr.DataOff(i)
		siz := int64(rr.DataSize(i))

		sym := Sym{
			SymID: symID,
			Kind:  objabi.SymKind(osym.Type()),
			DupOK: osym.Dupok(),
			Size:  int64(osym.Siz()),
			Data:  Data{int64(start + dataOff), siz},
		}
		r.p.Syms = append(r.p.Syms, &sym)

		// Reloc
		relocs := rr.Relocs(i)
		sym.Reloc = make([]Reloc, len(relocs))
		for j := range relocs {
			rel := &relocs[j]
			sym.Reloc[j] = Reloc{
				Offset: int64(rel.Off()),
				Size:   int64(rel.Siz()),
				Type:   objabi.RelocType(rel.Type()),
				Add:    rel.Add(),
				Sym:    resolveSymRef(rel.Sym()),
			}
		}

		// Aux symbol info
		isym := -1
		funcdata := make([]goobj2.SymRef, 0, 4)
		auxs := rr.Auxs(i)
		for j := range auxs {
			a := &auxs[j]
			switch a.Type() {
			case goobj2.AuxGotype:
				sym.Type = resolveSymRef(a.Sym())
			case goobj2.AuxFuncInfo:
				if a.Sym().PkgIdx != goobj2.PkgIdxSelf {
					panic("funcinfo symbol not defined in current package")
				}
				isym = int(a.Sym().SymIdx)
			case goobj2.AuxFuncdata:
				funcdata = append(funcdata, a.Sym())
			case goobj2.AuxDwarfInfo, goobj2.AuxDwarfLoc, goobj2.AuxDwarfRanges, goobj2.AuxDwarfLines:
				// nothing to do
			default:
				panic("unknown aux type")
			}
		}

		// Symbol Info
		if isym == -1 {
			continue
		}
		b := rr.BytesAt(rr.DataOff(isym), rr.DataSize(isym))
		info := goobj2.FuncInfo{}
		info.Read(b)

		info.Pcdata = append(info.Pcdata, info.PcdataEnd) // for the ease of knowing where it ends
		f := &Func{
			Args:     int64(info.Args),
			Frame:    int64(info.Locals),
			NoSplit:  osym.NoSplit(),
			Leaf:     osym.Leaf(),
			TopFrame: osym.TopFrame(),
			PCSP:     Data{int64(pcdataBase + info.Pcsp), int64(info.Pcfile - info.Pcsp)},
			PCFile:   Data{int64(pcdataBase + info.Pcfile), int64(info.Pcline - info.Pcfile)},
			PCLine:   Data{int64(pcdataBase + info.Pcline), int64(info.Pcinline - info.Pcline)},
			PCInline: Data{int64(pcdataBase + info.Pcinline), int64(info.Pcdata[0] - info.Pcinline)},
			PCData:   make([]Data, len(info.Pcdata)-1), // -1 as we appended one above
			FuncData: make([]FuncData, len(info.Funcdataoff)),
			File:     make([]string, len(info.File)),
			InlTree:  make([]InlinedCall, len(info.InlTree)),
		}
		sym.Func = f
		for k := range f.PCData {
			f.PCData[k] = Data{int64(pcdataBase + info.Pcdata[k]), int64(info.Pcdata[k+1] - info.Pcdata[k])}
		}
		for k := range f.FuncData {
			symID := resolveSymRef(funcdata[k])
			f.FuncData[k] = FuncData{symID, int64(info.Funcdataoff[k])}
		}
		for k := range f.File {
			symID := resolveSymRef(info.File[k])
			f.File[k] = symID.Name
		}
		for k := range f.InlTree {
			inl := &info.InlTree[k]
			f.InlTree[k] = InlinedCall{
				Parent:   int64(inl.Parent),
				File:     resolveSymRef(inl.File).Name,
				Line:     int64(inl.Line),
				Func:     resolveSymRef(inl.Func),
				ParentPC: int64(inl.ParentPC),
			}
		}
	}
}
