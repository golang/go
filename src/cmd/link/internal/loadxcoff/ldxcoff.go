// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loadxcoff implements a XCOFF file reader.
package loadxcoff

import (
	"cmd/internal/bio"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"errors"
	"fmt"
	"internal/xcoff"
)

// ldSection is an XCOFF section with its symbols.
type ldSection struct {
	xcoff.Section
	sym loader.Sym
}

// TODO(brainman): maybe just add ReadAt method to bio.Reader instead of creating xcoffBiobuf

// xcoffBiobuf makes bio.Reader look like io.ReaderAt.
type xcoffBiobuf bio.Reader

func (f *xcoffBiobuf) ReadAt(p []byte, off int64) (int, error) {
	ret := ((*bio.Reader)(f)).MustSeek(off, 0)
	if ret < 0 {
		return 0, errors.New("fail to seek")
	}
	n, err := f.Read(p)
	if err != nil {
		return 0, err
	}
	return n, nil
}

// loads the Xcoff file pn from f.
// Symbols are written into loader, and a slice of the text symbols is returned.
func Load(l *loader.Loader, arch *sys.Arch, localSymVersion int, input *bio.Reader, pkg string, length int64, pn string) (textp []loader.Sym, err error) {
	errorf := func(str string, args ...any) ([]loader.Sym, error) {
		return nil, fmt.Errorf("loadxcoff: %v: %v", pn, fmt.Sprintf(str, args...))
	}

	var ldSections []*ldSection

	f, err := xcoff.NewFile((*xcoffBiobuf)(input))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	for _, sect := range f.Sections {
		//only text, data and bss section
		if sect.Type < xcoff.STYP_TEXT || sect.Type > xcoff.STYP_BSS {
			continue
		}
		lds := new(ldSection)
		lds.Section = *sect
		name := fmt.Sprintf("%s(%s)", pkg, lds.Name)
		symbol := l.LookupOrCreateSym(name, localSymVersion)
		s := l.MakeSymbolUpdater(symbol)

		switch lds.Type {
		default:
			return errorf("unrecognized section type 0x%x", lds.Type)
		case xcoff.STYP_TEXT:
			s.SetType(sym.STEXT)
		case xcoff.STYP_DATA:
			s.SetType(sym.SNOPTRDATA)
		case xcoff.STYP_BSS:
			s.SetType(sym.SNOPTRBSS)
		}

		s.SetSize(int64(lds.Size))
		if s.Type() != sym.SNOPTRBSS {
			data, err := lds.Section.Data()
			if err != nil {
				return nil, err
			}
			s.SetData(data)
		}

		lds.sym = symbol
		ldSections = append(ldSections, lds)
	}

	// sx = symbol from file
	// s = symbol for loader
	for _, sx := range f.Symbols {
		// get symbol type
		stype, errmsg := getSymbolType(f, sx)
		if errmsg != "" {
			return errorf("error reading symbol %s: %s", sx.Name, errmsg)
		}
		if stype == sym.Sxxx {
			continue
		}

		s := l.LookupOrCreateSym(sx.Name, 0)

		// Text symbol
		if l.SymType(s).IsText() {
			if l.AttrOnList(s) {
				return errorf("symbol %s listed multiple times", l.SymName(s))
			}
			l.SetAttrOnList(s, true)
			textp = append(textp, s)
		}
	}

	// Read relocations
	for _, sect := range ldSections {
		// TODO(aix): Dwarf section relocation if needed
		if sect.Type != xcoff.STYP_TEXT && sect.Type != xcoff.STYP_DATA {
			continue
		}
		sb := l.MakeSymbolUpdater(sect.sym)
		for _, rx := range sect.Relocs {
			rSym := l.LookupOrCreateCgoExport(rx.Symbol.Name, 0)
			if uint64(int32(rx.VirtualAddress)) != rx.VirtualAddress {
				return errorf("virtual address of a relocation is too big: 0x%x", rx.VirtualAddress)
			}
			rOff := int32(rx.VirtualAddress)
			var rSize uint8
			var rType objabi.RelocType
			var rAdd int64
			switch rx.Type {
			default:
				return errorf("section %s: unknown relocation of type 0x%x", sect.Name, rx.Type)
			case xcoff.R_POS:
				// Reloc the address of r.Sym
				// Length should be 64
				if rx.Length != 64 {
					return errorf("section %s: relocation R_POS has length different from 64: %d", sect.Name, rx.Length)
				}
				rSize = 8
				rType = objabi.R_CONST
				rAdd = int64(rx.Symbol.Value)

			case xcoff.R_RBR:
				rSize = 4
				rType = objabi.R_CALLPOWER
				rAdd = 0
			}
			r, _ := sb.AddRel(rType)
			r.SetOff(rOff)
			r.SetSiz(rSize)
			r.SetSym(rSym)
			r.SetAdd(rAdd)
		}
	}
	return textp, nil
}

// Convert symbol xcoff type to sym.SymKind
// Returns nil if this shouldn't be added into loader (like .file or .dw symbols )
func getSymbolType(f *xcoff.File, s *xcoff.Symbol) (stype sym.SymKind, err string) {
	// .file symbol
	if s.SectionNumber == -2 {
		if s.StorageClass == xcoff.C_FILE {
			return sym.Sxxx, ""
		}
		return sym.Sxxx, "unrecognised StorageClass for sectionNumber = -2"
	}

	// extern symbols
	// TODO(aix)
	if s.SectionNumber == 0 {
		return sym.Sxxx, ""
	}

	sectType := f.Sections[s.SectionNumber-1].SectionHeader.Type
	switch sectType {
	default:
		return sym.Sxxx, fmt.Sprintf("getSymbolType for Section type 0x%x not implemented", sectType)
	case xcoff.STYP_DWARF, xcoff.STYP_DEBUG:
		return sym.Sxxx, ""
	case xcoff.STYP_DATA, xcoff.STYP_BSS, xcoff.STYP_TEXT:
	}

	switch s.StorageClass {
	default:
		return sym.Sxxx, fmt.Sprintf("getSymbolType for Storage class 0x%x not implemented", s.StorageClass)
	case xcoff.C_HIDEXT, xcoff.C_EXT, xcoff.C_WEAKEXT:
		switch s.AuxCSect.StorageMappingClass {
		default:
			return sym.Sxxx, fmt.Sprintf("getSymbolType for Storage class 0x%x and Storage Map 0x%x not implemented", s.StorageClass, s.AuxCSect.StorageMappingClass)

		// Program Code
		case xcoff.XMC_PR:
			if sectType == xcoff.STYP_TEXT {
				return sym.STEXT, ""
			}
			return sym.Sxxx, fmt.Sprintf("unrecognised Section Type 0x%x for Storage Class 0x%x with Storage Map XMC_PR", sectType, s.StorageClass)

		// Read/Write Data
		case xcoff.XMC_RW:
			if sectType == xcoff.STYP_DATA {
				return sym.SDATA, ""
			}
			if sectType == xcoff.STYP_BSS {
				return sym.SBSS, ""
			}
			return sym.Sxxx, fmt.Sprintf("unrecognised Section Type 0x%x for Storage Class 0x%x with Storage Map XMC_RW", sectType, s.StorageClass)

		// Function descriptor
		case xcoff.XMC_DS:
			if sectType == xcoff.STYP_DATA {
				return sym.SDATA, ""
			}
			return sym.Sxxx, fmt.Sprintf("unrecognised Section Type 0x%x for Storage Class 0x%x with Storage Map XMC_DS", sectType, s.StorageClass)

		// TOC anchor and TOC entry
		case xcoff.XMC_TC0, xcoff.XMC_TE:
			if sectType == xcoff.STYP_DATA {
				return sym.SXCOFFTOC, ""
			}
			return sym.Sxxx, fmt.Sprintf("unrecognised Section Type 0x%x for Storage Class 0x%x with Storage Map XMC_DS", sectType, s.StorageClass)

		}
	}
}
