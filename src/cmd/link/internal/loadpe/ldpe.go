// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loadpe implements a PE/COFF file reader.
package loadpe

import (
	"cmd/internal/bio"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"debug/pe"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

const (
	// TODO: the Microsoft doco says IMAGE_SYM_DTYPE_ARRAY is 3 (same with IMAGE_SYM_DTYPE_POINTER and IMAGE_SYM_DTYPE_FUNCTION)
	IMAGE_SYM_UNDEFINED              = 0
	IMAGE_SYM_ABSOLUTE               = -1
	IMAGE_SYM_DEBUG                  = -2
	IMAGE_SYM_TYPE_NULL              = 0
	IMAGE_SYM_TYPE_VOID              = 1
	IMAGE_SYM_TYPE_CHAR              = 2
	IMAGE_SYM_TYPE_SHORT             = 3
	IMAGE_SYM_TYPE_INT               = 4
	IMAGE_SYM_TYPE_LONG              = 5
	IMAGE_SYM_TYPE_FLOAT             = 6
	IMAGE_SYM_TYPE_DOUBLE            = 7
	IMAGE_SYM_TYPE_STRUCT            = 8
	IMAGE_SYM_TYPE_UNION             = 9
	IMAGE_SYM_TYPE_ENUM              = 10
	IMAGE_SYM_TYPE_MOE               = 11
	IMAGE_SYM_TYPE_BYTE              = 12
	IMAGE_SYM_TYPE_WORD              = 13
	IMAGE_SYM_TYPE_UINT              = 14
	IMAGE_SYM_TYPE_DWORD             = 15
	IMAGE_SYM_TYPE_PCODE             = 32768
	IMAGE_SYM_DTYPE_NULL             = 0
	IMAGE_SYM_DTYPE_POINTER          = 0x10
	IMAGE_SYM_DTYPE_FUNCTION         = 0x20
	IMAGE_SYM_DTYPE_ARRAY            = 0x30
	IMAGE_SYM_CLASS_END_OF_FUNCTION  = -1
	IMAGE_SYM_CLASS_NULL             = 0
	IMAGE_SYM_CLASS_AUTOMATIC        = 1
	IMAGE_SYM_CLASS_EXTERNAL         = 2
	IMAGE_SYM_CLASS_STATIC           = 3
	IMAGE_SYM_CLASS_REGISTER         = 4
	IMAGE_SYM_CLASS_EXTERNAL_DEF     = 5
	IMAGE_SYM_CLASS_LABEL            = 6
	IMAGE_SYM_CLASS_UNDEFINED_LABEL  = 7
	IMAGE_SYM_CLASS_MEMBER_OF_STRUCT = 8
	IMAGE_SYM_CLASS_ARGUMENT         = 9
	IMAGE_SYM_CLASS_STRUCT_TAG       = 10
	IMAGE_SYM_CLASS_MEMBER_OF_UNION  = 11
	IMAGE_SYM_CLASS_UNION_TAG        = 12
	IMAGE_SYM_CLASS_TYPE_DEFINITION  = 13
	IMAGE_SYM_CLASS_UNDEFINED_STATIC = 14
	IMAGE_SYM_CLASS_ENUM_TAG         = 15
	IMAGE_SYM_CLASS_MEMBER_OF_ENUM   = 16
	IMAGE_SYM_CLASS_REGISTER_PARAM   = 17
	IMAGE_SYM_CLASS_BIT_FIELD        = 18
	IMAGE_SYM_CLASS_FAR_EXTERNAL     = 68 /* Not in PECOFF v8 spec */
	IMAGE_SYM_CLASS_BLOCK            = 100
	IMAGE_SYM_CLASS_FUNCTION         = 101
	IMAGE_SYM_CLASS_END_OF_STRUCT    = 102
	IMAGE_SYM_CLASS_FILE             = 103
	IMAGE_SYM_CLASS_SECTION          = 104
	IMAGE_SYM_CLASS_WEAK_EXTERNAL    = 105
	IMAGE_SYM_CLASS_CLR_TOKEN        = 107
	IMAGE_REL_I386_ABSOLUTE          = 0x0000
	IMAGE_REL_I386_DIR16             = 0x0001
	IMAGE_REL_I386_REL16             = 0x0002
	IMAGE_REL_I386_DIR32             = 0x0006
	IMAGE_REL_I386_DIR32NB           = 0x0007
	IMAGE_REL_I386_SEG12             = 0x0009
	IMAGE_REL_I386_SECTION           = 0x000A
	IMAGE_REL_I386_SECREL            = 0x000B
	IMAGE_REL_I386_TOKEN             = 0x000C
	IMAGE_REL_I386_SECREL7           = 0x000D
	IMAGE_REL_I386_REL32             = 0x0014
	IMAGE_REL_AMD64_ABSOLUTE         = 0x0000
	IMAGE_REL_AMD64_ADDR64           = 0x0001
	IMAGE_REL_AMD64_ADDR32           = 0x0002
	IMAGE_REL_AMD64_ADDR32NB         = 0x0003
	IMAGE_REL_AMD64_REL32            = 0x0004
	IMAGE_REL_AMD64_REL32_1          = 0x0005
	IMAGE_REL_AMD64_REL32_2          = 0x0006
	IMAGE_REL_AMD64_REL32_3          = 0x0007
	IMAGE_REL_AMD64_REL32_4          = 0x0008
	IMAGE_REL_AMD64_REL32_5          = 0x0009
	IMAGE_REL_AMD64_SECTION          = 0x000A
	IMAGE_REL_AMD64_SECREL           = 0x000B
	IMAGE_REL_AMD64_SECREL7          = 0x000C
	IMAGE_REL_AMD64_TOKEN            = 0x000D
	IMAGE_REL_AMD64_SREL32           = 0x000E
	IMAGE_REL_AMD64_PAIR             = 0x000F
	IMAGE_REL_AMD64_SSPAN32          = 0x0010
	IMAGE_REL_ARM_ABSOLUTE           = 0x0000
	IMAGE_REL_ARM_ADDR32             = 0x0001
	IMAGE_REL_ARM_ADDR32NB           = 0x0002
	IMAGE_REL_ARM_BRANCH24           = 0x0003
	IMAGE_REL_ARM_BRANCH11           = 0x0004
	IMAGE_REL_ARM_SECTION            = 0x000E
	IMAGE_REL_ARM_SECREL             = 0x000F
	IMAGE_REL_ARM_MOV32              = 0x0010
	IMAGE_REL_THUMB_MOV32            = 0x0011
	IMAGE_REL_THUMB_BRANCH20         = 0x0012
	IMAGE_REL_THUMB_BRANCH24         = 0x0014
	IMAGE_REL_THUMB_BLX23            = 0x0015
	IMAGE_REL_ARM_PAIR               = 0x0016
)

// TODO(crawshaw): de-duplicate these symbols with cmd/internal/ld, ideally in debug/pe.
const (
	IMAGE_SCN_CNT_CODE               = 0x00000020
	IMAGE_SCN_CNT_INITIALIZED_DATA   = 0x00000040
	IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080
	IMAGE_SCN_MEM_DISCARDABLE        = 0x02000000
	IMAGE_SCN_MEM_EXECUTE            = 0x20000000
	IMAGE_SCN_MEM_READ               = 0x40000000
	IMAGE_SCN_MEM_WRITE              = 0x80000000
)

// TODO(brainman): maybe just add ReadAt method to bio.Reader instead of creating peBiobuf

// peBiobuf makes bio.Reader look like io.ReaderAt.
type peBiobuf bio.Reader

func (f *peBiobuf) ReadAt(p []byte, off int64) (int, error) {
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

// Load loads the PE file pn from input.
// Symbols are written into syms, and a slice of the text symbols is returned.
// If an .rsrc section is found, its symbol is returned as rsrc.
func Load(arch *sys.Arch, syms *sym.Symbols, input *bio.Reader, pkg string, length int64, pn string) (textp []*sym.Symbol, rsrc *sym.Symbol, err error) {
	localSymVersion := syms.IncVersion()

	sectsyms := make(map[*pe.Section]*sym.Symbol)
	sectdata := make(map[*pe.Section][]byte)

	// Some input files are archives containing multiple of
	// object files, and pe.NewFile seeks to the start of
	// input file and get confused. Create section reader
	// to stop pe.NewFile looking before current position.
	sr := io.NewSectionReader((*peBiobuf)(input), input.Offset(), 1<<63-1)

	// TODO: replace pe.NewFile with pe.Load (grep for "add Load function" in debug/pe for details)
	f, err := pe.NewFile(sr)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	// TODO return error if found .cormeta

	// create symbols for mapped sections
	for _, sect := range f.Sections {
		if sect.Characteristics&IMAGE_SCN_MEM_DISCARDABLE != 0 {
			continue
		}

		if sect.Characteristics&(IMAGE_SCN_CNT_CODE|IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0 {
			// This has been seen for .idata sections, which we
			// want to ignore. See issues 5106 and 5273.
			continue
		}

		name := fmt.Sprintf("%s(%s)", pkg, sect.Name)
		s := syms.Lookup(name, localSymVersion)

		switch sect.Characteristics & (IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE) {
		case IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ: //.rdata
			s.Type = sym.SRODATA

		case IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE: //.bss
			s.Type = sym.SNOPTRBSS

		case IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE: //.data
			s.Type = sym.SNOPTRDATA

		case IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ: //.text
			s.Type = sym.STEXT

		default:
			return nil, nil, fmt.Errorf("unexpected flags %#06x for PE section %s", sect.Characteristics, sect.Name)
		}

		if s.Type != sym.SNOPTRBSS {
			data, err := sect.Data()
			if err != nil {
				return nil, nil, err
			}
			sectdata[sect] = data
			s.P = data
		}
		s.Size = int64(sect.Size)
		sectsyms[sect] = s
		if sect.Name == ".rsrc" {
			rsrc = s
		}
	}

	// load relocations
	for _, rsect := range f.Sections {
		if _, found := sectsyms[rsect]; !found {
			continue
		}
		if rsect.NumberOfRelocations == 0 {
			continue
		}
		if rsect.Characteristics&IMAGE_SCN_MEM_DISCARDABLE != 0 {
			continue
		}
		if rsect.Characteristics&(IMAGE_SCN_CNT_CODE|IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0 {
			// This has been seen for .idata sections, which we
			// want to ignore. See issues 5106 and 5273.
			continue
		}

		rs := make([]sym.Reloc, rsect.NumberOfRelocations)
		for j, r := range rsect.Relocs {
			rp := &rs[j]
			if int(r.SymbolTableIndex) >= len(f.COFFSymbols) {
				return nil, nil, fmt.Errorf("relocation number %d symbol index idx=%d cannot be large then number of symbols %d", j, r.SymbolTableIndex, len(f.COFFSymbols))
			}
			pesym := &f.COFFSymbols[r.SymbolTableIndex]
			gosym, err := readpesym(arch, syms, f, pesym, sectsyms, localSymVersion)
			if err != nil {
				return nil, nil, err
			}
			if gosym == nil {
				name, err := pesym.FullName(f.StringTable)
				if err != nil {
					name = string(pesym.Name[:])
				}
				return nil, nil, fmt.Errorf("reloc of invalid sym %s idx=%d type=%d", name, r.SymbolTableIndex, pesym.Type)
			}

			rp.Sym = gosym
			rp.Siz = 4
			rp.Off = int32(r.VirtualAddress)
			switch arch.Family {
			default:
				return nil, nil, fmt.Errorf("%s: unsupported arch %v", pn, arch.Family)
			case sys.I386, sys.AMD64:
				switch r.Type {
				default:
					return nil, nil, fmt.Errorf("%s: %v: unknown relocation type %v", pn, sectsyms[rsect], r.Type)

				case IMAGE_REL_I386_REL32, IMAGE_REL_AMD64_REL32,
					IMAGE_REL_AMD64_ADDR32, // R_X86_64_PC32
					IMAGE_REL_AMD64_ADDR32NB:
					rp.Type = objabi.R_PCREL

					rp.Add = int64(int32(binary.LittleEndian.Uint32(sectdata[rsect][rp.Off:])))

				case IMAGE_REL_I386_DIR32NB, IMAGE_REL_I386_DIR32:
					rp.Type = objabi.R_ADDR

					// load addend from image
					rp.Add = int64(int32(binary.LittleEndian.Uint32(sectdata[rsect][rp.Off:])))

				case IMAGE_REL_AMD64_ADDR64: // R_X86_64_64
					rp.Siz = 8

					rp.Type = objabi.R_ADDR

					// load addend from image
					rp.Add = int64(binary.LittleEndian.Uint64(sectdata[rsect][rp.Off:]))
				}

			case sys.ARM:
				switch r.Type {
				default:
					return nil, nil, fmt.Errorf("%s: %v: unknown ARM relocation type %v", pn, sectsyms[rsect], r.Type)

				case IMAGE_REL_ARM_SECREL:
					rp.Type = objabi.R_PCREL

					rp.Add = int64(int32(binary.LittleEndian.Uint32(sectdata[rsect][rp.Off:])))

				case IMAGE_REL_ARM_ADDR32:
					rp.Type = objabi.R_ADDR

					rp.Add = int64(int32(binary.LittleEndian.Uint32(sectdata[rsect][rp.Off:])))

				case IMAGE_REL_ARM_BRANCH24:
					rp.Type = objabi.R_CALLARM

					rp.Add = int64(int32(binary.LittleEndian.Uint32(sectdata[rsect][rp.Off:])))
				}
			}

			// ld -r could generate multiple section symbols for the
			// same section but with different values, we have to take
			// that into account
			if issect(pesym) {
				rp.Add += int64(pesym.Value)
			}
		}

		sort.Sort(sym.RelocByOff(rs[:rsect.NumberOfRelocations]))

		s := sectsyms[rsect]
		s.R = rs
		s.R = s.R[:rsect.NumberOfRelocations]
	}

	// enter sub-symbols into symbol table.
	for i, numaux := 0, 0; i < len(f.COFFSymbols); i += numaux + 1 {
		pesym := &f.COFFSymbols[i]

		numaux = int(pesym.NumberOfAuxSymbols)

		name, err := pesym.FullName(f.StringTable)
		if err != nil {
			return nil, nil, err
		}
		if name == "" {
			continue
		}
		if issect(pesym) {
			continue
		}
		if int(pesym.SectionNumber) > len(f.Sections) {
			continue
		}
		if pesym.SectionNumber == IMAGE_SYM_DEBUG {
			continue
		}
		var sect *pe.Section
		if pesym.SectionNumber > 0 {
			sect = f.Sections[pesym.SectionNumber-1]
			if _, found := sectsyms[sect]; !found {
				continue
			}
		}

		s, err := readpesym(arch, syms, f, pesym, sectsyms, localSymVersion)
		if err != nil {
			return nil, nil, err
		}

		if pesym.SectionNumber == 0 { // extern
			if s.Type == sym.SDYNIMPORT {
				s.SetPlt(-2) // flag for dynimport in PE object files.
			}
			if s.Type == sym.SXREF && pesym.Value > 0 { // global data
				s.Type = sym.SNOPTRDATA
				s.Size = int64(pesym.Value)
			}

			continue
		} else if pesym.SectionNumber > 0 && int(pesym.SectionNumber) <= len(f.Sections) {
			sect = f.Sections[pesym.SectionNumber-1]
			if _, found := sectsyms[sect]; !found {
				return nil, nil, fmt.Errorf("%s: %v: missing sect.sym", pn, s)
			}
		} else {
			return nil, nil, fmt.Errorf("%s: %v: sectnum < 0!", pn, s)
		}

		if sect == nil {
			return nil, rsrc, nil
		}

		if s.Outer != nil {
			if s.Attr.DuplicateOK() {
				continue
			}
			return nil, nil, fmt.Errorf("%s: duplicate symbol reference: %s in both %s and %s", pn, s.Name, s.Outer.Name, sectsyms[sect].Name)
		}

		sectsym := sectsyms[sect]
		s.Sub = sectsym.Sub
		sectsym.Sub = s
		s.Type = sectsym.Type
		s.Attr |= sym.AttrSubSymbol
		s.Value = int64(pesym.Value)
		s.Size = 4
		s.Outer = sectsym
		if sectsym.Type == sym.STEXT {
			if s.Attr.External() && !s.Attr.DuplicateOK() {
				return nil, nil, fmt.Errorf("%s: duplicate symbol definition", s.Name)
			}
			s.Attr |= sym.AttrExternal
		}
	}

	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for _, sect := range f.Sections {
		s := sectsyms[sect]
		if s == nil {
			continue
		}
		if s.Sub != nil {
			s.Sub = sym.SortSub(s.Sub)
		}
		if s.Type == sym.STEXT {
			if s.Attr.OnList() {
				return nil, nil, fmt.Errorf("symbol %s listed multiple times", s.Name)
			}
			s.Attr |= sym.AttrOnList
			textp = append(textp, s)
			for s = s.Sub; s != nil; s = s.Sub {
				if s.Attr.OnList() {
					return nil, nil, fmt.Errorf("symbol %s listed multiple times", s.Name)
				}
				s.Attr |= sym.AttrOnList
				textp = append(textp, s)
			}
		}
	}

	return textp, rsrc, nil
}

func issect(s *pe.COFFSymbol) bool {
	return s.StorageClass == IMAGE_SYM_CLASS_STATIC && s.Type == 0 && s.Name[0] == '.'
}

func readpesym(arch *sys.Arch, syms *sym.Symbols, f *pe.File, pesym *pe.COFFSymbol, sectsyms map[*pe.Section]*sym.Symbol, localSymVersion int) (*sym.Symbol, error) {
	symname, err := pesym.FullName(f.StringTable)
	if err != nil {
		return nil, err
	}
	var name string
	if issect(pesym) {
		name = sectsyms[f.Sections[pesym.SectionNumber-1]].Name
	} else {
		name = symname
		name = strings.TrimPrefix(name, "__imp_") // __imp_Name => Name
		if arch.Family == sys.I386 && name[0] == '_' {
			name = name[1:] // _Name => Name
		}
	}

	// remove last @XXX
	if i := strings.LastIndex(name, "@"); i >= 0 {
		name = name[:i]
	}

	var s *sym.Symbol
	switch pesym.Type {
	default:
		return nil, fmt.Errorf("%s: invalid symbol type %d", symname, pesym.Type)

	case IMAGE_SYM_DTYPE_FUNCTION, IMAGE_SYM_DTYPE_NULL:
		switch pesym.StorageClass {
		case IMAGE_SYM_CLASS_EXTERNAL: //global
			s = syms.Lookup(name, 0)

		case IMAGE_SYM_CLASS_NULL, IMAGE_SYM_CLASS_STATIC, IMAGE_SYM_CLASS_LABEL:
			s = syms.Lookup(name, localSymVersion)
			s.Attr |= sym.AttrDuplicateOK

		default:
			return nil, fmt.Errorf("%s: invalid symbol binding %d", symname, pesym.StorageClass)
		}
	}

	if s != nil && s.Type == 0 && (pesym.StorageClass != IMAGE_SYM_CLASS_STATIC || pesym.Value != 0) {
		s.Type = sym.SXREF
	}
	if strings.HasPrefix(symname, "__imp_") {
		s.SetGot(-2) // flag for __imp_
	}

	return s, nil
}
