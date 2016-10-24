// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/bio"
	"cmd/internal/obj"
	"cmd/internal/sys"
	"debug/pe"
	"errors"
	"fmt"
	"io"
	"log"
	"sort"
	"strings"
)

const (
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
)

// TODO(brainman): maybe just add ReadAt method to bio.Reader instead of creating peBiobuf

// peBiobuf makes bio.Reader look like io.ReaderAt.
type peBiobuf bio.Reader

func (f *peBiobuf) ReadAt(p []byte, off int64) (int, error) {
	ret := ((*bio.Reader)(f)).Seek(off, 0)
	if ret < 0 {
		return 0, errors.New("fail to seek")
	}
	n, err := f.Read(p)
	if err != nil {
		return 0, err
	}
	return n, nil
}

func ldpe(ctxt *Link, input *bio.Reader, pkg string, length int64, pn string) {
	err := ldpeError(ctxt, input, pkg, length, pn)
	if err != nil {
		Errorf(nil, "%s: malformed pe file: %v", pn, err)
	}
}

func ldpeError(ctxt *Link, input *bio.Reader, pkg string, length int64, pn string) error {
	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f ldpe %s\n", obj.Cputime(), pn)
	}

	localSymVersion := ctxt.Syms.IncVersion()

	sectsyms := make(map[*pe.Section]*Symbol)
	sectdata := make(map[*pe.Section][]byte)

	// Some input files are archives containing multiple of
	// object files, and pe.NewFile seeks to the start of
	// input file and get confused. Create section reader
	// to stop pe.NewFile looking before current position.
	sr := io.NewSectionReader((*peBiobuf)(input), input.Offset(), 1<<63-1)

	// TODO: replace pe.NewFile with pe.Load (grep for "add Load function" in debug/pe for details)
	f, err := pe.NewFile(sr)
	if err != nil {
		return err
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

		data, err := sect.Data()
		if err != nil {
			return err
		}
		sectdata[sect] = data

		name := fmt.Sprintf("%s(%s)", pkg, sect.Name)
		s := ctxt.Syms.Lookup(name, localSymVersion)

		switch sect.Characteristics & (IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE) {
		case IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ: //.rdata
			s.Type = obj.SRODATA

		case IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE: //.bss
			s.Type = obj.SNOPTRBSS

		case IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE: //.data
			s.Type = obj.SNOPTRDATA

		case IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ: //.text
			s.Type = obj.STEXT

		default:
			return fmt.Errorf("unexpected flags %#06x for PE section %s", sect.Characteristics, sect.Name)
		}

		s.P = data
		s.Size = int64(len(data))
		sectsyms[sect] = s
		if sect.Name == ".rsrc" {
			setpersrc(ctxt, s)
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

		rs := make([]Reloc, rsect.NumberOfRelocations)
		for j, r := range rsect.Relocs {
			rp := &rs[j]
			if int(r.SymbolTableIndex) >= len(f.COFFSymbols) {
				return fmt.Errorf("relocation number %d symbol index idx=%d cannot be large then number of symbols %d", j, r.SymbolTableIndex, len(f.COFFSymbols))
			}
			pesym := &f.COFFSymbols[r.SymbolTableIndex]
			gosym, err := readpesym(ctxt, f, pesym, sectsyms, localSymVersion)
			if err != nil {
				return err
			}
			if gosym == nil {
				name, err := pesym.FullName(f.StringTable)
				if err != nil {
					name = string(pesym.Name[:])
				}
				return fmt.Errorf("reloc of invalid sym %s idx=%d type=%d", name, r.SymbolTableIndex, pesym.Type)
			}

			rp.Sym = gosym
			rp.Siz = 4
			rp.Off = int32(r.VirtualAddress)
			switch r.Type {
			default:
				Errorf(sectsyms[rsect], "%s: unknown relocation type %d;", pn, r.Type)
				fallthrough

			case IMAGE_REL_I386_REL32, IMAGE_REL_AMD64_REL32,
				IMAGE_REL_AMD64_ADDR32, // R_X86_64_PC32
				IMAGE_REL_AMD64_ADDR32NB:
				rp.Type = obj.R_PCREL

				rp.Add = int64(int32(Le32(sectdata[rsect][rp.Off:])))

			case IMAGE_REL_I386_DIR32NB, IMAGE_REL_I386_DIR32:
				rp.Type = obj.R_ADDR

				// load addend from image
				rp.Add = int64(int32(Le32(sectdata[rsect][rp.Off:])))

			case IMAGE_REL_AMD64_ADDR64: // R_X86_64_64
				rp.Siz = 8

				rp.Type = obj.R_ADDR

				// load addend from image
				rp.Add = int64(Le64(sectdata[rsect][rp.Off:]))
			}

			// ld -r could generate multiple section symbols for the
			// same section but with different values, we have to take
			// that into account
			if issect(pesym) {
				rp.Add += int64(pesym.Value)
			}
		}

		sort.Sort(rbyoff(rs[:rsect.NumberOfRelocations]))

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
			return err
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

		s, err := readpesym(ctxt, f, pesym, sectsyms, localSymVersion)
		if err != nil {
			return err
		}

		if pesym.SectionNumber == 0 { // extern
			if s.Type == obj.SDYNIMPORT {
				s.Plt = -2 // flag for dynimport in PE object files.
			}
			if s.Type == obj.SXREF && pesym.Value > 0 { // global data
				s.Type = obj.SNOPTRDATA
				s.Size = int64(pesym.Value)
			}

			continue
		} else if pesym.SectionNumber > 0 && int(pesym.SectionNumber) <= len(f.Sections) {
			sect = f.Sections[pesym.SectionNumber-1]
			if _, found := sectsyms[sect]; !found {
				Errorf(s, "%s: missing sect.sym", pn)
			}
		} else {
			Errorf(s, "%s: sectnum < 0!", pn)
		}

		if sect == nil {
			return nil
		}

		if s.Outer != nil {
			if s.Attr.DuplicateOK() {
				continue
			}
			Exitf("%s: duplicate symbol reference: %s in both %s and %s", pn, s.Name, s.Outer.Name, sectsyms[sect].Name)
		}

		sectsym := sectsyms[sect]
		s.Sub = sectsym.Sub
		sectsym.Sub = s
		s.Type = sectsym.Type | obj.SSUB
		s.Value = int64(pesym.Value)
		s.Size = 4
		s.Outer = sectsym
		if sectsym.Type == obj.STEXT {
			if s.Attr.External() && !s.Attr.DuplicateOK() {
				Errorf(s, "%s: duplicate symbol definition", pn)
			}
			s.Attr |= AttrExternal
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
			s.Sub = listsort(s.Sub)
		}
		if s.Type == obj.STEXT {
			if s.Attr.OnList() {
				log.Fatalf("symbol %s listed multiple times", s.Name)
			}
			s.Attr |= AttrOnList
			ctxt.Textp = append(ctxt.Textp, s)
			for s = s.Sub; s != nil; s = s.Sub {
				if s.Attr.OnList() {
					log.Fatalf("symbol %s listed multiple times", s.Name)
				}
				s.Attr |= AttrOnList
				ctxt.Textp = append(ctxt.Textp, s)
			}
		}
	}

	return nil
}

func issect(s *pe.COFFSymbol) bool {
	return s.StorageClass == IMAGE_SYM_CLASS_STATIC && s.Type == 0 && s.Name[0] == '.'
}

func readpesym(ctxt *Link, f *pe.File, sym *pe.COFFSymbol, sectsyms map[*pe.Section]*Symbol, localSymVersion int) (*Symbol, error) {
	symname, err := sym.FullName(f.StringTable)
	if err != nil {
		return nil, err
	}
	var name string
	if issect(sym) {
		name = sectsyms[f.Sections[sym.SectionNumber-1]].Name
	} else {
		name = symname
		if strings.HasPrefix(name, "__imp_") {
			name = name[6:] // __imp_Name => Name
		}
		if SysArch.Family == sys.I386 && name[0] == '_' {
			name = name[1:] // _Name => Name
		}
	}

	// remove last @XXX
	if i := strings.LastIndex(name, "@"); i >= 0 {
		name = name[:i]
	}

	var s *Symbol
	switch sym.Type {
	default:
		return nil, fmt.Errorf("%s: invalid symbol type %d", symname, sym.Type)

	case IMAGE_SYM_DTYPE_FUNCTION, IMAGE_SYM_DTYPE_NULL:
		switch sym.StorageClass {
		case IMAGE_SYM_CLASS_EXTERNAL: //global
			s = ctxt.Syms.Lookup(name, 0)

		case IMAGE_SYM_CLASS_NULL, IMAGE_SYM_CLASS_STATIC, IMAGE_SYM_CLASS_LABEL:
			s = ctxt.Syms.Lookup(name, localSymVersion)
			s.Attr |= AttrDuplicateOK

		default:
			return nil, fmt.Errorf("%s: invalid symbol binding %d", symname, sym.StorageClass)
		}
	}

	if s != nil && s.Type == 0 && (sym.StorageClass != IMAGE_SYM_CLASS_STATIC || sym.Value != 0) {
		s.Type = obj.SXREF
	}
	if strings.HasPrefix(symname, "__imp_") {
		s.Got = -2 // flag for __imp_
	}

	return s, nil
}
