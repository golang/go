// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/bio"
	"cmd/internal/obj"
	"cmd/internal/sys"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"sort"
	"strconv"
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

type PeSym struct {
	name    string
	value   uint32
	sectnum uint16
	type_   uint16
	sclass  uint8
	aux     uint8
	sym     *LSym
}

type PeSect struct {
	name string
	base []byte
	size uint64
	sym  *LSym
	sh   IMAGE_SECTION_HEADER
}

type PeObj struct {
	f      *bio.Reader
	name   string
	base   uint32
	sect   []PeSect
	nsect  uint
	pesym  []PeSym
	npesym uint
	fh     IMAGE_FILE_HEADER
	snames []byte
}

func ldpe(f *bio.Reader, pkg string, length int64, pn string) {
	if Debug['v'] != 0 {
		fmt.Fprintf(Bso, "%5.2f ldpe %s\n", obj.Cputime(), pn)
	}

	var sect *PeSect
	Ctxt.IncVersion()
	base := f.Offset()

	peobj := new(PeObj)
	peobj.f = f
	peobj.base = uint32(base)
	peobj.name = pn

	// read header
	var err error
	var j int
	var l uint32
	var name string
	var numaux int
	var r []Reloc
	var rp *Reloc
	var rsect *PeSect
	var s *LSym
	var sym *PeSym
	var symbuf [18]uint8
	if err = binary.Read(f, binary.LittleEndian, &peobj.fh); err != nil {
		goto bad
	}

	// load section list
	peobj.sect = make([]PeSect, peobj.fh.NumberOfSections)

	peobj.nsect = uint(peobj.fh.NumberOfSections)
	for i := 0; i < int(peobj.fh.NumberOfSections); i++ {
		if err = binary.Read(f, binary.LittleEndian, &peobj.sect[i].sh); err != nil {
			goto bad
		}
		peobj.sect[i].size = uint64(peobj.sect[i].sh.SizeOfRawData)
		peobj.sect[i].name = cstring(peobj.sect[i].sh.Name[:])
	}

	// TODO return error if found .cormeta

	// load string table
	f.Seek(base+int64(peobj.fh.PointerToSymbolTable)+int64(len(symbuf))*int64(peobj.fh.NumberOfSymbols), 0)

	if _, err := io.ReadFull(f, symbuf[:4]); err != nil {
		goto bad
	}
	l = Le32(symbuf[:])
	peobj.snames = make([]byte, l)
	f.Seek(base+int64(peobj.fh.PointerToSymbolTable)+int64(len(symbuf))*int64(peobj.fh.NumberOfSymbols), 0)
	if _, err := io.ReadFull(f, peobj.snames); err != nil {
		goto bad
	}

	// rewrite section names if they start with /
	for i := 0; i < int(peobj.fh.NumberOfSections); i++ {
		if peobj.sect[i].name == "" {
			continue
		}
		if peobj.sect[i].name[0] != '/' {
			continue
		}
		n, _ := strconv.Atoi(peobj.sect[i].name[1:])
		peobj.sect[i].name = cstring(peobj.snames[n:])
	}

	// read symbols
	peobj.pesym = make([]PeSym, peobj.fh.NumberOfSymbols)

	peobj.npesym = uint(peobj.fh.NumberOfSymbols)
	f.Seek(base+int64(peobj.fh.PointerToSymbolTable), 0)
	for i := 0; uint32(i) < peobj.fh.NumberOfSymbols; i += numaux + 1 {
		f.Seek(base+int64(peobj.fh.PointerToSymbolTable)+int64(len(symbuf))*int64(i), 0)
		if _, err := io.ReadFull(f, symbuf[:]); err != nil {
			goto bad
		}

		if (symbuf[0] == 0) && (symbuf[1] == 0) && (symbuf[2] == 0) && (symbuf[3] == 0) {
			l = Le32(symbuf[4:])
			peobj.pesym[i].name = cstring(peobj.snames[l:]) // sym name length <= 8
		} else {
			peobj.pesym[i].name = cstring(symbuf[:8])
		}

		peobj.pesym[i].value = Le32(symbuf[8:])
		peobj.pesym[i].sectnum = Le16(symbuf[12:])
		peobj.pesym[i].sclass = symbuf[16]
		peobj.pesym[i].aux = symbuf[17]
		peobj.pesym[i].type_ = Le16(symbuf[14:])
		numaux = int(peobj.pesym[i].aux)
		if numaux < 0 {
			numaux = 0
		}
	}

	// create symbols for mapped sections
	for i := 0; uint(i) < peobj.nsect; i++ {
		sect = &peobj.sect[i]
		if sect.sh.Characteristics&IMAGE_SCN_MEM_DISCARDABLE != 0 {
			continue
		}

		if sect.sh.Characteristics&(IMAGE_SCN_CNT_CODE|IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0 {
			// This has been seen for .idata sections, which we
			// want to ignore. See issues 5106 and 5273.
			continue
		}

		if pemap(peobj, sect) < 0 {
			goto bad
		}

		name = fmt.Sprintf("%s(%s)", pkg, sect.name)
		s = Linklookup(Ctxt, name, Ctxt.Version)

		switch sect.sh.Characteristics & (IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE) {
		case IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ: //.rdata
			s.Type = obj.SRODATA

		case IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE: //.bss
			s.Type = obj.SNOPTRBSS

		case IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE: //.data
			s.Type = obj.SNOPTRDATA

		case IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ: //.text
			s.Type = obj.STEXT

		default:
			err = fmt.Errorf("unexpected flags %#06x for PE section %s", sect.sh.Characteristics, sect.name)
			goto bad
		}

		s.P = sect.base
		s.P = s.P[:sect.size]
		s.Size = int64(sect.size)
		sect.sym = s
		if sect.name == ".rsrc" {
			setpersrc(sect.sym)
		}
	}

	// load relocations
	for i := 0; uint(i) < peobj.nsect; i++ {
		rsect = &peobj.sect[i]
		if rsect.sym == nil || rsect.sh.NumberOfRelocations == 0 {
			continue
		}
		if rsect.sh.Characteristics&IMAGE_SCN_MEM_DISCARDABLE != 0 {
			continue
		}
		if sect.sh.Characteristics&(IMAGE_SCN_CNT_CODE|IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0 {
			// This has been seen for .idata sections, which we
			// want to ignore. See issues 5106 and 5273.
			continue
		}

		r = make([]Reloc, rsect.sh.NumberOfRelocations)
		f.Seek(int64(peobj.base)+int64(rsect.sh.PointerToRelocations), 0)
		for j = 0; j < int(rsect.sh.NumberOfRelocations); j++ {
			rp = &r[j]
			if _, err := io.ReadFull(f, symbuf[:10]); err != nil {
				goto bad
			}
			rva := Le32(symbuf[0:])
			symindex := Le32(symbuf[4:])
			type_ := Le16(symbuf[8:])
			if err = readpesym(peobj, int(symindex), &sym); err != nil {
				goto bad
			}
			if sym.sym == nil {
				err = fmt.Errorf("reloc of invalid sym %s idx=%d type=%d", sym.name, symindex, sym.type_)
				goto bad
			}

			rp.Sym = sym.sym
			rp.Siz = 4
			rp.Off = int32(rva)
			switch type_ {
			default:
				Diag("%s: unknown relocation type %d;", pn, type_)
				fallthrough

			case IMAGE_REL_I386_REL32, IMAGE_REL_AMD64_REL32,
				IMAGE_REL_AMD64_ADDR32, // R_X86_64_PC32
				IMAGE_REL_AMD64_ADDR32NB:
				rp.Type = obj.R_PCREL

				rp.Add = int64(int32(Le32(rsect.base[rp.Off:])))

			case IMAGE_REL_I386_DIR32NB, IMAGE_REL_I386_DIR32:
				rp.Type = obj.R_ADDR

				// load addend from image
				rp.Add = int64(int32(Le32(rsect.base[rp.Off:])))

			case IMAGE_REL_AMD64_ADDR64: // R_X86_64_64
				rp.Siz = 8

				rp.Type = obj.R_ADDR

				// load addend from image
				rp.Add = int64(Le64(rsect.base[rp.Off:]))
			}

			// ld -r could generate multiple section symbols for the
			// same section but with different values, we have to take
			// that into account
			if issect(&peobj.pesym[symindex]) {
				rp.Add += int64(peobj.pesym[symindex].value)
			}
		}

		sort.Sort(rbyoff(r[:rsect.sh.NumberOfRelocations]))

		s = rsect.sym
		s.R = r
		s.R = s.R[:rsect.sh.NumberOfRelocations]
	}

	// enter sub-symbols into symbol table.
	for i := 0; uint(i) < peobj.npesym; i++ {
		if peobj.pesym[i].name == "" {
			continue
		}
		if issect(&peobj.pesym[i]) {
			continue
		}
		if uint(peobj.pesym[i].sectnum) > peobj.nsect {
			continue
		}
		if peobj.pesym[i].sectnum > 0 {
			sect = &peobj.sect[peobj.pesym[i].sectnum-1]
			if sect.sym == nil {
				continue
			}
		}

		if err = readpesym(peobj, i, &sym); err != nil {
			goto bad
		}

		s = sym.sym
		if sym.sectnum == 0 { // extern
			if s.Type == obj.SDYNIMPORT {
				s.Plt = -2 // flag for dynimport in PE object files.
			}
			if s.Type == obj.SXREF && sym.value > 0 { // global data
				s.Type = obj.SNOPTRDATA
				s.Size = int64(sym.value)
			}

			continue
		} else if sym.sectnum > 0 && uint(sym.sectnum) <= peobj.nsect {
			sect = &peobj.sect[sym.sectnum-1]
			if sect.sym == nil {
				Diag("%s: %s sym == 0!", pn, s.Name)
			}
		} else {
			Diag("%s: %s sectnum < 0!", pn, s.Name)
		}

		if sect == nil {
			return
		}

		if s.Outer != nil {
			if s.Attr.DuplicateOK() {
				continue
			}
			Exitf("%s: duplicate symbol reference: %s in both %s and %s", pn, s.Name, s.Outer.Name, sect.sym.Name)
		}

		s.Sub = sect.sym.Sub
		sect.sym.Sub = s
		s.Type = sect.sym.Type | obj.SSUB
		s.Value = int64(sym.value)
		s.Size = 4
		s.Outer = sect.sym
		if sect.sym.Type == obj.STEXT {
			if s.Attr.External() && !s.Attr.DuplicateOK() {
				Diag("%s: duplicate definition of %s", pn, s.Name)
			}
			s.Attr |= AttrExternal
		}
	}

	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for i := 0; uint(i) < peobj.nsect; i++ {
		s = peobj.sect[i].sym
		if s == nil {
			continue
		}
		if s.Sub != nil {
			s.Sub = listsort(s.Sub, valuecmp, listsubp)
		}
		if s.Type == obj.STEXT {
			if s.Attr.OnList() {
				log.Fatalf("symbol %s listed multiple times", s.Name)
			}
			s.Attr |= AttrOnList
			Ctxt.Textp = append(Ctxt.Textp, s)
			for s = s.Sub; s != nil; s = s.Sub {
				if s.Attr.OnList() {
					log.Fatalf("symbol %s listed multiple times", s.Name)
				}
				s.Attr |= AttrOnList
				Ctxt.Textp = append(Ctxt.Textp, s)
			}
		}
	}

	return

bad:
	Diag("%s: malformed pe file: %v", pn, err)
}

func pemap(peobj *PeObj, sect *PeSect) int {
	if sect.base != nil {
		return 0
	}

	sect.base = make([]byte, sect.sh.SizeOfRawData)
	if sect.sh.PointerToRawData == 0 { // .bss doesn't have data in object file
		return 0
	}
	if peobj.f.Seek(int64(peobj.base)+int64(sect.sh.PointerToRawData), 0) < 0 {
		return -1
	}
	if _, err := io.ReadFull(peobj.f, sect.base); err != nil {
		return -1
	}

	return 0
}

func issect(s *PeSym) bool {
	return s.sclass == IMAGE_SYM_CLASS_STATIC && s.type_ == 0 && s.name[0] == '.'
}

func readpesym(peobj *PeObj, i int, y **PeSym) (err error) {
	if uint(i) >= peobj.npesym || i < 0 {
		err = fmt.Errorf("invalid pe symbol index")
		return err
	}

	sym := &peobj.pesym[i]
	*y = sym

	var name string
	if issect(sym) {
		name = peobj.sect[sym.sectnum-1].sym.Name
	} else {
		name = sym.name
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

	var s *LSym
	switch sym.type_ {
	default:
		err = fmt.Errorf("%s: invalid symbol type %d", sym.name, sym.type_)
		return err

	case IMAGE_SYM_DTYPE_FUNCTION, IMAGE_SYM_DTYPE_NULL:
		switch sym.sclass {
		case IMAGE_SYM_CLASS_EXTERNAL: //global
			s = Linklookup(Ctxt, name, 0)

		case IMAGE_SYM_CLASS_NULL, IMAGE_SYM_CLASS_STATIC, IMAGE_SYM_CLASS_LABEL:
			s = Linklookup(Ctxt, name, Ctxt.Version)
			s.Attr |= AttrDuplicateOK

		default:
			err = fmt.Errorf("%s: invalid symbol binding %d", sym.name, sym.sclass)
			return err
		}
	}

	if s != nil && s.Type == 0 && (sym.sclass != IMAGE_SYM_CLASS_STATIC || sym.value != 0) {
		s.Type = obj.SXREF
	}
	if strings.HasPrefix(sym.name, "__imp_") {
		s.Got = -2 // flag for __imp_
	}
	sym.sym = s

	return nil
}
