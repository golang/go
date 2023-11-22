// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loadpe implements a PE/COFF file reader.
package loadpe

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/pe"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
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
	IMAGE_SYM_DTYPE_POINTER          = 1
	IMAGE_SYM_DTYPE_FUNCTION         = 2
	IMAGE_SYM_DTYPE_ARRAY            = 3
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
	IMAGE_REL_ARM64_ABSOLUTE         = 0x0000
	IMAGE_REL_ARM64_ADDR32           = 0x0001
	IMAGE_REL_ARM64_ADDR32NB         = 0x0002
	IMAGE_REL_ARM64_BRANCH26         = 0x0003
	IMAGE_REL_ARM64_PAGEBASE_REL21   = 0x0004
	IMAGE_REL_ARM64_REL21            = 0x0005
	IMAGE_REL_ARM64_PAGEOFFSET_12A   = 0x0006
	IMAGE_REL_ARM64_PAGEOFFSET_12L   = 0x0007
	IMAGE_REL_ARM64_SECREL           = 0x0008
	IMAGE_REL_ARM64_SECREL_LOW12A    = 0x0009
	IMAGE_REL_ARM64_SECREL_HIGH12A   = 0x000A
	IMAGE_REL_ARM64_SECREL_LOW12L    = 0x000B
	IMAGE_REL_ARM64_TOKEN            = 0x000C
	IMAGE_REL_ARM64_SECTION          = 0x000D
	IMAGE_REL_ARM64_ADDR64           = 0x000E
	IMAGE_REL_ARM64_BRANCH19         = 0x000F
	IMAGE_REL_ARM64_BRANCH14         = 0x0010
	IMAGE_REL_ARM64_REL32            = 0x0011
)

const (
	// When stored into the PLT value for a symbol, this token tells
	// windynrelocsym to redirect direct references to this symbol to a stub
	// that loads from the corresponding import symbol and then does
	// a jump to the loaded value.
	CreateImportStubPltToken = -2

	// When stored into the GOT value for an import symbol __imp_X this
	// token tells windynrelocsym to redirect references to the
	// underlying DYNIMPORT symbol X.
	RedirectToDynImportGotToken = -2
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

// makeUpdater creates a loader.SymbolBuilder if one hasn't been created previously.
// We use this to lazily make SymbolBuilders as we don't always need a builder, and creating them for all symbols might be an error.
func makeUpdater(l *loader.Loader, bld *loader.SymbolBuilder, s loader.Sym) *loader.SymbolBuilder {
	if bld != nil {
		return bld
	}
	bld = l.MakeSymbolUpdater(s)
	return bld
}

// peImportSymsState tracks the set of DLL import symbols we've seen
// while reading host objects. We create a singleton instance of this
// type, which will persist across multiple host objects.
type peImportSymsState struct {

	// Text and non-text sections read in by the host object loader.
	secSyms []loader.Sym

	// Loader and arch, for use in postprocessing.
	l    *loader.Loader
	arch *sys.Arch
}

var importSymsState *peImportSymsState

func createImportSymsState(l *loader.Loader, arch *sys.Arch) {
	if importSymsState != nil {
		return
	}
	importSymsState = &peImportSymsState{
		l:    l,
		arch: arch,
	}
}

// peLoaderState holds various bits of useful state information needed
// while loading a single PE object file.
type peLoaderState struct {
	l               *loader.Loader
	arch            *sys.Arch
	f               *pe.File
	pn              string
	sectsyms        map[*pe.Section]loader.Sym
	comdats         map[uint16]int64 // key is section index, val is size
	sectdata        map[*pe.Section][]byte
	localSymVersion int
}

// comdatDefinitions records the names of symbols for which we've
// previously seen a definition in COMDAT. Key is symbol name, value
// is symbol size (or -1 if we're using the "any" strategy).
var comdatDefinitions map[string]int64

// Symbols contains the symbols that can be loaded from a PE file.
type Symbols struct {
	Textp     []loader.Sym // text symbols
	Resources []loader.Sym // .rsrc section or set of .rsrc$xx sections
	PData     loader.Sym
	XData     loader.Sym
}

// Load loads the PE file pn from input.
// Symbols from the object file are created via the loader 'l'.
func Load(l *loader.Loader, arch *sys.Arch, localSymVersion int, input *bio.Reader, pkg string, length int64, pn string) (*Symbols, error) {
	state := &peLoaderState{
		l:               l,
		arch:            arch,
		sectsyms:        make(map[*pe.Section]loader.Sym),
		sectdata:        make(map[*pe.Section][]byte),
		localSymVersion: localSymVersion,
		pn:              pn,
	}
	createImportSymsState(state.l, state.arch)
	if comdatDefinitions == nil {
		comdatDefinitions = make(map[string]int64)
	}

	// Some input files are archives containing multiple of
	// object files, and pe.NewFile seeks to the start of
	// input file and get confused. Create section reader
	// to stop pe.NewFile looking before current position.
	sr := io.NewSectionReader((*peBiobuf)(input), input.Offset(), 1<<63-1)

	// TODO: replace pe.NewFile with pe.Load (grep for "add Load function" in debug/pe for details)
	f, err := pe.NewFile(sr)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	state.f = f

	var ls Symbols

	// TODO return error if found .cormeta

	// create symbols for mapped sections
	for _, sect := range f.Sections {
		if sect.Characteristics&pe.IMAGE_SCN_MEM_DISCARDABLE != 0 {
			continue
		}

		if sect.Characteristics&(pe.IMAGE_SCN_CNT_CODE|pe.IMAGE_SCN_CNT_INITIALIZED_DATA|pe.IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0 {
			// This has been seen for .idata sections, which we
			// want to ignore. See issues 5106 and 5273.
			continue
		}

		name := fmt.Sprintf("%s(%s)", pkg, sect.Name)
		s := state.l.LookupOrCreateCgoExport(name, localSymVersion)
		bld := l.MakeSymbolUpdater(s)

		switch sect.Characteristics & (pe.IMAGE_SCN_CNT_UNINITIALIZED_DATA | pe.IMAGE_SCN_CNT_INITIALIZED_DATA | pe.IMAGE_SCN_MEM_READ | pe.IMAGE_SCN_MEM_WRITE | pe.IMAGE_SCN_CNT_CODE | pe.IMAGE_SCN_MEM_EXECUTE) {
		case pe.IMAGE_SCN_CNT_INITIALIZED_DATA | pe.IMAGE_SCN_MEM_READ: //.rdata
			if issehsect(arch, sect) {
				bld.SetType(sym.SSEHSECT)
				bld.SetAlign(4)
			} else {
				bld.SetType(sym.SRODATA)
			}

		case pe.IMAGE_SCN_CNT_UNINITIALIZED_DATA | pe.IMAGE_SCN_MEM_READ | pe.IMAGE_SCN_MEM_WRITE: //.bss
			bld.SetType(sym.SNOPTRBSS)

		case pe.IMAGE_SCN_CNT_INITIALIZED_DATA | pe.IMAGE_SCN_MEM_READ | pe.IMAGE_SCN_MEM_WRITE: //.data
			bld.SetType(sym.SNOPTRDATA)

		case pe.IMAGE_SCN_CNT_CODE | pe.IMAGE_SCN_MEM_EXECUTE | pe.IMAGE_SCN_MEM_READ: //.text
			bld.SetType(sym.STEXT)

		default:
			return nil, fmt.Errorf("unexpected flags %#06x for PE section %s", sect.Characteristics, sect.Name)
		}

		if bld.Type() != sym.SNOPTRBSS {
			data, err := sect.Data()
			if err != nil {
				return nil, err
			}
			state.sectdata[sect] = data
			bld.SetData(data)
		}
		bld.SetSize(int64(sect.Size))
		state.sectsyms[sect] = s
		if sect.Name == ".rsrc" || strings.HasPrefix(sect.Name, ".rsrc$") {
			ls.Resources = append(ls.Resources, s)
		} else if bld.Type() == sym.SSEHSECT {
			if sect.Name == ".pdata" {
				ls.PData = s
			} else if sect.Name == ".xdata" {
				ls.XData = s
			}
		}
	}

	// Make a prepass over the symbols to collect info about COMDAT symbols.
	if err := state.preprocessSymbols(); err != nil {
		return nil, err
	}

	// load relocations
	for _, rsect := range f.Sections {
		if _, found := state.sectsyms[rsect]; !found {
			continue
		}
		if rsect.NumberOfRelocations == 0 {
			continue
		}
		if rsect.Characteristics&pe.IMAGE_SCN_MEM_DISCARDABLE != 0 {
			continue
		}
		if rsect.Characteristics&(pe.IMAGE_SCN_CNT_CODE|pe.IMAGE_SCN_CNT_INITIALIZED_DATA|pe.IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0 {
			// This has been seen for .idata sections, which we
			// want to ignore. See issues 5106 and 5273.
			continue
		}

		splitResources := strings.HasPrefix(rsect.Name, ".rsrc$")
		issehsect := issehsect(arch, rsect)
		sb := l.MakeSymbolUpdater(state.sectsyms[rsect])
		for j, r := range rsect.Relocs {
			if int(r.SymbolTableIndex) >= len(f.COFFSymbols) {
				return nil, fmt.Errorf("relocation number %d symbol index idx=%d cannot be large then number of symbols %d", j, r.SymbolTableIndex, len(f.COFFSymbols))
			}
			pesym := &f.COFFSymbols[r.SymbolTableIndex]
			_, gosym, err := state.readpesym(pesym)
			if err != nil {
				return nil, err
			}
			if gosym == 0 {
				name, err := pesym.FullName(f.StringTable)
				if err != nil {
					name = string(pesym.Name[:])
				}
				return nil, fmt.Errorf("reloc of invalid sym %s idx=%d type=%d", name, r.SymbolTableIndex, pesym.Type)
			}

			rSym := gosym
			rSize := uint8(4)
			rOff := int32(r.VirtualAddress)
			var rAdd int64
			var rType objabi.RelocType
			switch arch.Family {
			default:
				return nil, fmt.Errorf("%s: unsupported arch %v", pn, arch.Family)
			case sys.I386, sys.AMD64:
				switch r.Type {
				default:
					return nil, fmt.Errorf("%s: %v: unknown relocation type %v", pn, state.sectsyms[rsect], r.Type)

				case IMAGE_REL_I386_REL32, IMAGE_REL_AMD64_REL32,
					IMAGE_REL_AMD64_ADDR32, // R_X86_64_PC32
					IMAGE_REL_AMD64_ADDR32NB:
					if r.Type == IMAGE_REL_AMD64_ADDR32NB {
						rType = objabi.R_PEIMAGEOFF
					} else {
						rType = objabi.R_PCREL
					}

					rAdd = int64(int32(binary.LittleEndian.Uint32(state.sectdata[rsect][rOff:])))

				case IMAGE_REL_I386_DIR32NB, IMAGE_REL_I386_DIR32:
					if r.Type == IMAGE_REL_I386_DIR32NB {
						rType = objabi.R_PEIMAGEOFF
					} else {
						rType = objabi.R_ADDR
					}

					// load addend from image
					rAdd = int64(int32(binary.LittleEndian.Uint32(state.sectdata[rsect][rOff:])))

				case IMAGE_REL_AMD64_ADDR64: // R_X86_64_64
					rSize = 8

					rType = objabi.R_ADDR

					// load addend from image
					rAdd = int64(binary.LittleEndian.Uint64(state.sectdata[rsect][rOff:]))
				}

			case sys.ARM:
				switch r.Type {
				default:
					return nil, fmt.Errorf("%s: %v: unknown ARM relocation type %v", pn, state.sectsyms[rsect], r.Type)

				case IMAGE_REL_ARM_SECREL:
					rType = objabi.R_PCREL

					rAdd = int64(int32(binary.LittleEndian.Uint32(state.sectdata[rsect][rOff:])))

				case IMAGE_REL_ARM_ADDR32, IMAGE_REL_ARM_ADDR32NB:
					if r.Type == IMAGE_REL_ARM_ADDR32NB {
						rType = objabi.R_PEIMAGEOFF
					} else {
						rType = objabi.R_ADDR
					}

					rAdd = int64(int32(binary.LittleEndian.Uint32(state.sectdata[rsect][rOff:])))

				case IMAGE_REL_ARM_BRANCH24:
					rType = objabi.R_CALLARM

					rAdd = int64(int32(binary.LittleEndian.Uint32(state.sectdata[rsect][rOff:])))
				}

			case sys.ARM64:
				switch r.Type {
				default:
					return nil, fmt.Errorf("%s: %v: unknown ARM64 relocation type %v", pn, state.sectsyms[rsect], r.Type)

				case IMAGE_REL_ARM64_ADDR32, IMAGE_REL_ARM64_ADDR32NB:
					if r.Type == IMAGE_REL_ARM64_ADDR32NB {
						rType = objabi.R_PEIMAGEOFF
					} else {
						rType = objabi.R_ADDR
					}

					rAdd = int64(int32(binary.LittleEndian.Uint32(state.sectdata[rsect][rOff:])))
				}
			}

			// ld -r could generate multiple section symbols for the
			// same section but with different values, we have to take
			// that into account, or in the case of split resources,
			// the section and its symbols are split into two sections.
			if issect(pesym) || splitResources {
				rAdd += int64(pesym.Value)
			}
			if issehsect {
				// .pdata and .xdata sections can contain records
				// associated to functions that won't be used in
				// the final binary, in which case the relocation
				// target symbol won't be reachable.
				rType |= objabi.R_WEAK
			}

			rel, _ := sb.AddRel(rType)
			rel.SetOff(rOff)
			rel.SetSiz(rSize)
			rel.SetSym(rSym)
			rel.SetAdd(rAdd)

		}

		sb.SortRelocs()
	}

	// enter sub-symbols into symbol table.
	for i, numaux := 0, 0; i < len(f.COFFSymbols); i += numaux + 1 {
		pesym := &f.COFFSymbols[i]

		numaux = int(pesym.NumberOfAuxSymbols)

		name, err := pesym.FullName(f.StringTable)
		if err != nil {
			return nil, err
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
		if pesym.SectionNumber == IMAGE_SYM_ABSOLUTE && bytes.Equal(pesym.Name[:], []byte("@feat.00")) {
			// Microsoft's linker looks at whether all input objects have an empty
			// section called @feat.00. If all of them do, then it enables SEH;
			// otherwise it doesn't enable that feature. So, since around the Windows
			// XP SP2 era, most tools that make PE objects just tack on that section,
			// so that it won't gimp Microsoft's linker logic. Go doesn't support SEH,
			// so in theory, none of this really matters to us. But actually, if the
			// linker tries to ingest an object with @feat.00 -- which are produced by
			// LLVM's resource compiler, for example -- it chokes because of the
			// IMAGE_SYM_ABSOLUTE section that it doesn't know how to deal with. Since
			// @feat.00 is just a marking anyway, skip IMAGE_SYM_ABSOLUTE sections that
			// are called @feat.00.
			continue
		}
		var sect *pe.Section
		if pesym.SectionNumber > 0 {
			sect = f.Sections[pesym.SectionNumber-1]
			if _, found := state.sectsyms[sect]; !found {
				continue
			}
		}

		bld, s, err := state.readpesym(pesym)
		if err != nil {
			return nil, err
		}

		if pesym.SectionNumber == 0 { // extern
			if l.SymType(s) == sym.SXREF && pesym.Value > 0 { // global data
				bld = makeUpdater(l, bld, s)
				bld.SetType(sym.SNOPTRDATA)
				bld.SetSize(int64(pesym.Value))
			}

			continue
		} else if pesym.SectionNumber > 0 && int(pesym.SectionNumber) <= len(f.Sections) {
			sect = f.Sections[pesym.SectionNumber-1]
			if _, found := state.sectsyms[sect]; !found {
				return nil, fmt.Errorf("%s: %v: missing sect.sym", pn, s)
			}
		} else {
			return nil, fmt.Errorf("%s: %v: sectnum < 0!", pn, s)
		}

		if sect == nil {
			return nil, nil
		}

		// Check for COMDAT symbol.
		if sz, ok1 := state.comdats[uint16(pesym.SectionNumber-1)]; ok1 {
			if psz, ok2 := comdatDefinitions[l.SymName(s)]; ok2 {
				if sz == psz {
					//  OK to discard, we've seen an instance
					// already.
					continue
				}
			}
		}
		if l.OuterSym(s) != 0 {
			if l.AttrDuplicateOK(s) {
				continue
			}
			outerName := l.SymName(l.OuterSym(s))
			sectName := l.SymName(state.sectsyms[sect])
			return nil, fmt.Errorf("%s: duplicate symbol reference: %s in both %s and %s", pn, l.SymName(s), outerName, sectName)
		}

		bld = makeUpdater(l, bld, s)
		sectsym := state.sectsyms[sect]
		bld.SetType(l.SymType(sectsym))
		l.AddInteriorSym(sectsym, s)
		bld.SetValue(int64(pesym.Value))
		bld.SetSize(4)
		if l.SymType(sectsym) == sym.STEXT {
			if bld.External() && !bld.DuplicateOK() {
				return nil, fmt.Errorf("%s: duplicate symbol definition", l.SymName(s))
			}
			bld.SetExternal(true)
		}
		if sz, ok := state.comdats[uint16(pesym.SectionNumber-1)]; ok {
			// This is a COMDAT definition. Record that we're picking
			// this instance so that we can ignore future defs.
			if _, ok := comdatDefinitions[l.SymName(s)]; ok {
				return nil, fmt.Errorf("internal error: preexisting COMDAT definition for %q", name)
			}
			comdatDefinitions[l.SymName(s)] = sz
		}
	}

	// Sort outer lists by address, adding to textp.
	// This keeps textp in increasing address order.
	for _, sect := range f.Sections {
		s := state.sectsyms[sect]
		if s == 0 {
			continue
		}
		l.SortSub(s)
		importSymsState.secSyms = append(importSymsState.secSyms, s)
		if l.SymType(s) == sym.STEXT {
			for ; s != 0; s = l.SubSym(s) {
				if l.AttrOnList(s) {
					return nil, fmt.Errorf("symbol %s listed multiple times", l.SymName(s))
				}
				l.SetAttrOnList(s, true)
				ls.Textp = append(ls.Textp, s)
			}
		}
	}

	if ls.PData != 0 {
		processSEH(l, arch, ls.PData, ls.XData)
	}

	return &ls, nil
}

// PostProcessImports works to resolve inconsistencies with DLL import
// symbols; it is needed when building with more "modern" C compilers
// with internal linkage.
//
// Background: DLL import symbols are data (SNOPTRDATA) symbols whose
// name is of the form "__imp_XXX", which contain a pointer/reference
// to symbol XXX. It's possible to have import symbols for both data
// symbols ("__imp__fmode") and text symbols ("__imp_CreateEventA").
// In some case import symbols are just references to some external
// thing, and in other cases we see actual definitions of import
// symbols when reading host objects.
//
// Previous versions of the linker would in most cases immediately
// "forward" import symbol references, e.g. treat a references to
// "__imp_XXX" a references to "XXX", however this doesn't work well
// with more modern compilers, where you can sometimes see import
// symbols that are defs (as opposed to external refs).
//
// The main actions taken below are to search for references to
// SDYNIMPORT symbols in host object text/data sections and flag the
// symbols for later fixup. When we see a reference to an import
// symbol __imp_XYZ where XYZ corresponds to some SDYNIMPORT symbol,
// we flag the symbol (via GOT setting) so that it can be redirected
// to XYZ later in windynrelocsym. When we see a direct reference to
// an SDYNIMPORT symbol XYZ, we also flag the symbol (via PLT setting)
// to indicated that the reference will need to be redirected to a
// stub.
func PostProcessImports() error {
	ldr := importSymsState.l
	arch := importSymsState.arch
	keeprelocneeded := make(map[loader.Sym]loader.Sym)
	for _, s := range importSymsState.secSyms {
		isText := ldr.SymType(s) == sym.STEXT
		relocs := ldr.Relocs(s)
		for i := 0; i < relocs.Count(); i++ {
			r := relocs.At(i)
			rs := r.Sym()
			if ldr.SymType(rs) == sym.SDYNIMPORT {
				// Tag the symbol for later stub generation.
				ldr.SetPlt(rs, CreateImportStubPltToken)
				continue
			}
			isym, err := LookupBaseFromImport(rs, ldr, arch)
			if err != nil {
				return err
			}
			if isym == 0 {
				continue
			}
			if ldr.SymType(isym) != sym.SDYNIMPORT {
				continue
			}
			// For non-text symbols, forward the reference from __imp_X to
			// X immediately.
			if !isText {
				r.SetSym(isym)
				continue
			}
			// Flag this imp symbol to be processed later in windynrelocsym.
			ldr.SetGot(rs, RedirectToDynImportGotToken)
			// Consistency check: should be no PLT token here.
			splt := ldr.SymPlt(rs)
			if splt != -1 {
				return fmt.Errorf("internal error: import symbol %q has invalid PLT setting %d", ldr.SymName(rs), splt)
			}
			// Flag for dummy relocation.
			keeprelocneeded[rs] = isym
		}
	}
	for k, v := range keeprelocneeded {
		sb := ldr.MakeSymbolUpdater(k)
		r, _ := sb.AddRel(objabi.R_KEEP)
		r.SetSym(v)
	}
	importSymsState = nil
	return nil
}

func issehsect(arch *sys.Arch, s *pe.Section) bool {
	return arch.Family == sys.AMD64 && (s.Name == ".pdata" || s.Name == ".xdata")
}

func issect(s *pe.COFFSymbol) bool {
	return s.StorageClass == IMAGE_SYM_CLASS_STATIC && s.Type == 0 && s.Name[0] == '.'
}

func (state *peLoaderState) readpesym(pesym *pe.COFFSymbol) (*loader.SymbolBuilder, loader.Sym, error) {
	symname, err := pesym.FullName(state.f.StringTable)
	if err != nil {
		return nil, 0, err
	}
	var name string
	if issect(pesym) {
		name = state.l.SymName(state.sectsyms[state.f.Sections[pesym.SectionNumber-1]])
	} else {
		name = symname
		// A note on the "_main" exclusion below: the main routine
		// defined by the Go runtime is named "_main", not "main", so
		// when reading references to _main from a host object we want
		// to avoid rewriting "_main" to "main" in this specific
		// instance. See #issuecomment-1143698749 on #35006 for more
		// details on this problem.
		if state.arch.Family == sys.I386 && name[0] == '_' && name != "_main" && !strings.HasPrefix(name, "__imp_") {
			name = name[1:] // _Name => Name
		}
	}

	// remove last @XXX
	if i := strings.LastIndex(name, "@"); i >= 0 {
		name = name[:i]
	}

	var s loader.Sym
	var bld *loader.SymbolBuilder
	// Microsoft's PE documentation is contradictory. It says that the symbol's complex type
	// is stored in the pesym.Type most significant byte, but MSVC, LLVM, and mingw store it
	// in the 4 high bits of the less significant byte.
	switch uint8(pesym.Type&0xf0) >> 4 {
	default:
		return nil, 0, fmt.Errorf("%s: invalid symbol type %d", symname, pesym.Type)

	case IMAGE_SYM_DTYPE_FUNCTION, IMAGE_SYM_DTYPE_NULL:
		switch pesym.StorageClass {
		case IMAGE_SYM_CLASS_EXTERNAL: //global
			s = state.l.LookupOrCreateCgoExport(name, 0)

		case IMAGE_SYM_CLASS_NULL, IMAGE_SYM_CLASS_STATIC, IMAGE_SYM_CLASS_LABEL:
			s = state.l.LookupOrCreateCgoExport(name, state.localSymVersion)
			bld = makeUpdater(state.l, bld, s)
			bld.SetDuplicateOK(true)

		default:
			return nil, 0, fmt.Errorf("%s: invalid symbol binding %d", symname, pesym.StorageClass)
		}
	}

	if s != 0 && state.l.SymType(s) == 0 && (pesym.StorageClass != IMAGE_SYM_CLASS_STATIC || pesym.Value != 0) {
		bld = makeUpdater(state.l, bld, s)
		bld.SetType(sym.SXREF)
	}

	return bld, s, nil
}

// preprocessSymbols walks the COFF symbols for the PE file we're
// reading and looks for cases where we have both a symbol definition
// for "XXX" and an "__imp_XXX" symbol, recording these cases in a map
// in the state struct. This information will be used in readpesym()
// above to give such symbols special treatment. This function also
// gathers information about COMDAT sections/symbols for later use
// in readpesym().
func (state *peLoaderState) preprocessSymbols() error {

	// Locate comdat sections.
	state.comdats = make(map[uint16]int64)
	for i, s := range state.f.Sections {
		if s.Characteristics&uint32(pe.IMAGE_SCN_LNK_COMDAT) != 0 {
			state.comdats[uint16(i)] = int64(s.Size)
		}
	}

	// Examine symbol defs.
	for i, numaux := 0, 0; i < len(state.f.COFFSymbols); i += numaux + 1 {
		pesym := &state.f.COFFSymbols[i]
		numaux = int(pesym.NumberOfAuxSymbols)
		if pesym.SectionNumber == 0 { // extern
			continue
		}
		symname, err := pesym.FullName(state.f.StringTable)
		if err != nil {
			return err
		}
		if _, isc := state.comdats[uint16(pesym.SectionNumber-1)]; !isc {
			continue
		}
		if pesym.StorageClass != uint8(IMAGE_SYM_CLASS_STATIC) {
			continue
		}
		// This symbol corresponds to a COMDAT section. Read the
		// aux data for it.
		auxsymp, err := state.f.COFFSymbolReadSectionDefAux(i)
		if err != nil {
			return fmt.Errorf("unable to read aux info for section def symbol %d %s: pe.COFFSymbolReadComdatInfo returns %v", i, symname, err)
		}
		if auxsymp.Selection == pe.IMAGE_COMDAT_SELECT_SAME_SIZE {
			// This is supported.
		} else if auxsymp.Selection == pe.IMAGE_COMDAT_SELECT_ANY {
			// Also supported.
			state.comdats[uint16(pesym.SectionNumber-1)] = int64(-1)
		} else {
			// We don't support any of the other strategies at the
			// moment. I suspect that we may need to also support
			// "associative", we'll see.
			return fmt.Errorf("internal error: unsupported COMDAT selection strategy found in path=%s sec=%d strategy=%d idx=%d, please file a bug", state.pn, auxsymp.SecNum, auxsymp.Selection, i)
		}
	}
	return nil
}

// LookupBaseFromImport examines the symbol "s" to see if it
// corresponds to an import symbol (name of the form "__imp_XYZ") and
// if so, it looks up the underlying target of the import symbol and
// returns it. An error is returned if the symbol is of the form
// "__imp_XYZ" but no XYZ can be found.
func LookupBaseFromImport(s loader.Sym, ldr *loader.Loader, arch *sys.Arch) (loader.Sym, error) {
	sname := ldr.SymName(s)
	if !strings.HasPrefix(sname, "__imp_") {
		return 0, nil
	}
	basename := sname[len("__imp_"):]
	if arch.Family == sys.I386 && basename[0] == '_' {
		basename = basename[1:] // _Name => Name
	}
	isym := ldr.Lookup(basename, 0)
	if isym == 0 {
		return 0, fmt.Errorf("internal error: import symbol %q with no underlying sym", sname)
	}
	return isym, nil
}
