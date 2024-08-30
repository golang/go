// Derived from Inferno utils/6l/l.h and related files.
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/l.h
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

package sym

import "cmd/internal/objabi"

// A SymKind describes the kind of memory represented by a symbol.
type SymKind uint8

// Defined SymKind values.
//
// TODO(rsc): Give idiomatic Go names.
//
//go:generate stringer -type=SymKind
const (
	Sxxx SymKind = iota
	STEXT
	SELFRXSECT
	SMACHOPLT

	// Read-only sections.
	STYPE
	SSTRING
	SGOSTRING
	SGOFUNC
	SGCBITS
	SRODATA
	SFUNCTAB

	SELFROSECT

	// Read-only sections with relocations.
	//
	// Types STYPE-SFUNCTAB above are written to the .rodata section by default.
	// When linking a shared object, some conceptually "read only" types need to
	// be written to by relocations and putting them in a section called
	// ".rodata" interacts poorly with the system linkers. The GNU linkers
	// support this situation by arranging for sections of the name
	// ".data.rel.ro.XXX" to be mprotected read only by the dynamic linker after
	// relocations have applied, so when the Go linker is creating a shared
	// object it checks all objects of the above types and bumps any object that
	// has a relocation to it to the corresponding type below, which are then
	// written to sections with appropriate magic names.
	STYPERELRO
	SSTRINGRELRO
	SGOSTRINGRELRO
	SGOFUNCRELRO
	SGCBITSRELRO
	SRODATARELRO
	SFUNCTABRELRO
	SELFRELROSECT

	// Part of .data.rel.ro if it exists, otherwise part of .rodata.
	STYPELINK
	SITABLINK
	SSYMTAB
	SPCLNTAB

	// Writable sections.
	SFirstWritable
	SBUILDINFO
	SELFSECT
	SMACHO
	SMACHOGOT
	SWINDOWS
	SELFGOT
	SNOPTRDATA
	SINITARR
	SDATA
	SXCOFFTOC
	SBSS
	SNOPTRBSS
	SLIBFUZZER_8BIT_COUNTER
	SCOVERAGE_COUNTER
	SCOVERAGE_AUXVAR
	STLSBSS
	SXREF
	SMACHOSYMSTR
	SMACHOSYMTAB
	SMACHOINDIRECTPLT
	SMACHOINDIRECTGOT
	SFILEPATH
	SDYNIMPORT
	SHOSTOBJ
	SUNDEFEXT // Undefined symbol for resolution by external linker

	// Sections for debugging information
	SDWARFSECT
	// DWARF symbol types
	SDWARFCUINFO
	SDWARFCONST
	SDWARFFCN
	SDWARFABSFCN
	SDWARFTYPE
	SDWARFVAR
	SDWARFRANGE
	SDWARFLOC
	SDWARFLINES

	// SEH symbol types
	SSEHUNWINDINFO
	SSEHSECT
)

// AbiSymKindToSymKind maps values read from object files (which are
// of type cmd/internal/objabi.SymKind) to values of type SymKind.
var AbiSymKindToSymKind = [...]SymKind{
	objabi.Sxxx:                    Sxxx,
	objabi.STEXT:                   STEXT,
	objabi.SRODATA:                 SRODATA,
	objabi.SNOPTRDATA:              SNOPTRDATA,
	objabi.SDATA:                   SDATA,
	objabi.SBSS:                    SBSS,
	objabi.SNOPTRBSS:               SNOPTRBSS,
	objabi.STLSBSS:                 STLSBSS,
	objabi.SDWARFCUINFO:            SDWARFCUINFO,
	objabi.SDWARFCONST:             SDWARFCONST,
	objabi.SDWARFFCN:               SDWARFFCN,
	objabi.SDWARFABSFCN:            SDWARFABSFCN,
	objabi.SDWARFTYPE:              SDWARFTYPE,
	objabi.SDWARFVAR:               SDWARFVAR,
	objabi.SDWARFRANGE:             SDWARFRANGE,
	objabi.SDWARFLOC:               SDWARFLOC,
	objabi.SDWARFLINES:             SDWARFLINES,
	objabi.SLIBFUZZER_8BIT_COUNTER: SLIBFUZZER_8BIT_COUNTER,
	objabi.SCOVERAGE_COUNTER:       SCOVERAGE_COUNTER,
	objabi.SCOVERAGE_AUXVAR:        SCOVERAGE_AUXVAR,
	objabi.SSEHUNWINDINFO:          SSEHUNWINDINFO,
}

// ReadOnly are the symbol kinds that form read-only sections. In some
// cases, if they will require relocations, they are transformed into
// rel-ro sections using relROMap.
var ReadOnly = []SymKind{
	STYPE,
	SSTRING,
	SGOSTRING,
	SGOFUNC,
	SGCBITS,
	SRODATA,
	SFUNCTAB,
}

// RelROMap describes the transformation of read-only symbols to rel-ro
// symbols.
var RelROMap = map[SymKind]SymKind{
	STYPE:     STYPERELRO,
	SSTRING:   SSTRINGRELRO,
	SGOSTRING: SGOSTRINGRELRO,
	SGOFUNC:   SGOFUNCRELRO,
	SGCBITS:   SGCBITSRELRO,
	SRODATA:   SRODATARELRO,
	SFUNCTAB:  SFUNCTABRELRO,
}

// IsData returns true if the type is a data type.
func (t SymKind) IsData() bool {
	return t == SDATA || t == SNOPTRDATA || t == SBSS || t == SNOPTRBSS
}

func (t SymKind) IsDWARF() bool {
	return t >= SDWARFSECT && t <= SDWARFLINES
}
