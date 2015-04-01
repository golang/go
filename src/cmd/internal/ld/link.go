// Derived from Inferno utils/6l/l.h and related files.
// http://code.google.com/p/inferno-os/source/browse/utils/6l/l.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

import "encoding/binary"

type LSym struct {
	Name        string
	Extname     string
	Type        int16
	Version     int16
	Dupok       uint8
	Cfunc       uint8
	External    uint8
	Nosplit     uint8
	Reachable   bool
	Cgoexport   uint8
	Special     uint8
	Stkcheck    uint8
	Hide        uint8
	Leaf        uint8
	Localentry  uint8
	Onlist      uint8
	Dynid       int32
	Plt         int32
	Got         int32
	Align       int32
	Elfsym      int32
	Args        int32
	Locals      int32
	Value       int64
	Size        int64
	Hash        *LSym
	Allsym      *LSym
	Next        *LSym
	Sub         *LSym
	Outer       *LSym
	Gotype      *LSym
	Reachparent *LSym
	Queue       *LSym
	File        string
	Dynimplib   string
	Dynimpvers  string
	Sect        interface{}
	Autom       *Auto
	Pcln        *Pcln
	P           []byte
	R           []Reloc
	Local       bool
}

type Reloc struct {
	Off     int32
	Siz     uint8
	Done    uint8
	Type    int32
	Variant int32
	Add     int64
	Xadd    int64
	Sym     *LSym
	Xsym    *LSym
}

type Auto struct {
	Asym    *LSym
	Link    *Auto
	Aoffset int32
	Name    int16
	Gotype  *LSym
}

type Link struct {
	Thechar   int32
	Thestring string
	Goarm     int32
	Headtype  int
	Arch      *LinkArch
	Debugasm  int32
	Debugvlog int32
	Bso       *Biobuf
	Windows   int32
	Goroot    string
	Hash      map[symVer]*LSym
	Allsym    *LSym
	Nsymbol   int32
	Tlsg      *LSym
	Libdir    []string
	Library   []Library
	Shlibs    []string
	Tlsoffset int
	Diag      func(string, ...interface{})
	Cursym    *LSym
	Version   int
	Textp     *LSym
	Etextp    *LSym
	Nhistfile int32
	Filesyms  *LSym
}

type LinkArch struct {
	ByteOrder binary.ByteOrder
	Name      string
	Thechar   int
	Minlc     int
	Ptrsize   int
	Regsize   int
}

type Library struct {
	Objref string
	Srcref string
	File   string
	Pkg    string
	Shlib  string
}

type Pcln struct {
	Pcsp        Pcdata
	Pcfile      Pcdata
	Pcline      Pcdata
	Pcdata      []Pcdata
	Npcdata     int
	Funcdata    []*LSym
	Funcdataoff []int64
	Nfuncdata   int
	File        []*LSym
	Nfile       int
	Mfile       int
	Lastfile    *LSym
	Lastindex   int
}

type Pcdata struct {
	P []byte
}

type Pciter struct {
	d       Pcdata
	p       []byte
	pc      uint32
	nextpc  uint32
	pcscale uint32
	value   int32
	start   int
	done    int
}

// LSym.type
const (
	Sxxx = iota
	STEXT
	SELFRXSECT
	STYPE
	SSTRING
	SGOSTRING
	SGOFUNC
	SRODATA
	SFUNCTAB
	STYPELINK
	SSYMTAB
	SPCLNTAB
	SELFROSECT
	SMACHOPLT
	SELFSECT
	SMACHO
	SMACHOGOT
	SWINDOWS
	SELFGOT
	SNOPTRDATA
	SINITARR
	SDATA
	SBSS
	SNOPTRBSS
	STLSBSS
	SXREF
	SMACHOSYMSTR
	SMACHOSYMTAB
	SMACHOINDIRECTPLT
	SMACHOINDIRECTGOT
	SFILE
	SFILEPATH
	SCONST
	SDYNIMPORT
	SHOSTOBJ
	SSUB    = 1 << 8
	SMASK   = SSUB - 1
	SHIDDEN = 1 << 9
)

// Reloc.type
const (
	R_ADDR = 1 + iota
	R_ADDRPOWER
	R_ADDRARM64
	R_SIZE
	R_CALL
	R_CALLARM
	R_CALLARM64
	R_CALLIND
	R_CALLPOWER
	R_CONST
	R_PCREL
	R_TLS
	R_TLS_LE
	R_TLS_IE
	R_GOTOFF
	R_PLT0
	R_PLT1
	R_PLT2
	R_USEFIELD
	R_POWER_TOC
	R_GOTPCREL
)

// Reloc.variant
const (
	RV_NONE = iota
	RV_POWER_LO
	RV_POWER_HI
	RV_POWER_HA
	RV_POWER_DS
	RV_CHECK_OVERFLOW = 1 << 8
	RV_TYPE_MASK      = RV_CHECK_OVERFLOW - 1
)

// Auto.name
const (
	A_AUTO = 1 + iota
	A_PARAM
)

const (
	LINKHASH = 100003
)

// Pcdata iterator.
//	for(pciterinit(ctxt, &it, &pcd); !it.done; pciternext(&it)) { it.value holds in [it.pc, it.nextpc) }

// symbol version, incremented each time a file is loaded.
// version==1 is reserved for savehist.
const (
	HistVersion = 1
)

// Link holds the context for writing object code from a compiler
// to be linker input or for reading that input into the linker.

// LinkArch is the definition of a single architecture.

/* executable header types */
const (
	Hunknown = 0 + iota
	Hdarwin
	Hdragonfly
	Helf
	Hfreebsd
	Hlinux
	Hnacl
	Hnetbsd
	Hopenbsd
	Hplan9
	Hsolaris
	Hwindows
)

const (
	LinkAuto = 0 + iota
	LinkInternal
	LinkExternal
)
