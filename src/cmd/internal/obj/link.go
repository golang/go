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

package obj

import "encoding/binary"

type Addr struct {
	Offset int64
	U      struct {
		Sval   string
		Dval   float64
		Branch *Prog
	}
	Sym     *LSym
	Gotype  *LSym
	Type    int16
	Index   uint8
	Scale   int8
	Reg     int8
	Name    int8
	Class   int8
	Etype   uint8
	Offset2 int32
	Node    interface{}
	Width   int64
}

type Prog struct {
	Ctxt     *Link
	Pc       int64
	Lineno   int32
	Link     *Prog
	As       int16
	Scond    uint8
	From     Addr
	Reg      uint8
	From3    Addr
	To       Addr
	Opt      interface{}
	Forwd    *Prog
	Pcond    *Prog
	Comefrom *Prog
	Pcrel    *Prog
	Spadj    int32
	Mark     uint16
	Optab    uint16
	Back     uint8
	Ft       uint8
	Tt       uint8
	Isize    uint8
	Printed  uint8
	Width    int8
	Mode     int8
	TEXTFLAG uint8
}

type LSym struct {
	Name        string
	Extname     string
	Type        int16
	Version     int16
	Dupok       uint8
	Cfunc       uint8
	External    uint8
	Nosplit     uint8
	Reachable   uint8
	Cgoexport   uint8
	Special     uint8
	Stkcheck    uint8
	Hide        uint8
	Leaf        uint8
	Fnptr       uint8
	Localentry  uint8
	Seenglobl   uint8
	Onlist      uint8
	Printed     uint8
	Symid       int16
	Dynid       int32
	Sig         int32
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
	Sect        *struct{}
	Autom       *Auto
	Text        *Prog
	Etext       *Prog
	Pcln        *Pcln
	P           []byte
	R           []Reloc
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
	Type    int16
	Gotype  *LSym
}

type Hist struct {
	Link    *Hist
	Name    string
	Line    int32
	Offset  int32
	Printed uint8
}

type Link struct {
	Thechar            int32
	Thestring          string
	Goarm              int32
	Headtype           int
	Arch               *LinkArch
	Ignore             func(string) int32
	Debugasm           int32
	Debugline          int32
	Debughist          int32
	Debugread          int32
	Debugvlog          int32
	Debugstack         int32
	Debugzerostack     int32
	Debugdivmod        int32
	Debugfloat         int32
	Debugpcln          int32
	Flag_shared        int32
	Iself              int32
	Bso                *Biobuf
	Pathname           string
	Windows            int32
	Trimpath           string
	Goroot             string
	Goroot_final       string
	Enforce_data_order int32
	Hash               [LINKHASH]*LSym
	Allsym             *LSym
	Nsymbol            int32
	Hist               *Hist
	Ehist              *Hist
	Plist              *Plist
	Plast              *Plist
	Sym_div            *LSym
	Sym_divu           *LSym
	Sym_mod            *LSym
	Sym_modu           *LSym
	Symmorestack       [2]*LSym
	Tlsg               *LSym
	Plan9privates      *LSym
	Curp               *Prog
	Printp             *Prog
	Blitrl             *Prog
	Elitrl             *Prog
	Rexflag            int
	Rep                int
	Repn               int
	Lock               int
	Asmode             int
	Andptr             []byte
	And                [100]uint8
	Instoffset         int64
	Autosize           int32
	Armsize            int32
	Pc                 int64
	Libdir             []string
	Library            []Library
	Tlsoffset          int
	Diag               func(string, ...interface{})
	Mode               int
	Curauto            *Auto
	Curhist            *Auto
	Cursym             *LSym
	Version            int
	Textp              *LSym
	Etextp             *LSym
	Histdepth          int32
	Nhistfile          int32
	Filesyms           *LSym
}

type Plist struct {
	Name    *LSym
	Firstpc *Prog
	Recur   int
	Link    *Plist
}

type LinkArch struct {
	Pconv         func(*Prog) string
	Name          string
	Thechar       int
	Endian        int32
	ByteOrder     binary.ByteOrder
	Addstacksplit func(*Link, *LSym)
	Assemble      func(*Link, *LSym)
	Datasize      func(*Prog) int
	Follow        func(*Link, *LSym)
	Iscall        func(*Prog) bool
	Isdata        func(*Prog) bool
	Prg           func() *Prog
	Progedit      func(*Link, *Prog)
	Settextflag   func(*Prog, int)
	Symtype       func(*Addr) int
	Textflag      func(*Prog) int
	Minlc         int
	Ptrsize       int
	Regsize       int
	D_ADDR        int
	D_AUTO        int
	D_BRANCH      int
	D_CONST       int
	D_EXTERN      int
	D_FCONST      int
	D_NONE        int
	D_PARAM       int
	D_SCONST      int
	D_STATIC      int
	D_OREG        int
	ACALL         int
	ADATA         int
	AEND          int
	AFUNCDATA     int
	AGLOBL        int
	AJMP          int
	ANOP          int
	APCDATA       int
	ARET          int
	ATEXT         int
	ATYPE         int
	AUSEFIELD     int
}

type Library struct {
	Objref string
	Srcref string
	File   string
	Pkg    string
}

type Pcln struct {
	Pcsp        Pcdata
	Pcfile      Pcdata
	Pcline      Pcdata
	Pcdata      []Pcdata
	Funcdata    []*LSym
	Funcdataoff []int64
	File        []*LSym
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

// prevent incompatible type signatures between liblink and 8l on Plan 9

// prevent incompatible type signatures between liblink and 8l on Plan 9

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
	R_SIZE
	R_CALL
	R_CALLARM
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

// Auto.type
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

const (
	LittleEndian = 0x04030201
	BigEndian    = 0x01020304
)

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

// asm5.c

// asm6.c

// asm8.c

// asm9.c

// data.c

// go.c

// ld.c
