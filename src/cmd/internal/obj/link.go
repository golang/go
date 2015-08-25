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

// An Addr is an argument to an instruction.
// The general forms and their encodings are:
//
//	sym±offset(symkind)(reg)(index*scale)
//		Memory reference at address &sym(symkind) + offset + reg + index*scale.
//		Any of sym(symkind), ±offset, (reg), (index*scale), and *scale can be omitted.
//		If (reg) and *scale are both omitted, the resulting expression (index) is parsed as (reg).
//		To force a parsing as index*scale, write (index*1).
//		Encoding:
//			type = TYPE_MEM
//			name = symkind (NAME_AUTO, ...) or 0 (NAME_NONE)
//			sym = sym
//			offset = ±offset
//			reg = reg (REG_*)
//			index = index (REG_*)
//			scale = scale (1, 2, 4, 8)
//
//	$<mem>
//		Effective address of memory reference <mem>, defined above.
//		Encoding: same as memory reference, but type = TYPE_ADDR.
//
//	$<±integer value>
//		This is a special case of $<mem>, in which only ±offset is present.
//		It has a separate type for easy recognition.
//		Encoding:
//			type = TYPE_CONST
//			offset = ±integer value
//
//	*<mem>
//		Indirect reference through memory reference <mem>, defined above.
//		Only used on x86 for CALL/JMP *sym(SB), which calls/jumps to a function
//		pointer stored in the data word sym(SB), not a function named sym(SB).
//		Encoding: same as above, but type = TYPE_INDIR.
//
//	$*$<mem>
//		No longer used.
//		On machines with actual SB registers, $*$<mem> forced the
//		instruction encoding to use a full 32-bit constant, never a
//		reference relative to SB.
//
//	$<floating point literal>
//		Floating point constant value.
//		Encoding:
//			type = TYPE_FCONST
//			val = floating point value
//
//	$<string literal, up to 8 chars>
//		String literal value (raw bytes used for DATA instruction).
//		Encoding:
//			type = TYPE_SCONST
//			val = string
//
//	<register name>
//		Any register: integer, floating point, control, segment, and so on.
//		If looking for specific register kind, must check type and reg value range.
//		Encoding:
//			type = TYPE_REG
//			reg = reg (REG_*)
//
//	x(PC)
//		Encoding:
//			type = TYPE_BRANCH
//			val = Prog* reference OR ELSE offset = target pc (branch takes priority)
//
//	$±x-±y
//		Final argument to TEXT, specifying local frame size x and argument size y.
//		In this form, x and y are integer literals only, not arbitrary expressions.
//		This avoids parsing ambiguities due to the use of - as a separator.
//		The ± are optional.
//		If the final argument to TEXT omits the -±y, the encoding should still
//		use TYPE_TEXTSIZE (not TYPE_CONST), with u.argsize = ArgsSizeUnknown.
//		Encoding:
//			type = TYPE_TEXTSIZE
//			offset = x
//			val = int32(y)
//
//	reg<<shift, reg>>shift, reg->shift, reg@>shift
//		Shifted register value, for ARM.
//		In this form, reg must be a register and shift can be a register or an integer constant.
//		Encoding:
//			type = TYPE_SHIFT
//			offset = (reg&15) | shifttype<<5 | count
//			shifttype = 0, 1, 2, 3 for <<, >>, ->, @>
//			count = (reg&15)<<8 | 1<<4 for a register shift count, (n&31)<<7 for an integer constant.
//
//	(reg, reg)
//		A destination register pair. When used as the last argument of an instruction,
//		this form makes clear that both registers are destinations.
//		Encoding:
//			type = TYPE_REGREG
//			reg = first register
//			offset = second register
//
//	[reg, reg, reg-reg]
//		Register list for ARM.
//		Encoding:
//			type = TYPE_REGLIST
//			offset = bit mask of registers in list; R0 is low bit.
//
//	reg, reg
//		Register pair for ARM.
//		TYPE_REGREG2
//
//	(reg+reg)
//		Register pair for PPC64.
//		Encoding:
//			type = TYPE_MEM
//			reg = first register
//			index = second register
//			scale = 1
//
type Addr struct {
	Type   int16
	Reg    int16
	Index  int16
	Scale  int16 // Sometimes holds a register.
	Name   int8
	Class  int8
	Etype  uint8
	Offset int64
	Width  int64
	Sym    *LSym
	Gotype *LSym

	// argument value:
	//	for TYPE_SCONST, a string
	//	for TYPE_FCONST, a float64
	//	for TYPE_BRANCH, a *Prog (optional)
	//	for TYPE_TEXTSIZE, an int32 (optional)
	Val interface{}

	Node interface{} // for use by compiler
}

const (
	NAME_NONE = 0 + iota
	NAME_EXTERN
	NAME_STATIC
	NAME_AUTO
	NAME_PARAM
	// A reference to name@GOT(SB) is a reference to the entry in the global offset
	// table for 'name'.
	NAME_GOTREF
)

const (
	TYPE_NONE = 0
)

const (
	TYPE_BRANCH = 5 + iota
	TYPE_TEXTSIZE
	TYPE_MEM
	TYPE_CONST
	TYPE_FCONST
	TYPE_SCONST
	TYPE_REG
	TYPE_ADDR
	TYPE_SHIFT
	TYPE_REGREG
	TYPE_REGREG2
	TYPE_INDIR
	TYPE_REGLIST
)

// TODO(rsc): Describe prog.
// TODO(rsc): Describe TEXT/GLOBL flag in from3, DATA width in from3.
type Prog struct {
	Ctxt   *Link
	Link   *Prog
	From   Addr
	From3  *Addr // optional
	To     Addr
	Opt    interface{}
	Forwd  *Prog
	Pcond  *Prog
	Rel    *Prog // Source of forward jumps on x86; pcrel on arm
	Pc     int64
	Lineno int32
	Spadj  int32
	As     int16
	Reg    int16
	RegTo2 int16 // 2nd register output operand
	Mark   uint16
	Optab  uint16
	Scond  uint8
	Back   uint8
	Ft     uint8
	Tt     uint8
	Isize  uint8
	Mode   int8

	Info ProgInfo
}

// From3Type returns From3.Type, or TYPE_NONE when From3 is nil.
func (p *Prog) From3Type() int16 {
	if p.From3 == nil {
		return TYPE_NONE
	}
	return p.From3.Type
}

// From3Offset returns From3.Offset, or 0 when From3 is nil.
func (p *Prog) From3Offset() int64 {
	if p.From3 == nil {
		return 0
	}
	return p.From3.Offset
}

// ProgInfo holds information about the instruction for use
// by clients such as the compiler. The exact meaning of this
// data is up to the client and is not interpreted by the cmd/internal/obj/... packages.
type ProgInfo struct {
	Flags    uint32   // flag bits
	Reguse   uint64   // registers implicitly used by this instruction
	Regset   uint64   // registers implicitly set by this instruction
	Regindex uint64   // registers used by addressing mode
	_        struct{} // to prevent unkeyed literals
}

// Prog.as opcodes.
// These are the portable opcodes, common to all architectures.
// Each architecture defines many more arch-specific opcodes,
// with values starting at A_ARCHSPECIFIC.
// Each architecture adds an offset to this so each machine has
// distinct space for its instructions. The offset is a power of
// two so it can be masked to return to origin zero.
// See the definitions of ABase386 etc.
const (
	AXXX = 0 + iota
	ACALL
	ACHECKNIL
	ADATA
	ADUFFCOPY
	ADUFFZERO
	AEND
	AFUNCDATA
	AGLOBL
	AJMP
	ANOP
	APCDATA
	ARET
	ATEXT
	ATYPE
	AUNDEF
	AUSEFIELD
	AVARDEF
	AVARKILL
	A_ARCHSPECIFIC
)

// An LSym is the sort of symbol that is written to an object file.
type LSym struct {
	Name      string
	Type      int16
	Version   int16
	Dupok     uint8
	Cfunc     uint8
	Nosplit   uint8
	Leaf      uint8
	Seenglobl uint8
	Onlist    uint8
	// Local means make the symbol local even when compiling Go code to reference Go
	// symbols in other shared libraries, as in this mode symbols are global by
	// default. "local" here means in the sense of the dynamic linker, i.e. not
	// visible outside of the module (shared library or executable) that contains its
	// definition. (When not compiling to support Go shared libraries, all symbols are
	// local in this sense unless there is a cgo_export_* directive).
	Local  bool
	Args   int32
	Locals int32
	Value  int64
	Size   int64
	Next   *LSym
	Gotype *LSym
	Autom  *Auto
	Text   *Prog
	Etext  *Prog
	Pcln   *Pcln
	P      []byte
	R      []Reloc
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

// LSym.type
const (
	Sxxx = iota
	STEXT
	SELFRXSECT
	STYPE
	SSTRING
	SGOSTRING
	SGOFUNC
	SGCBITS
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
	SSUB       = 1 << 8
	SMASK      = SSUB - 1
	SHIDDEN    = 1 << 9
	SCONTAINER = 1 << 10 // has a sub-symbol
)

type Reloc struct {
	Off  int32
	Siz  uint8
	Type int32
	Add  int64
	Sym  *LSym
}

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
	// R_TLS (only used on arm currently, and not on android and darwin where tlsg is
	// a regular variable) resolves to data needed to access the thread-local g. It is
	// interpreted differently depending on toolchain flags to implement either the
	// "local exec" or "inital exec" model for tls access.
	// TODO(mwhudson): change to use R_TLS_LE or R_TLS_IE as appropriate, not having
	// R_TLS do double duty.
	R_TLS
	// R_TLS_LE (only used on 386 and amd64 currently) resolves to the offset of the
	// thread-local g from the thread local base and is used to implement the "local
	// exec" model for tls access (r.Sym is not set by the compiler for this case but
	// is set to Tlsg in the linker when externally linking).
	R_TLS_LE
	// R_TLS_IE (only used on 386 and amd64 currently) resolves to the PC-relative
	// offset to a GOT slot containing the offset the thread-local g from the thread
	// local base and is used to implemented the "initial exec" model for tls access
	// (r.Sym is not set by the compiler for this case but is set to Tlsg in the
	// linker when externally linking).
	R_TLS_IE
	R_GOTOFF
	R_PLT0
	R_PLT1
	R_PLT2
	R_USEFIELD
	R_POWER_TOC
	R_GOTPCREL
)

type Auto struct {
	Asym    *LSym
	Link    *Auto
	Aoffset int32
	Name    int16
	Gotype  *LSym
}

// Auto.name
const (
	A_AUTO = 1 + iota
	A_PARAM
)

type Pcdata struct {
	P []byte
}

// Pcdata iterator.
//      for(pciterinit(ctxt, &it, &pcd); !it.done; pciternext(&it)) { it.value holds in [it.pc, it.nextpc) }
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

// symbol version, incremented each time a file is loaded.
// version==1 is reserved for savehist.
const (
	HistVersion = 1
)

// Link holds the context for writing object code from a compiler
// to be linker input or for reading that input into the linker.
type Link struct {
	Goarm              int32
	Headtype           int
	Arch               *LinkArch
	Debugasm           int32
	Debugvlog          int32
	Debugdivmod        int32
	Debugpcln          int32
	Flag_shared        int32
	Flag_dynlink       bool
	Bso                *Biobuf
	Pathname           string
	Windows            int32
	Goroot             string
	Goroot_final       string
	Enforce_data_order int32
	Hash               map[SymVer]*LSym
	LineHist           LineHist
	Imports            []string
	Plist              *Plist
	Plast              *Plist
	Sym_div            *LSym
	Sym_divu           *LSym
	Sym_mod            *LSym
	Sym_modu           *LSym
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
	Tlsoffset          int
	Diag               func(string, ...interface{})
	Mode               int
	Cursym             *LSym
	Version            int
	Textp              *LSym
	Etextp             *LSym
}

type SymVer struct {
	Name    string
	Version int // TODO: make int16 to match LSym.Version?
}

// LinkArch is the definition of a single architecture.
type LinkArch struct {
	ByteOrder  binary.ByteOrder
	Name       string
	Thechar    int
	Preprocess func(*Link, *LSym)
	Assemble   func(*Link, *LSym)
	Follow     func(*Link, *LSym)
	Progedit   func(*Link, *Prog)
	UnaryDst   map[int]bool // Instruction takes one operand, a destination.
	Minlc      int
	Ptrsize    int
	Regsize    int
}

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

type Plist struct {
	Name    *LSym
	Firstpc *Prog
	Recur   int
	Link    *Plist
}

/*
 * start a new Prog list.
 */
func Linknewplist(ctxt *Link) *Plist {
	pl := new(Plist)
	if ctxt.Plist == nil {
		ctxt.Plist = pl
	} else {
		ctxt.Plast.Link = pl
	}
	ctxt.Plast = pl
	return pl
}
