// Derived from Inferno utils/6l/l.h and related files.
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/l.h
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

package obj

import (
	"bufio"
	"cmd/internal/sys"
	"fmt"
)

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
//		Shifted register value, for ARM and ARM64.
//		In this form, reg must be a register and shift can be a register or an integer constant.
//		Encoding:
//			type = TYPE_SHIFT
//		On ARM:
//			offset = (reg&15) | shifttype<<5 | count
//			shifttype = 0, 1, 2, 3 for <<, >>, ->, @>
//			count = (reg&15)<<8 | 1<<4 for a register shift count, (n&31)<<7 for an integer constant.
//		On ARM64:
//			offset = (reg&31)<<16 | shifttype<<22 | (count&63)<<10
//			shifttype = 0, 1, 2 for <<, >>, ->
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
	Reg    int16
	Index  int16
	Scale  int16 // Sometimes holds a register.
	Type   AddrType
	Name   int8
	Class  int8
	Offset int64
	Sym    *LSym

	// argument value:
	//	for TYPE_SCONST, a string
	//	for TYPE_FCONST, a float64
	//	for TYPE_BRANCH, a *Prog (optional)
	//	for TYPE_TEXTSIZE, an int32 (optional)
	Val interface{}

	Node interface{} // for use by compiler
}

type AddrType uint8

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
	TYPE_NONE AddrType = 0

	TYPE_BRANCH AddrType = 5 + iota
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

// Prog describes a single machine instruction.
//
// The general instruction form is:
//
//	As.Scond From, Reg, From3, To, RegTo2
//
// where As is an opcode and the others are arguments:
// From, Reg, From3 are sources, and To, RegTo2 are destinations.
// Usually, not all arguments are present.
// For example, MOVL R1, R2 encodes using only As=MOVL, From=R1, To=R2.
// The Scond field holds additional condition bits for systems (like arm)
// that have generalized conditional execution.
//
// Jump instructions use the Pcond field to point to the target instruction,
// which must be in the same linked list as the jump instruction.
//
// The Progs for a given function are arranged in a list linked through the Link field.
//
// Each Prog is charged to a specific source line in the debug information,
// specified by Lineno, an index into the line history (see LineHist).
// Every Prog has a Ctxt field that defines various context, including the current LineHist.
// Progs should be allocated using ctxt.NewProg(), not new(Prog).
//
// The other fields not yet mentioned are for use by the back ends and should
// be left zeroed by creators of Prog lists.
type Prog struct {
	Ctxt   *Link       // linker context
	Link   *Prog       // next Prog in linked list
	From   Addr        // first source operand
	From3  *Addr       // third source operand (second is Reg below)
	To     Addr        // destination operand (second is RegTo2 below)
	Pcond  *Prog       // target of conditional jump
	Opt    interface{} // available to optimization passes to hold per-Prog state
	Forwd  *Prog       // for x86 back end
	Rel    *Prog       // for x86, arm back ends
	Pc     int64       // for back ends or assembler: virtual or actual program counter, depending on phase
	Lineno int32       // line number of this instruction
	Spadj  int32       // effect of instruction on stack pointer (increment or decrement amount)
	As     As          // assembler opcode
	Reg    int16       // 2nd source operand
	RegTo2 int16       // 2nd destination operand
	Mark   uint16      // bitmask of arch-specific items
	Optab  uint16      // arch-specific opcode index
	Scond  uint8       // condition bits for conditional instruction (e.g., on ARM)
	Back   uint8       // for x86 back end: backwards branch state
	Ft     uint8       // for x86 back end: type index of Prog.From
	Tt     uint8       // for x86 back end: type index of Prog.To
	Isize  uint8       // for x86 back end: size of the instruction in bytes
	Mode   int8        // for x86 back end: 32- or 64-bit mode
}

// From3Type returns From3.Type, or TYPE_NONE when From3 is nil.
func (p *Prog) From3Type() AddrType {
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

// An As denotes an assembler opcode.
// There are some portable opcodes, declared here in package obj,
// that are common to all architectures.
// However, the majority of opcodes are arch-specific
// and are declared in their respective architecture's subpackage.
type As int16

// These are the portable opcodes.
const (
	AXXX As = iota
	ACALL
	ADUFFCOPY
	ADUFFZERO
	AEND
	AFUNCDATA
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
	AVARLIVE
	A_ARCHSPECIFIC
)

// Each architecture is allotted a distinct subspace of opcode values
// for declaring its arch-specific opcodes.
// Within this subspace, the first arch-specific opcode should be
// at offset A_ARCHSPECIFIC.
//
// Subspaces are aligned to a power of two so opcodes can be masked
// with AMask and used as compact array indices.
const (
	ABase386 = (1 + iota) << 10
	ABaseARM
	ABaseAMD64
	ABasePPC64
	ABaseARM64
	ABaseMIPS
	ABaseS390X

	AllowedOpCodes = 1 << 10            // The number of opcodes available for any given architecture.
	AMask          = AllowedOpCodes - 1 // AND with this to use the opcode as an array index.
)

// An LSym is the sort of symbol that is written to an object file.
type LSym struct {
	Name    string
	Type    SymKind
	Version int16
	Attribute

	RefIdx int // Index of this symbol in the symbol reference list.
	Args   int32
	Locals int32
	Size   int64
	Gotype *LSym
	Autom  *Auto
	Text   *Prog
	Pcln   *Pcln
	P      []byte
	R      []Reloc
}

// Attribute is a set of symbol attributes.
type Attribute int16

const (
	AttrDuplicateOK Attribute = 1 << iota
	AttrCFunc
	AttrNoSplit
	AttrLeaf
	AttrSeenGlobl
	AttrOnList

	// MakeTypelink means that the type should have an entry in the typelink table.
	AttrMakeTypelink

	// ReflectMethod means the function may call reflect.Type.Method or
	// reflect.Type.MethodByName. Matching is imprecise (as reflect.Type
	// can be used through a custom interface), so ReflectMethod may be
	// set in some cases when the reflect package is not called.
	//
	// Used by the linker to determine what methods can be pruned.
	AttrReflectMethod

	// Local means make the symbol local even when compiling Go code to reference Go
	// symbols in other shared libraries, as in this mode symbols are global by
	// default. "local" here means in the sense of the dynamic linker, i.e. not
	// visible outside of the module (shared library or executable) that contains its
	// definition. (When not compiling to support Go shared libraries, all symbols are
	// local in this sense unless there is a cgo_export_* directive).
	AttrLocal
)

func (a Attribute) DuplicateOK() bool   { return a&AttrDuplicateOK != 0 }
func (a Attribute) MakeTypelink() bool  { return a&AttrMakeTypelink != 0 }
func (a Attribute) CFunc() bool         { return a&AttrCFunc != 0 }
func (a Attribute) NoSplit() bool       { return a&AttrNoSplit != 0 }
func (a Attribute) Leaf() bool          { return a&AttrLeaf != 0 }
func (a Attribute) SeenGlobl() bool     { return a&AttrSeenGlobl != 0 }
func (a Attribute) OnList() bool        { return a&AttrOnList != 0 }
func (a Attribute) ReflectMethod() bool { return a&AttrReflectMethod != 0 }
func (a Attribute) Local() bool         { return a&AttrLocal != 0 }

func (a *Attribute) Set(flag Attribute, value bool) {
	if value {
		*a |= flag
	} else {
		*a &^= flag
	}
}

// The compiler needs LSym to satisfy fmt.Stringer, because it stores
// an LSym in ssa.ExternSymbol.
func (s *LSym) String() string {
	return s.Name
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

// A SymKind describes the kind of memory represented by a symbol.
type SymKind int16

// Defined SymKind values.
//
// TODO(rsc): Give idiomatic Go names.
// TODO(rsc): Reduce the number of symbol types in the object files.
//go:generate stringer -type=SymKind
const (
	Sxxx SymKind = iota
	STEXT
	SELFRXSECT

	// Read-only sections.
	STYPE
	SSTRING
	SGOSTRING
	SGOFUNC
	SGCBITS
	SRODATA
	SFUNCTAB

	SELFROSECT
	SMACHOPLT

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

	// Part of .data.rel.ro if it exists, otherwise part of .rodata.
	STYPELINK
	SITABLINK
	SSYMTAB
	SPCLNTAB

	// Writable sections.
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
	SDWARFSECT
	SDWARFINFO
	SSUB       = SymKind(1 << 8)
	SMASK      = SymKind(SSUB - 1)
	SHIDDEN    = SymKind(1 << 9)
	SCONTAINER = SymKind(1 << 10) // has a sub-symbol
)

// ReadOnly are the symbol kinds that form read-only sections. In some
// cases, if they will require relocations, they are transformed into
// rel-ro sections using RelROMap.
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

type Reloc struct {
	Off  int32
	Siz  uint8
	Type RelocType
	Add  int64
	Sym  *LSym
}

type RelocType int32

//go:generate stringer -type=RelocType
const (
	R_ADDR RelocType = 1 + iota
	// R_ADDRPOWER relocates a pair of "D-form" instructions (instructions with 16-bit
	// immediates in the low half of the instruction word), usually addis followed by
	// another add or a load, inserting the "high adjusted" 16 bits of the address of
	// the referenced symbol into the immediate field of the first instruction and the
	// low 16 bits into that of the second instruction.
	R_ADDRPOWER
	// R_ADDRARM64 relocates an adrp, add pair to compute the address of the
	// referenced symbol.
	R_ADDRARM64
	// R_ADDRMIPS (only used on mips/mips64) resolves to the low 16 bits of an external
	// address, by encoding it into the instruction.
	R_ADDRMIPS
	// R_ADDROFF resolves to a 32-bit offset from the beginning of the section
	// holding the data being relocated to the referenced symbol.
	R_ADDROFF
	// R_WEAKADDROFF resolves just like R_ADDROFF but is a weak relocation.
	// A weak relocation does not make the symbol it refers to reachable,
	// and is only honored by the linker if the symbol is in some other way
	// reachable.
	R_WEAKADDROFF
	R_SIZE
	R_CALL
	R_CALLARM
	R_CALLARM64
	R_CALLIND
	R_CALLPOWER
	// R_CALLMIPS (only used on mips64) resolves to non-PC-relative target address
	// of a CALL (JAL) instruction, by encoding the address into the instruction.
	R_CALLMIPS
	R_CONST
	R_PCREL
	// R_TLS_LE, used on 386, amd64, and ARM, resolves to the offset of the
	// thread-local symbol from the thread local base and is used to implement the
	// "local exec" model for tls access (r.Sym is not set on intel platforms but is
	// set to a TLS symbol -- runtime.tlsg -- in the linker when externally linking).
	R_TLS_LE
	// R_TLS_IE, used 386, amd64, and ARM resolves to the PC-relative offset to a GOT
	// slot containing the offset from the thread-local symbol from the thread local
	// base and is used to implemented the "initial exec" model for tls access (r.Sym
	// is not set on intel platforms but is set to a TLS symbol -- runtime.tlsg -- in
	// the linker when externally linking).
	R_TLS_IE
	R_GOTOFF
	R_PLT0
	R_PLT1
	R_PLT2
	R_USEFIELD
	// R_USETYPE resolves to an *rtype, but no relocation is created. The
	// linker uses this as a signal that the pointed-to type information
	// should be linked into the final binary, even if there are no other
	// direct references. (This is used for types reachable by reflection.)
	R_USETYPE
	// R_METHODOFF resolves to a 32-bit offset from the beginning of the section
	// holding the data being relocated to the referenced symbol.
	// It is a variant of R_ADDROFF used when linking from the uncommonType of a
	// *rtype, and may be set to zero by the linker if it determines the method
	// text is unreachable by the linked program.
	R_METHODOFF
	R_POWER_TOC
	R_GOTPCREL
	// R_JMPMIPS (only used on mips64) resolves to non-PC-relative target address
	// of a JMP instruction, by encoding the address into the instruction.
	// The stack nosplit check ignores this since it is not a function call.
	R_JMPMIPS
	// R_DWARFREF resolves to the offset of the symbol from its section.
	R_DWARFREF

	// Platform dependent relocations. Architectures with fixed width instructions
	// have the inherent issue that a 32-bit (or 64-bit!) displacement cannot be
	// stuffed into a 32-bit instruction, so an address needs to be spread across
	// several instructions, and in turn this requires a sequence of relocations, each
	// updating a part of an instruction. This leads to relocation codes that are
	// inherently processor specific.

	// Arm64.

	// Set a MOV[NZ] immediate field to bits [15:0] of the offset from the thread
	// local base to the thread local variable defined by the referenced (thread
	// local) symbol. Error if the offset does not fit into 16 bits.
	R_ARM64_TLS_LE

	// Relocates an ADRP; LD64 instruction sequence to load the offset between
	// the thread local base and the thread local variable defined by the
	// referenced (thread local) symbol from the GOT.
	R_ARM64_TLS_IE

	// R_ARM64_GOTPCREL relocates an adrp, ld64 pair to compute the address of the GOT
	// slot of the referenced symbol.
	R_ARM64_GOTPCREL

	// PPC64.

	// R_POWER_TLS_LE is used to implement the "local exec" model for tls
	// access. It resolves to the offset of the thread-local symbol from the
	// thread pointer (R13) and inserts this value into the low 16 bits of an
	// instruction word.
	R_POWER_TLS_LE

	// R_POWER_TLS_IE is used to implement the "initial exec" model for tls access. It
	// relocates a D-form, DS-form instruction sequence like R_ADDRPOWER_DS. It
	// inserts to the offset of GOT slot for the thread-local symbol from the TOC (the
	// GOT slot is filled by the dynamic linker with the offset of the thread-local
	// symbol from the thread pointer (R13)).
	R_POWER_TLS_IE

	// R_POWER_TLS marks an X-form instruction such as "MOVD 0(R13)(R31*1), g" as
	// accessing a particular thread-local symbol. It does not affect code generation
	// but is used by the system linker when relaxing "initial exec" model code to
	// "local exec" model code.
	R_POWER_TLS

	// R_ADDRPOWER_DS is similar to R_ADDRPOWER above, but assumes the second
	// instruction is a "DS-form" instruction, which has an immediate field occupying
	// bits [15:2] of the instruction word. Bits [15:2] of the address of the
	// relocated symbol are inserted into this field; it is an error if the last two
	// bits of the address are not 0.
	R_ADDRPOWER_DS

	// R_ADDRPOWER_PCREL relocates a D-form, DS-form instruction sequence like
	// R_ADDRPOWER_DS but inserts the offset of the GOT slot for the referenced symbol
	// from the TOC rather than the symbol's address.
	R_ADDRPOWER_GOT

	// R_ADDRPOWER_PCREL relocates two D-form instructions like R_ADDRPOWER, but
	// inserts the displacement from the place being relocated to the address of the
	// the relocated symbol instead of just its address.
	R_ADDRPOWER_PCREL

	// R_ADDRPOWER_TOCREL relocates two D-form instructions like R_ADDRPOWER, but
	// inserts the offset from the TOC to the address of the the relocated symbol
	// rather than the symbol's address.
	R_ADDRPOWER_TOCREL

	// R_ADDRPOWER_TOCREL relocates a D-form, DS-form instruction sequence like
	// R_ADDRPOWER_DS but inserts the offset from the TOC to the address of the the
	// relocated symbol rather than the symbol's address.
	R_ADDRPOWER_TOCREL_DS

	// R_PCRELDBL relocates s390x 2-byte aligned PC-relative addresses.
	// TODO(mundaym): remove once variants can be serialized - see issue 14218.
	R_PCRELDBL

	// R_ADDRMIPSU (only used on mips/mips64) resolves to the sign-adjusted "upper" 16
	// bits (bit 16-31) of an external address, by encoding it into the instruction.
	R_ADDRMIPSU
	// R_ADDRMIPSTLS (only used on mips64) resolves to the low 16 bits of a TLS
	// address (offset from thread pointer), by encoding it into the instruction.
	R_ADDRMIPSTLS
)

// IsDirectJump returns whether r is a relocation for a direct jump.
// A direct jump is a CALL or JMP instruction that takes the target address
// as immediate. The address is embedded into the instruction, possibly
// with limited width.
// An indirect jump is a CALL or JMP instruction that takes the target address
// in register or memory.
func (r RelocType) IsDirectJump() bool {
	switch r {
	case R_CALL, R_CALLARM, R_CALLARM64, R_CALLPOWER, R_CALLMIPS, R_JMPMIPS:
		return true
	}
	return false
}

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

// symbol version, incremented each time a file is loaded.
// version==1 is reserved for savehist.
const (
	HistVersion = 1
)

// Link holds the context for writing object code from a compiler
// to be linker input or for reading that input into the linker.
type Link struct {
	Headtype      HeadType
	Arch          *LinkArch
	Debugasm      int32
	Debugvlog     int32
	Debugdivmod   int32
	Debugpcln     int32
	Flag_shared   bool
	Flag_dynlink  bool
	Flag_optimize bool
	Bso           *bufio.Writer
	Pathname      string
	Hash          map[SymVer]*LSym
	LineHist      LineHist
	Imports       []string
	Plists        []*Plist
	Sym_div       *LSym
	Sym_divu      *LSym
	Sym_mod       *LSym
	Sym_modu      *LSym
	Plan9privates *LSym
	Curp          *Prog
	Printp        *Prog
	Blitrl        *Prog
	Elitrl        *Prog
	Rexflag       int
	Vexflag       int
	Rep           int
	Repn          int
	Lock          int
	Asmode        int
	AsmBuf        AsmBuf // instruction buffer for x86
	Instoffset    int64
	Autosize      int32
	Armsize       int32
	Pc            int64
	DiagFunc      func(string, ...interface{})
	Mode          int
	Cursym        *LSym
	Version       int
	Errors        int

	Framepointer_enabled bool

	// state for writing objects
	Text []*LSym
	Data []*LSym

	// Cache of Progs
	allocIdx int
	progs    [10000]Prog
}

func (ctxt *Link) Diag(format string, args ...interface{}) {
	ctxt.Errors++
	ctxt.DiagFunc(format, args...)
}

func (ctxt *Link) Logf(format string, args ...interface{}) {
	fmt.Fprintf(ctxt.Bso, format, args...)
	ctxt.Bso.Flush()
}

// The smallest possible offset from the hardware stack pointer to a local
// variable on the stack. Architectures that use a link register save its value
// on the stack in the function prologue and so always have a pointer between
// the hardware stack pointer and the local variable area.
func (ctxt *Link) FixedFrameSize() int64 {
	switch ctxt.Arch.Family {
	case sys.AMD64, sys.I386:
		return 0
	case sys.PPC64:
		// PIC code on ppc64le requires 32 bytes of stack, and it's easier to
		// just use that much stack always on ppc64x.
		return int64(4 * ctxt.Arch.PtrSize)
	default:
		return int64(ctxt.Arch.PtrSize)
	}
}

type SymVer struct {
	Name    string
	Version int // TODO: make int16 to match LSym.Version?
}

// LinkArch is the definition of a single architecture.
type LinkArch struct {
	*sys.Arch
	Preprocess func(*Link, *LSym)
	Assemble   func(*Link, *LSym)
	Follow     func(*Link, *LSym)
	Progedit   func(*Link, *Prog)
	UnaryDst   map[As]bool // Instruction takes one operand, a destination.
}

// HeadType is the executable header type.
type HeadType uint8

const (
	Hunknown HeadType = iota
	Hdarwin
	Hdragonfly
	Hfreebsd
	Hlinux
	Hnacl
	Hnetbsd
	Hopenbsd
	Hplan9
	Hsolaris
	Hwindows
	Hwindowsgui
)

func (h *HeadType) Set(s string) error {
	switch s {
	case "darwin":
		*h = Hdarwin
	case "dragonfly":
		*h = Hdragonfly
	case "freebsd":
		*h = Hfreebsd
	case "linux", "android":
		*h = Hlinux
	case "nacl":
		*h = Hnacl
	case "netbsd":
		*h = Hnetbsd
	case "openbsd":
		*h = Hopenbsd
	case "plan9":
		*h = Hplan9
	case "solaris":
		*h = Hsolaris
	case "windows":
		*h = Hwindows
	case "windowsgui":
		*h = Hwindowsgui
	default:
		return fmt.Errorf("invalid headtype: %q", s)
	}
	return nil
}

func (h *HeadType) String() string {
	switch *h {
	case Hdarwin:
		return "darwin"
	case Hdragonfly:
		return "dragonfly"
	case Hfreebsd:
		return "freebsd"
	case Hlinux:
		return "linux"
	case Hnacl:
		return "nacl"
	case Hnetbsd:
		return "netbsd"
	case Hopenbsd:
		return "openbsd"
	case Hplan9:
		return "plan9"
	case Hsolaris:
		return "solaris"
	case Hwindows:
		return "windows"
	case Hwindowsgui:
		return "windowsgui"
	}
	return fmt.Sprintf("HeadType(%d)", *h)
}

// AsmBuf is a simple buffer to assemble variable-length x86 instructions into.
type AsmBuf struct {
	buf [100]byte
	off int
}

// Put1 appends one byte to the end of the buffer.
func (a *AsmBuf) Put1(x byte) {
	a.buf[a.off] = x
	a.off++
}

// Put2 appends two bytes to the end of the buffer.
func (a *AsmBuf) Put2(x, y byte) {
	a.buf[a.off+0] = x
	a.buf[a.off+1] = y
	a.off += 2
}

// Put3 appends three bytes to the end of the buffer.
func (a *AsmBuf) Put3(x, y, z byte) {
	a.buf[a.off+0] = x
	a.buf[a.off+1] = y
	a.buf[a.off+2] = z
	a.off += 3
}

// Put4 appends four bytes to the end of the buffer.
func (a *AsmBuf) Put4(x, y, z, w byte) {
	a.buf[a.off+0] = x
	a.buf[a.off+1] = y
	a.buf[a.off+2] = z
	a.buf[a.off+3] = w
	a.off += 4
}

// PutInt16 writes v into the buffer using little-endian encoding.
func (a *AsmBuf) PutInt16(v int16) {
	a.buf[a.off+0] = byte(v)
	a.buf[a.off+1] = byte(v >> 8)
	a.off += 2
}

// PutInt32 writes v into the buffer using little-endian encoding.
func (a *AsmBuf) PutInt32(v int32) {
	a.buf[a.off+0] = byte(v)
	a.buf[a.off+1] = byte(v >> 8)
	a.buf[a.off+2] = byte(v >> 16)
	a.buf[a.off+3] = byte(v >> 24)
	a.off += 4
}

// PutInt64 writes v into the buffer using little-endian encoding.
func (a *AsmBuf) PutInt64(v int64) {
	a.buf[a.off+0] = byte(v)
	a.buf[a.off+1] = byte(v >> 8)
	a.buf[a.off+2] = byte(v >> 16)
	a.buf[a.off+3] = byte(v >> 24)
	a.buf[a.off+4] = byte(v >> 32)
	a.buf[a.off+5] = byte(v >> 40)
	a.buf[a.off+6] = byte(v >> 48)
	a.buf[a.off+7] = byte(v >> 56)
	a.off += 8
}

// Put copies b into the buffer.
func (a *AsmBuf) Put(b []byte) {
	copy(a.buf[a.off:], b)
	a.off += len(b)
}

// Insert inserts b at offset i.
func (a *AsmBuf) Insert(i int, b byte) {
	a.off++
	copy(a.buf[i+1:a.off], a.buf[i:a.off-1])
	a.buf[i] = b
}

// Last returns the byte at the end of the buffer.
func (a *AsmBuf) Last() byte { return a.buf[a.off-1] }

// Len returns the length of the buffer.
func (a *AsmBuf) Len() int { return a.off }

// Bytes returns the contents of the buffer.
func (a *AsmBuf) Bytes() []byte { return a.buf[:a.off] }

// Reset empties the buffer.
func (a *AsmBuf) Reset() { a.off = 0 }

// Peek returns the byte at offset i.
func (a *AsmBuf) Peek(i int) byte { return a.buf[i] }
