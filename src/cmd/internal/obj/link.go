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
	"cmd/internal/dwarf"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"fmt"
	"sync"
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
	Name   AddrName
	Class  int8
	Offset int64
	Sym    *LSym

	// argument value:
	//	for TYPE_SCONST, a string
	//	for TYPE_FCONST, a float64
	//	for TYPE_BRANCH, a *Prog (optional)
	//	for TYPE_TEXTSIZE, an int32 (optional)
	Val interface{}
}

type AddrName int8

const (
	NAME_NONE AddrName = iota
	NAME_EXTERN
	NAME_STATIC
	NAME_AUTO
	NAME_PARAM
	// A reference to name@GOT(SB) is a reference to the entry in the global offset
	// table for 'name'.
	NAME_GOTREF
)

type AddrType uint8

const (
	TYPE_NONE AddrType = iota
	TYPE_BRANCH
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
// specified by Pos.Line().
// Every Prog has a Ctxt field that defines its context.
// For performance reasons, Progs usually are usually bulk allocated, cached, and reused;
// those bulk allocators should always be used, rather than new(Prog).
//
// The other fields not yet mentioned are for use by the back ends and should
// be left zeroed by creators of Prog lists.
type Prog struct {
	Ctxt   *Link    // linker context
	Link   *Prog    // next Prog in linked list
	From   Addr     // first source operand
	From3  *Addr    // third source operand (second is Reg below)
	To     Addr     // destination operand (second is RegTo2 below)
	Pcond  *Prog    // target of conditional jump
	Forwd  *Prog    // for x86 back end
	Rel    *Prog    // for x86, arm back ends
	Pc     int64    // for back ends or assembler: virtual or actual program counter, depending on phase
	Pos    src.XPos // source position of this instruction
	Spadj  int32    // effect of instruction on stack pointer (increment or decrement amount)
	As     As       // assembler opcode
	Reg    int16    // 2nd source operand
	RegTo2 int16    // 2nd destination operand
	Mark   uint16   // bitmask of arch-specific items
	Optab  uint16   // arch-specific opcode index
	Scond  uint8    // condition bits for conditional instruction (e.g., on ARM)
	Back   uint8    // for x86 back end: backwards branch state
	Ft     uint8    // for x86 back end: type index of Prog.From
	Tt     uint8    // for x86 back end: type index of Prog.To
	Isize  uint8    // for x86 back end: size of the instruction in bytes
}

// From3Type returns From3.Type, or TYPE_NONE when From3 is nil.
func (p *Prog) From3Type() AddrType {
	if p.From3 == nil {
		return TYPE_NONE
	}
	return p.From3.Type
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
	AUNDEF
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
	Name string
	Type objabi.SymKind
	Attribute

	RefIdx int // Index of this symbol in the symbol reference list.
	Size   int64
	Gotype *LSym
	P      []byte
	R      []Reloc

	Func *FuncInfo
}

// A FuncInfo contains extra fields for STEXT symbols.
type FuncInfo struct {
	Args   int32
	Locals int32
	Text   *Prog
	Autom  []*Auto
	Pcln   Pcln

	dwarfSym       *LSym
	dwarfRangesSym *LSym

	GCArgs   LSym
	GCLocals LSym
}

// Attribute is a set of symbol attributes.
type Attribute int16

const (
	AttrDuplicateOK Attribute = 1 << iota
	AttrCFunc
	AttrNoSplit
	AttrLeaf
	AttrWrapper
	AttrNeedCtxt
	AttrNoFrame
	AttrSeenGlobl
	AttrOnList
	AttrStatic

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
func (a Attribute) Wrapper() bool       { return a&AttrWrapper != 0 }
func (a Attribute) NeedCtxt() bool      { return a&AttrNeedCtxt != 0 }
func (a Attribute) NoFrame() bool       { return a&AttrNoFrame != 0 }
func (a Attribute) Static() bool        { return a&AttrStatic != 0 }

func (a *Attribute) Set(flag Attribute, value bool) {
	if value {
		*a |= flag
	} else {
		*a &^= flag
	}
}

var textAttrStrings = [...]struct {
	bit Attribute
	s   string
}{
	{bit: AttrDuplicateOK, s: "DUPOK"},
	{bit: AttrMakeTypelink, s: ""},
	{bit: AttrCFunc, s: "CFUNC"},
	{bit: AttrNoSplit, s: "NOSPLIT"},
	{bit: AttrLeaf, s: "LEAF"},
	{bit: AttrSeenGlobl, s: ""},
	{bit: AttrOnList, s: ""},
	{bit: AttrReflectMethod, s: "REFLECTMETHOD"},
	{bit: AttrLocal, s: "LOCAL"},
	{bit: AttrWrapper, s: "WRAPPER"},
	{bit: AttrNeedCtxt, s: "NEEDCTXT"},
	{bit: AttrNoFrame, s: "NOFRAME"},
	{bit: AttrStatic, s: "STATIC"},
}

// TextAttrString formats a for printing in as part of a TEXT prog.
func (a Attribute) TextAttrString() string {
	var s string
	for _, x := range textAttrStrings {
		if a&x.bit != 0 {
			if x.s != "" {
				s += x.s + "|"
			}
			a &^= x.bit
		}
	}
	if a != 0 {
		s += fmt.Sprintf("UnknownAttribute(%d)|", a)
	}
	// Chop off trailing |, if present.
	if len(s) > 0 {
		s = s[:len(s)-1]
	}
	return s
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
	Pcinline    Pcdata
	Pcdata      []Pcdata
	Funcdata    []*LSym
	Funcdataoff []int64
	File        []string
	Lastfile    string
	Lastindex   int
	InlTree     InlTree // per-function inlining tree extracted from the global tree
}

type Reloc struct {
	Off  int32
	Siz  uint8
	Type objabi.RelocType
	Add  int64
	Sym  *LSym
}

type Auto struct {
	Asym    *LSym
	Aoffset int32
	Name    AddrName
	Gotype  *LSym
}

type Pcdata struct {
	P []byte
}

// Link holds the context for writing object code from a compiler
// to be linker input or for reading that input into the linker.
type Link struct {
	Headtype      objabi.HeadType
	Arch          *LinkArch
	Debugasm      bool
	Debugvlog     bool
	Debugpcln     string
	Flag_shared   bool
	Flag_dynlink  bool
	Flag_optimize bool
	Bso           *bufio.Writer
	Pathname      string
	hashmu        sync.Mutex       // protects hash
	hash          map[string]*LSym // name -> sym mapping
	statichash    map[string]*LSym // name -> sym mapping for static syms
	PosTable      src.PosTable
	InlTree       InlTree // global inlining tree used by gc/inl.go
	Imports       []string
	DiagFunc      func(string, ...interface{})
	DebugInfo     func(fn *LSym, curfn interface{}) []dwarf.Scope // if non-nil, curfn is a *gc.Node
	Errors        int

	Framepointer_enabled bool

	// state for writing objects
	Text []*LSym
	Data []*LSym
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

// LinkArch is the definition of a single architecture.
type LinkArch struct {
	*sys.Arch
	Init       func(*Link)
	Preprocess func(*Link, *LSym, ProgAlloc)
	Assemble   func(*Link, *LSym, ProgAlloc)
	Progedit   func(*Link, *Prog, ProgAlloc)
	UnaryDst   map[As]bool // Instruction takes one operand, a destination.
}
