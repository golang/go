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

package obj

import (
	"bufio"
	"cmd/internal/dwarf"
	"cmd/internal/goobj"
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
//		Register list for ARM, ARM64, 386/AMD64.
//		Encoding:
//			type = TYPE_REGLIST
//		On ARM:
//			offset = bit mask of registers in list; R0 is low bit.
//		On ARM64:
//			offset = register count (Q:size) | arrangement (opcode) | first register
//		On 386/AMD64:
//			reg = range low register
//			offset = 2 packed registers + kind tag (see x86.EncodeRegisterRange)
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
//	reg.[US]XT[BHWX]
//		Register extension for ARM64
//		Encoding:
//			type = TYPE_REG
//			reg = REG_[US]XT[BHWX] + register + shift amount
//			offset = ((reg&31) << 16) | (exttype << 13) | (amount<<10)
//
//	reg.<T>
//		Register arrangement for ARM64 SIMD register
//		e.g.: V1.S4, V2.S2, V7.D2, V2.H4, V6.B16
//		Encoding:
//			type = TYPE_REG
//			reg = REG_ARNG + register + arrangement
//
//	reg.<T>[index]
//		Register element for ARM64
//		Encoding:
//			type = TYPE_REG
//			reg = REG_ELEM + register + arrangement
//			index = element index

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
	// Indicates that this is a reference to a TOC anchor.
	NAME_TOCREF
)

//go:generate stringer -type AddrType

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

func (a *Addr) Target() *Prog {
	if a.Type == TYPE_BRANCH && a.Val != nil {
		return a.Val.(*Prog)
	}
	return nil
}
func (a *Addr) SetTarget(t *Prog) {
	if a.Type != TYPE_BRANCH {
		panic("setting branch target when type is not TYPE_BRANCH")
	}
	a.Val = t
}

// Prog describes a single machine instruction.
//
// The general instruction form is:
//
//	(1) As.Scond From [, ...RestArgs], To
//	(2) As.Scond From, Reg [, ...RestArgs], To, RegTo2
//
// where As is an opcode and the others are arguments:
// From, Reg are sources, and To, RegTo2 are destinations.
// RestArgs can hold additional sources and destinations.
// Usually, not all arguments are present.
// For example, MOVL R1, R2 encodes using only As=MOVL, From=R1, To=R2.
// The Scond field holds additional condition bits for systems (like arm)
// that have generalized conditional execution.
// (2) form is present for compatibility with older code,
// to avoid too much changes in a single swing.
// (1) scheme is enough to express any kind of operand combination.
//
// Jump instructions use the To.Val field to point to the target *Prog,
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
	Ctxt     *Link     // linker context
	Link     *Prog     // next Prog in linked list
	From     Addr      // first source operand
	RestArgs []AddrPos // can pack any operands that not fit into {Prog.From, Prog.To}
	To       Addr      // destination operand (second is RegTo2 below)
	Pool     *Prog     // constant pool entry, for arm,arm64 back ends
	Forwd    *Prog     // for x86 back end
	Rel      *Prog     // for x86, arm back ends
	Pc       int64     // for back ends or assembler: virtual or actual program counter, depending on phase
	Pos      src.XPos  // source position of this instruction
	Spadj    int32     // effect of instruction on stack pointer (increment or decrement amount)
	As       As        // assembler opcode
	Reg      int16     // 2nd source operand
	RegTo2   int16     // 2nd destination operand
	Mark     uint16    // bitmask of arch-specific items
	Optab    uint16    // arch-specific opcode index
	Scond    uint8     // bits that describe instruction suffixes (e.g. ARM conditions)
	Back     uint8     // for x86 back end: backwards branch state
	Ft       uint8     // for x86 back end: type index of Prog.From
	Tt       uint8     // for x86 back end: type index of Prog.To
	Isize    uint8     // for x86 back end: size of the instruction in bytes
}

// Pos indicates whether the oprand is the source or the destination.
type AddrPos struct {
	Addr
	Pos OperandPos
}

type OperandPos int8

const (
	Source OperandPos = iota
	Destination
)

// From3Type returns p.GetFrom3().Type, or TYPE_NONE when
// p.GetFrom3() returns nil.
//
// Deprecated: for the same reasons as Prog.GetFrom3.
func (p *Prog) From3Type() AddrType {
	if p.RestArgs == nil {
		return TYPE_NONE
	}
	return p.RestArgs[0].Type
}

// GetFrom3 returns second source operand (the first is Prog.From).
// In combination with Prog.From and Prog.To it makes common 3 operand
// case easier to use.
//
// Should be used only when RestArgs is set with SetFrom3.
//
// Deprecated: better use RestArgs directly or define backend-specific getters.
// Introduced to simplify transition to []Addr.
// Usage of this is discouraged due to fragility and lack of guarantees.
func (p *Prog) GetFrom3() *Addr {
	if p.RestArgs == nil {
		return nil
	}
	return &p.RestArgs[0].Addr
}

// SetFrom3 assigns []Args{{a, 0}} to p.RestArgs.
// In pair with Prog.GetFrom3 it can help in emulation of Prog.From3.
//
// Deprecated: for the same reasons as Prog.GetFrom3.
func (p *Prog) SetFrom3(a Addr) {
	p.RestArgs = []AddrPos{{a, Source}}
}

// SetTo2 assings []Args{{a, 1}} to p.RestArgs when the second destination
// operand does not fit into prog.RegTo2.
func (p *Prog) SetTo2(a Addr) {
	p.RestArgs = []AddrPos{{a, Destination}}
}

// GetTo2 returns the second destination operand.
func (p *Prog) GetTo2() *Addr {
	if p.RestArgs == nil {
		return nil
	}
	return &p.RestArgs[0].Addr
}

// SetRestArgs assigns more than one source operands to p.RestArgs.
func (p *Prog) SetRestArgs(args []Addr) {
	for i := range args {
		p.RestArgs = append(p.RestArgs, AddrPos{args[i], Source})
	}
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
	APCALIGN
	APCDATA
	ARET
	AGETCALLERPC
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
	ABase386 = (1 + iota) << 11
	ABaseARM
	ABaseAMD64
	ABasePPC64
	ABaseARM64
	ABaseMIPS
	ABaseRISCV
	ABaseS390X
	ABaseWasm

	AllowedOpCodes = 1 << 11            // The number of opcodes available for any given architecture.
	AMask          = AllowedOpCodes - 1 // AND with this to use the opcode as an array index.
)

// An LSym is the sort of symbol that is written to an object file.
// It represents Go symbols in a flat pkg+"."+name namespace.
type LSym struct {
	Name string
	Type objabi.SymKind
	Attribute

	Size   int64
	Gotype *LSym
	P      []byte
	R      []Reloc

	Extra *interface{} // *FuncInfo or *FileInfo, if present

	Pkg    string
	PkgIdx int32
	SymIdx int32
}

// A FuncInfo contains extra fields for STEXT symbols.
type FuncInfo struct {
	Args     int32
	Locals   int32
	Align    int32
	FuncID   objabi.FuncID
	Text     *Prog
	Autot    map[*LSym]struct{}
	Pcln     Pcln
	InlMarks []InlMark

	dwarfInfoSym       *LSym
	dwarfLocSym        *LSym
	dwarfRangesSym     *LSym
	dwarfAbsFnSym      *LSym
	dwarfDebugLinesSym *LSym

	GCArgs             *LSym
	GCLocals           *LSym
	StackObjects       *LSym
	OpenCodedDeferInfo *LSym

	FuncInfoSym *LSym
}

// NewFuncInfo allocates and returns a FuncInfo for LSym.
func (s *LSym) NewFuncInfo() *FuncInfo {
	if s.Extra != nil {
		panic(fmt.Sprintf("invalid use of LSym - NewFuncInfo with Extra of type %T", *s.Extra))
	}
	f := new(FuncInfo)
	s.Extra = new(interface{})
	*s.Extra = f
	return f
}

// Func returns the *FuncInfo associated with s, or else nil.
func (s *LSym) Func() *FuncInfo {
	if s.Extra == nil {
		return nil
	}
	f, _ := (*s.Extra).(*FuncInfo)
	return f
}

// A FileInfo contains extra fields for SDATA symbols backed by files.
// (If LSym.Extra is a *FileInfo, LSym.P == nil.)
type FileInfo struct {
	Name string // name of file to read into object file
	Size int64  // length of file
}

// NewFileInfo allocates and returns a FileInfo for LSym.
func (s *LSym) NewFileInfo() *FileInfo {
	if s.Extra != nil {
		panic(fmt.Sprintf("invalid use of LSym - NewFileInfo with Extra of type %T", *s.Extra))
	}
	f := new(FileInfo)
	s.Extra = new(interface{})
	*s.Extra = f
	return f
}

// File returns the *FileInfo associated with s, or else nil.
func (s *LSym) File() *FileInfo {
	if s.Extra == nil {
		return nil
	}
	f, _ := (*s.Extra).(*FileInfo)
	return f
}

type InlMark struct {
	// When unwinding from an instruction in an inlined body, mark
	// where we should unwind to.
	// id records the global inlining id of the inlined body.
	// p records the location of an instruction in the parent (inliner) frame.
	p  *Prog
	id int32
}

// Mark p as the instruction to set as the pc when
// "unwinding" the inlining global frame id. Usually it should be
// instruction with a file:line at the callsite, and occur
// just before the body of the inlined function.
func (fi *FuncInfo) AddInlMark(p *Prog, id int32) {
	fi.InlMarks = append(fi.InlMarks, InlMark{p: p, id: id})
}

// Record the type symbol for an auto variable so that the linker
// an emit DWARF type information for the type.
func (fi *FuncInfo) RecordAutoType(gotype *LSym) {
	if fi.Autot == nil {
		fi.Autot = make(map[*LSym]struct{})
	}
	fi.Autot[gotype] = struct{}{}
}

//go:generate stringer -type ABI

// ABI is the calling convention of a text symbol.
type ABI uint8

const (
	// ABI0 is the stable stack-based ABI. It's important that the
	// value of this is "0": we can't distinguish between
	// references to data and ABI0 text symbols in assembly code,
	// and hence this doesn't distinguish between symbols without
	// an ABI and text symbols with ABI0.
	ABI0 ABI = iota

	// ABIInternal is the internal ABI that may change between Go
	// versions. All Go functions use the internal ABI and the
	// compiler generates wrappers for calls to and from other
	// ABIs.
	ABIInternal

	ABICount
)

// ParseABI converts from a string representation in 'abistr' to the
// corresponding ABI value. Second return value is TRUE if the
// abi string is recognized, FALSE otherwise.
func ParseABI(abistr string) (ABI, bool) {
	switch abistr {
	default:
		return ABI0, false
	case "ABI0":
		return ABI0, true
	case "ABIInternal":
		return ABIInternal, true
	}
}

// Attribute is a set of symbol attributes.
type Attribute uint32

const (
	AttrDuplicateOK Attribute = 1 << iota
	AttrCFunc
	AttrNoSplit
	AttrLeaf
	AttrWrapper
	AttrNeedCtxt
	AttrNoFrame
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

	// For function symbols; indicates that the specified function was the
	// target of an inline during compilation
	AttrWasInlined

	// TopFrame means that this function is an entry point and unwinders should not
	// keep unwinding beyond this frame.
	AttrTopFrame

	// Indexed indicates this symbol has been assigned with an index (when using the
	// new object file format).
	AttrIndexed

	// Only applied on type descriptor symbols, UsedInIface indicates this type is
	// converted to an interface.
	//
	// Used by the linker to determine what methods can be pruned.
	AttrUsedInIface

	// ContentAddressable indicates this is a content-addressable symbol.
	AttrContentAddressable

	// attrABIBase is the value at which the ABI is encoded in
	// Attribute. This must be last; all bits after this are
	// assumed to be an ABI value.
	//
	// MUST BE LAST since all bits above this comprise the ABI.
	attrABIBase
)

func (a Attribute) DuplicateOK() bool        { return a&AttrDuplicateOK != 0 }
func (a Attribute) MakeTypelink() bool       { return a&AttrMakeTypelink != 0 }
func (a Attribute) CFunc() bool              { return a&AttrCFunc != 0 }
func (a Attribute) NoSplit() bool            { return a&AttrNoSplit != 0 }
func (a Attribute) Leaf() bool               { return a&AttrLeaf != 0 }
func (a Attribute) OnList() bool             { return a&AttrOnList != 0 }
func (a Attribute) ReflectMethod() bool      { return a&AttrReflectMethod != 0 }
func (a Attribute) Local() bool              { return a&AttrLocal != 0 }
func (a Attribute) Wrapper() bool            { return a&AttrWrapper != 0 }
func (a Attribute) NeedCtxt() bool           { return a&AttrNeedCtxt != 0 }
func (a Attribute) NoFrame() bool            { return a&AttrNoFrame != 0 }
func (a Attribute) Static() bool             { return a&AttrStatic != 0 }
func (a Attribute) WasInlined() bool         { return a&AttrWasInlined != 0 }
func (a Attribute) TopFrame() bool           { return a&AttrTopFrame != 0 }
func (a Attribute) Indexed() bool            { return a&AttrIndexed != 0 }
func (a Attribute) UsedInIface() bool        { return a&AttrUsedInIface != 0 }
func (a Attribute) ContentAddressable() bool { return a&AttrContentAddressable != 0 }

func (a *Attribute) Set(flag Attribute, value bool) {
	if value {
		*a |= flag
	} else {
		*a &^= flag
	}
}

func (a Attribute) ABI() ABI { return ABI(a / attrABIBase) }
func (a *Attribute) SetABI(abi ABI) {
	const mask = 1 // Only one ABI bit for now.
	*a = (*a &^ (mask * attrABIBase)) | Attribute(abi)*attrABIBase
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
	{bit: AttrOnList, s: ""},
	{bit: AttrReflectMethod, s: "REFLECTMETHOD"},
	{bit: AttrLocal, s: "LOCAL"},
	{bit: AttrWrapper, s: "WRAPPER"},
	{bit: AttrNeedCtxt, s: "NEEDCTXT"},
	{bit: AttrNoFrame, s: "NOFRAME"},
	{bit: AttrStatic, s: "STATIC"},
	{bit: AttrWasInlined, s: ""},
	{bit: AttrTopFrame, s: "TOPFRAME"},
	{bit: AttrIndexed, s: ""},
	{bit: AttrContentAddressable, s: ""},
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
	switch a.ABI() {
	case ABI0:
	case ABIInternal:
		s += "ABIInternal|"
		a.SetABI(0) // Clear ABI so we don't print below.
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

func (s *LSym) String() string {
	return s.Name
}

// The compiler needs *LSym to be assignable to cmd/compile/internal/ssa.Sym.
func (s *LSym) CanBeAnSSASym() {
}

type Pcln struct {
	// Aux symbols for pcln
	Pcsp        *LSym
	Pcfile      *LSym
	Pcline      *LSym
	Pcinline    *LSym
	Pcdata      []*LSym
	Funcdata    []*LSym
	Funcdataoff []int64
	UsedFiles   map[goobj.CUFileIndex]struct{} // file indices used while generating pcfile
	InlTree     InlTree                        // per-function inlining tree extracted from the global tree
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

// Link holds the context for writing object code from a compiler
// to be linker input or for reading that input into the linker.
type Link struct {
	Headtype           objabi.HeadType
	Arch               *LinkArch
	Debugasm           int
	Debugvlog          bool
	Debugpcln          string
	Flag_shared        bool
	Flag_dynlink       bool
	Flag_linkshared    bool
	Flag_optimize      bool
	Flag_locationlists bool
	Retpoline          bool // emit use of retpoline stubs for indirect jmp/call
	Bso                *bufio.Writer
	Pathname           string
	Pkgpath            string           // the current package's import path, "" if unknown
	hashmu             sync.Mutex       // protects hash, funchash
	hash               map[string]*LSym // name -> sym mapping
	funchash           map[string]*LSym // name -> sym mapping for ABIInternal syms
	statichash         map[string]*LSym // name -> sym mapping for static syms
	PosTable           src.PosTable
	InlTree            InlTree // global inlining tree used by gc/inl.go
	DwFixups           *DwarfFixupTable
	Imports            []goobj.ImportedPkg
	DiagFunc           func(string, ...interface{})
	DiagFlush          func()
	DebugInfo          func(fn *LSym, info *LSym, curfn interface{}) ([]dwarf.Scope, dwarf.InlCalls) // if non-nil, curfn is a *gc.Node
	GenAbstractFunc    func(fn *LSym)
	Errors             int

	InParallel    bool // parallel backend phase in effect
	UseBASEntries bool // use Base Address Selection Entries in location lists and PC ranges
	IsAsm         bool // is the source assembly language, which may contain surprising idioms (e.g., call tables)

	// state for writing objects
	Text []*LSym
	Data []*LSym

	// ABIAliases are text symbols that should be aliased to all
	// ABIs. These symbols may only be referenced and not defined
	// by this object, since the need for an alias may appear in a
	// different object than the definition. Hence, this
	// information can't be carried in the symbol definition.
	//
	// TODO(austin): Replace this with ABI wrappers once the ABIs
	// actually diverge.
	ABIAliases []*LSym

	// Constant symbols (e.g. $i64.*) are data symbols created late
	// in the concurrent phase. To ensure a deterministic order, we
	// add them to a separate list, sort at the end, and append it
	// to Data.
	constSyms []*LSym

	// pkgIdx maps package path to index. The index is used for
	// symbol reference in the object file.
	pkgIdx map[string]int32

	defs         []*LSym // list of defined symbols in the current package
	hashed64defs []*LSym // list of defined short (64-bit or less) hashed (content-addressable) symbols
	hasheddefs   []*LSym // list of defined hashed (content-addressable) symbols
	nonpkgdefs   []*LSym // list of defined non-package symbols
	nonpkgrefs   []*LSym // list of referenced non-package symbols

	Fingerprint goobj.FingerprintType // fingerprint of symbol indices, to catch index mismatch
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
	case sys.AMD64, sys.I386, sys.Wasm:
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
	Init           func(*Link)
	Preprocess     func(*Link, *LSym, ProgAlloc)
	Assemble       func(*Link, *LSym, ProgAlloc)
	Progedit       func(*Link, *Prog, ProgAlloc)
	UnaryDst       map[As]bool // Instruction takes one operand, a destination.
	DWARFRegisters map[int16]int16
}
