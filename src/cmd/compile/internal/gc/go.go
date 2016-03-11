// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"cmd/internal/obj"
)

const (
	UINF            = 100
	PRIME1          = 3
	BADWIDTH        = -1000000000
	MaxStackVarSize = 10 * 1024 * 1024
)

type Val struct {
	// U contains one of:
	// bool     bool when n.ValCtype() == CTBOOL
	// *Mpint   int when n.ValCtype() == CTINT, rune when n.ValCtype() == CTRUNE
	// *Mpflt   float when n.ValCtype() == CTFLT
	// *Mpcplx  pair of floats when n.ValCtype() == CTCPLX
	// string   string when n.ValCtype() == CTSTR
	// *Nilval  when n.ValCtype() == CTNIL
	U interface{}
}

type NilVal struct{}

func (v Val) Ctype() Ctype {
	switch x := v.U.(type) {
	default:
		Fatalf("unexpected Ctype for %T", v.U)
		panic("not reached")
	case nil:
		return 0
	case *NilVal:
		return CTNIL
	case bool:
		return CTBOOL
	case *Mpint:
		if x.Rune {
			return CTRUNE
		}
		return CTINT
	case *Mpflt:
		return CTFLT
	case *Mpcplx:
		return CTCPLX
	case string:
		return CTSTR
	}
}

type Pkg struct {
	Name     string // package name
	Path     string // string literal used in import statement
	Pathsym  *Sym
	Prefix   string // escaped path for use in symbol table
	Imported bool   // export data of this package was parsed
	Exported bool   // import line written in export data
	Direct   bool   // imported directly
	Safe     bool   // whether the package is marked as safe
	Syms     map[string]*Sym
}

type Sym struct {
	Flags     SymFlags
	Link      *Sym
	Importdef *Pkg   // where imported definition was found
	Linkname  string // link name

	// saved and restored by dcopy
	Pkg        *Pkg
	Name       string // variable name
	Def        *Node  // definition: ONAME OTYPE OPACK or OLITERAL
	Block      int32  // blocknumber to catch redeclaration
	Lastlineno int32  // last declaration for diagnostic

	Label   *Label // corresponding label (ephemeral)
	Origpkg *Pkg   // original package for . import
	Lsym    *obj.LSym
	Fsym    *Sym // funcsym
}

type Label struct {
	Sym *Sym
	Def *Node
	Use []*Node

	// for use during gen
	Gotopc   *obj.Prog // pointer to unresolved gotos
	Labelpc  *obj.Prog // pointer to code
	Breakpc  *obj.Prog // pointer to code
	Continpc *obj.Prog // pointer to code

	Used bool
}

type InitEntry struct {
	Xoffset int64 // struct, array only
	Expr    *Node // bytes of run-time computed expressions
}

type InitPlan struct {
	Lit  int64
	Zero int64
	Expr int64
	E    []InitEntry
}

type SymFlags uint8

const (
	SymExport SymFlags = 1 << iota // to be exported
	SymPackage
	SymExported // already written out by export
	SymUniq
	SymSiggen
	SymAsm
	SymAlgGen
)

var dclstack *Sym

// Ctype describes the constant kind of an "ideal" (untyped) constant.
type Ctype int8

const (
	CTxxx Ctype = iota

	CTINT
	CTRUNE
	CTFLT
	CTCPLX
	CTSTR
	CTBOOL
	CTNIL
)

const (
	// types of channel
	// must match ../../../../reflect/type.go:/ChanDir
	Crecv = 1 << 0
	Csend = 1 << 1
	Cboth = Crecv | Csend
)

// The Class of a variable/function describes the "storage class"
// of a variable or function. During parsing, storage classes are
// called declaration contexts.
type Class uint8

const (
	Pxxx      Class = iota
	PEXTERN         // global variable
	PAUTO           // local variables
	PPARAM          // input arguments
	PPARAMOUT       // output results
	PPARAMREF       // closure variable reference
	PFUNC           // global function

	PDISCARD // discard during parse of duplicate import

	PHEAP = 1 << 7 // an extra bit to identify an escaped variable
)

const (
	Etop      = 1 << 1 // evaluated at statement level
	Erv       = 1 << 2 // evaluated in value context
	Etype     = 1 << 3
	Ecall     = 1 << 4  // call-only expressions are ok
	Efnstruct = 1 << 5  // multivalue function returns are ok
	Eiota     = 1 << 6  // iota is ok
	Easgn     = 1 << 7  // assigning to expression
	Eindir    = 1 << 8  // indirecting through expression
	Eaddr     = 1 << 9  // taking address of expression
	Eproc     = 1 << 10 // inside a go statement
	Ecomplit  = 1 << 11 // type in composite literal
)

type Sig struct {
	name   string
	pkg    *Pkg
	isym   *Sym
	tsym   *Sym
	type_  *Type
	mtype  *Type
	offset int32
}

// argument passing to/from
// smagic and umagic
type Magic struct {
	W   int // input for both - width
	S   int // output for both - shift
	Bad int // output for both - unexpected failure

	// magic multiplier for signed literal divisors
	Sd int64 // input - literal divisor
	Sm int64 // output - multiplier

	// magic multiplier for unsigned literal divisors
	Ud uint64 // input - literal divisor
	Um uint64 // output - multiplier
	Ua int    // output - adder
}

// note this is the runtime representation
// of the compilers arrays.
//
// typedef	struct
// {					// must not move anything
// 	uchar	array[8];	// pointer to data
// 	uchar	nel[4];		// number of elements
// 	uchar	cap[4];		// allocated number of elements
// } Array;
var Array_array int // runtime offsetof(Array,array) - same for String

var Array_nel int // runtime offsetof(Array,nel) - same for String

var Array_cap int // runtime offsetof(Array,cap)

var sizeof_Array int // runtime sizeof(Array)

// note this is the runtime representation
// of the compilers strings.
//
// typedef	struct
// {					// must not move anything
// 	uchar	array[8];	// pointer to data
// 	uchar	nel[4];		// number of elements
// } String;
var sizeof_String int // runtime sizeof(String)

// lexlineno is the line number _after_ the most recently read rune.
// In particular, it's advanced (or rewound) as newlines are read (or unread).
var lexlineno int32

// lineno is the line number at the start of the most recently lexed token.
var lineno int32

var pragcgobuf string

var infile string

var outfile string

var bout *obj.Biobuf

var nerrors int

var nsavederrors int

var nsyntaxerrors int

var decldepth int32

var safemode int

var nolocalimports int

var lexbuf bytes.Buffer
var strbuf bytes.Buffer
var litbuf string // LLITERAL value for use in syntax error messages

var Debug [256]int

var debugstr string

var Debug_checknil int
var Debug_typeassert int

var localpkg *Pkg // package being compiled

var importpkg *Pkg // package being imported

var structpkg *Pkg // package that declared struct, during import

var builtinpkg *Pkg // fake package for builtins

var gostringpkg *Pkg // fake pkg for Go strings

var itabpkg *Pkg // fake pkg for itab cache

var Runtimepkg *Pkg // package runtime

var racepkg *Pkg // package runtime/race

var msanpkg *Pkg // package runtime/msan

var typepkg *Pkg // fake package for runtime type info (headers)

var typelinkpkg *Pkg // fake package for runtime type info (data)

var unsafepkg *Pkg // package unsafe

var trackpkg *Pkg // fake package for field tracking

var Tptr EType // either TPTR32 or TPTR64

var myimportpath string

var localimport string

var asmhdr string

var Simtype [NTYPE]EType

var (
	Isptr     [NTYPE]bool
	isforw    [NTYPE]bool
	Isint     [NTYPE]bool
	Isfloat   [NTYPE]bool
	Iscomplex [NTYPE]bool
	Issigned  [NTYPE]bool
	issimple  [NTYPE]bool
)

var (
	okforeq    [NTYPE]bool
	okforadd   [NTYPE]bool
	okforand   [NTYPE]bool
	okfornone  [NTYPE]bool
	okforcmp   [NTYPE]bool
	okforbool  [NTYPE]bool
	okforcap   [NTYPE]bool
	okforlen   [NTYPE]bool
	okforarith [NTYPE]bool
	okforconst [NTYPE]bool
)

var (
	okfor [OEND][]bool
	iscmp [OEND]bool
)

var Minintval [NTYPE]*Mpint

var Maxintval [NTYPE]*Mpint

var minfltval [NTYPE]*Mpflt

var maxfltval [NTYPE]*Mpflt

var xtop []*Node

var externdcl []*Node

var exportlist []*Node

var importlist []*Node // imported functions and methods with inlinable bodies

var funcsyms []*Node

var dclcontext Class // PEXTERN/PAUTO

var incannedimport int

var statuniqgen int // name generator for static temps

var iota_ int32

var lastconst []*Node

var lasttype *Node

var Maxarg int64

var Stksize int64 // stack size for current frame

var stkptrsize int64 // prefix of stack containing pointers

var blockgen int32 // max block number

var block int32 // current block number

var hasdefer bool // flag that curfn has defer statement

var Curfn *Node

var Widthptr int

var Widthint int

var Widthreg int

var nblank *Node

var Funcdepth int32

var typecheckok bool

var compiling_runtime int

var compiling_wrappers int

var use_writebarrier int

var pure_go int

var flag_installsuffix string

var flag_race int

var flag_msan int

var flag_largemodel int

// Whether we are adding any sort of code instrumentation, such as
// when the race detector is enabled.
var instrumenting bool

var debuglive int

var Ctxt *obj.Link

var writearchive int

var bstdout obj.Biobuf

var Nacl bool

var continpc *obj.Prog

var breakpc *obj.Prog

var Pc *obj.Prog

var nodfp *Node

var Disable_checknil int

type Flow struct {
	Prog   *obj.Prog // actual instruction
	P1     *Flow     // predecessors of this instruction: p1,
	P2     *Flow     // and then p2 linked though p2link.
	P2link *Flow
	S1     *Flow // successors of this instruction (at most two: s1 and s2).
	S2     *Flow
	Link   *Flow // next instruction in function code

	Active int32 // usable by client

	Id     int32  // sequence number in flow graph
	Rpo    int32  // reverse post ordering
	Loop   uint16 // x5 for every loop
	Refset bool   // diagnostic generated

	Data interface{} // for use by client
}

type Graph struct {
	Start *Flow
	Num   int

	// After calling flowrpo, rpo lists the flow nodes in reverse postorder,
	// and each non-dead Flow node f has g->rpo[f->rpo] == f.
	Rpo []*Flow
}

// interface to back end

const (
	// Pseudo-op, like TEXT, GLOBL, TYPE, PCDATA, FUNCDATA.
	Pseudo = 1 << 1

	// There's nothing to say about the instruction,
	// but it's still okay to see.
	OK = 1 << 2

	// Size of right-side write, or right-side read if no write.
	SizeB = 1 << 3
	SizeW = 1 << 4
	SizeL = 1 << 5
	SizeQ = 1 << 6
	SizeF = 1 << 7
	SizeD = 1 << 8

	// Left side (Prog.from): address taken, read, write.
	LeftAddr  = 1 << 9
	LeftRead  = 1 << 10
	LeftWrite = 1 << 11

	// Register in middle (Prog.reg); only ever read. (arm, ppc64)
	RegRead    = 1 << 12
	CanRegRead = 1 << 13

	// Right side (Prog.to): address taken, read, write.
	RightAddr  = 1 << 14
	RightRead  = 1 << 15
	RightWrite = 1 << 16

	// Instruction kinds
	Move  = 1 << 17 // straight move
	Conv  = 1 << 18 // size conversion
	Cjmp  = 1 << 19 // conditional jump
	Break = 1 << 20 // breaks control flow (no fallthrough)
	Call  = 1 << 21 // function call
	Jump  = 1 << 22 // jump
	Skip  = 1 << 23 // data instruction

	// Set, use, or kill of carry bit.
	// Kill means we never look at the carry bit after this kind of instruction.
	SetCarry  = 1 << 24
	UseCarry  = 1 << 25
	KillCarry = 1 << 26

	// Special cases for register use. (amd64, 386)
	ShiftCX  = 1 << 27 // possible shift by CX
	ImulAXDX = 1 << 28 // possible multiply into DX:AX

	// Instruction updates whichever of from/to is type D_OREG. (ppc64)
	PostInc = 1 << 29
)

type Arch struct {
	Thechar      int
	Thestring    string
	Thelinkarch  *obj.LinkArch
	REGSP        int
	REGCTXT      int
	REGCALLX     int // BX
	REGCALLX2    int // AX
	REGRETURN    int // AX
	REGMIN       int
	REGMAX       int
	REGZERO      int // architectural zero register, if available
	FREGMIN      int
	FREGMAX      int
	MAXWIDTH     int64
	ReservedRegs []int

	AddIndex     func(*Node, int64, *Node) bool // optional
	Betypeinit   func()
	Bgen_float   func(*Node, bool, int, *obj.Prog) // optional
	Cgen64       func(*Node, *Node)                // only on 32-bit systems
	Cgenindex    func(*Node, *Node, bool) *obj.Prog
	Cgen_bmul    func(Op, *Node, *Node, *Node) bool
	Cgen_float   func(*Node, *Node) // optional
	Cgen_hmul    func(*Node, *Node, *Node)
	Cgen_shift   func(Op, bool, *Node, *Node, *Node)
	Clearfat     func(*Node)
	Cmp64        func(*Node, *Node, Op, int, *obj.Prog) // only on 32-bit systems
	Defframe     func(*obj.Prog)
	Dodiv        func(Op, *Node, *Node, *Node)
	Excise       func(*Flow)
	Expandchecks func(*obj.Prog)
	Getg         func(*Node)
	Gins         func(obj.As, *Node, *Node) *obj.Prog

	// Ginscmp generates code comparing n1 to n2 and jumping away if op is satisfied.
	// The returned prog should be Patch'ed with the jump target.
	// If op is not satisfied, code falls through to the next emitted instruction.
	// Likely is the branch prediction hint: +1 for likely, -1 for unlikely, 0 for no opinion.
	//
	// Ginscmp must be able to handle all kinds of arguments for n1 and n2,
	// not just simple registers, although it can assume that there are no
	// function calls needed during the evaluation, and on 32-bit systems
	// the values are guaranteed not to be 64-bit values, so no in-memory
	// temporaries are necessary.
	Ginscmp func(op Op, t *Type, n1, n2 *Node, likely int) *obj.Prog

	// Ginsboolval inserts instructions to convert the result
	// of a just-completed comparison to a boolean value.
	// The first argument is the conditional jump instruction
	// corresponding to the desired value.
	// The second argument is the destination.
	// If not present, Ginsboolval will be emulated with jumps.
	Ginsboolval func(obj.As, *Node)

	Ginscon      func(obj.As, int64, *Node)
	Ginsnop      func()
	Gmove        func(*Node, *Node)
	Igenindex    func(*Node, *Node, bool) *obj.Prog
	Linkarchinit func()
	Peep         func(*obj.Prog)
	Proginfo     func(*obj.Prog) // fills in Prog.Info
	Regtyp       func(*obj.Addr) bool
	Sameaddr     func(*obj.Addr, *obj.Addr) bool
	Smallindir   func(*obj.Addr, *obj.Addr) bool
	Stackaddr    func(*obj.Addr) bool
	Blockcopy    func(*Node, *Node, int64, int64, int64)
	Sudoaddable  func(obj.As, *Node, *obj.Addr) bool
	Sudoclean    func()
	Excludedregs func() uint64
	RtoB         func(int) uint64
	FtoB         func(int) uint64
	BtoR         func(uint64) int
	BtoF         func(uint64) int
	Optoas       func(Op, *Type) obj.As
	Doregbits    func(int) uint64
	Regnames     func(*int) []string
	Use387       bool // should 8g use 387 FP instructions instead of sse2.
}

var pcloc int32

var Thearch Arch

var Newproc *Node

var Deferproc *Node

var Deferreturn *Node

var Panicindex *Node

var panicslice *Node

var panicdivide *Node

var throwreturn *Node

var growslice *Node

var writebarrierptr *Node
var typedmemmove *Node

var panicdottype *Node
