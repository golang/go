// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bufio"
	"cmd/compile/internal/ssa"
	"cmd/internal/bio"
	"cmd/internal/obj"
)

const (
	UINF            = 100
	BADWIDTH        = -1000000000
	MaxStackVarSize = 10 * 1024 * 1024
)

type Pkg struct {
	Name     string // package name, e.g. "sys"
	Path     string // string literal used in import statement, e.g. "runtime/internal/sys"
	Pathsym  *obj.LSym
	Prefix   string // escaped path for use in symbol table
	Imported bool   // export data of this package was parsed
	Exported bool   // import line written in export data
	Direct   bool   // imported directly
	Safe     bool   // whether the package is marked as safe
	Syms     map[string]*Sym
}

// Sym represents an object name. Most commonly, this is a Go identifier naming
// an object declared within a package, but Syms are also used to name internal
// synthesized objects.
//
// As a special exception, field and method names that are exported use the Sym
// associated with localpkg instead of the package that declared them. This
// allows using Sym pointer equality to test for Go identifier uniqueness when
// handling selector expressions.
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

// The Class of a variable/function describes the "storage class"
// of a variable or function. During parsing, storage classes are
// called declaration contexts.
type Class uint8

const (
	Pxxx      Class = iota
	PEXTERN         // global variable
	PAUTO           // local variables
	PAUTOHEAP       // local variable or parameter moved to heap
	PPARAM          // input arguments
	PPARAMOUT       // output results
	PFUNC           // global function

	PDISCARD // discard during parse of duplicate import
)

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

var pragcgobuf string

var infile string

var outfile string
var linkobj string

var bout *bio.Writer

var nerrors int

var nsavederrors int

var nsyntaxerrors int

var decldepth int32

var safemode bool

var nolocalimports bool

var Debug [256]int

var debugstr string

var Debug_checknil int
var Debug_typeassert int

var localpkg *Pkg // package being compiled

var autopkg *Pkg // fake package for allocating auto variables

var importpkg *Pkg // package being imported

var itabpkg *Pkg // fake pkg for itab entries

var itablinkpkg *Pkg // fake package for runtime itab entries

var Runtimepkg *Pkg // package runtime

var racepkg *Pkg // package runtime/race

var msanpkg *Pkg // package runtime/msan

var typepkg *Pkg // fake package for runtime type info (headers)

var unsafepkg *Pkg // package unsafe

var trackpkg *Pkg // fake package for field tracking

var mappkg *Pkg // fake package for map zero value
var zerosize int64

var Tptr EType // either TPTR32 or TPTR64

var myimportpath string

var localimport string

var asmhdr string

var Simtype [NTYPE]EType

var (
	isforw    [NTYPE]bool
	Isint     [NTYPE]bool
	Isfloat   [NTYPE]bool
	Iscomplex [NTYPE]bool
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

var hasdefer bool // flag that curfn has defer statement

var Curfn *Node

var Widthptr int

var Widthint int

var Widthreg int

var nblank *Node

var typecheckok bool

var compiling_runtime bool

var compiling_wrappers int

var use_writebarrier bool

var pure_go bool

var flag_installsuffix string

var flag_race bool

var flag_msan bool

var flag_largemodel bool

// Whether we are adding any sort of code instrumentation, such as
// when the race detector is enabled.
var instrumenting bool

var debuglive int

var Ctxt *obj.Link

var writearchive bool

var bstdout *bufio.Writer

var Nacl bool

var continpc *obj.Prog

var breakpc *obj.Prog

var Pc *obj.Prog

var nodfp *Node

var Disable_checknil int

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
	// Originally for understanding ADC, RCR, and so on, but now also
	// tracks set, use, and kill of the zero and overflow bits as well.
	// TODO rename to {Set,Use,Kill}Flags
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
	LinkArch *obj.LinkArch

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

	AddIndex            func(*Node, int64, *Node) bool // optional
	Betypeinit          func()
	Bgen_float          func(*Node, bool, int, *obj.Prog) // optional
	Cgen64              func(*Node, *Node)                // only on 32-bit systems
	Cgenindex           func(*Node, *Node, bool) *obj.Prog
	Cgen_bmul           func(Op, *Node, *Node, *Node) bool
	Cgen_float          func(*Node, *Node) // optional
	Cgen_hmul           func(*Node, *Node, *Node)
	RightShiftWithCarry func(*Node, uint, *Node)  // only on systems without RROTC instruction
	AddSetCarry         func(*Node, *Node, *Node) // only on systems when ADD does not update carry flag
	Cgen_shift          func(Op, bool, *Node, *Node, *Node)
	Clearfat            func(*Node)
	Cmp64               func(*Node, *Node, Op, int, *obj.Prog) // only on 32-bit systems
	Defframe            func(*obj.Prog)
	Dodiv               func(Op, *Node, *Node, *Node)
	Excise              func(*Flow)
	Expandchecks        func(*obj.Prog)
	Getg                func(*Node)
	Gins                func(obj.As, *Node, *Node) *obj.Prog

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

	// SSARegToReg maps ssa register numbers to obj register numbers.
	SSARegToReg []int16

	// SSAMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
	SSAMarkMoves func(*SSAGenState, *ssa.Block)

	// SSAGenValue emits Prog(s) for the Value.
	SSAGenValue func(*SSAGenState, *ssa.Value)

	// SSAGenBlock emits end-of-block Progs. SSAGenValue should be called
	// for all values in the block before SSAGenBlock.
	SSAGenBlock func(s *SSAGenState, b, next *ssa.Block)
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
