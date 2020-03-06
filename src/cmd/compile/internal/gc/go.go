// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"sync"
)

const (
	BADWIDTH = types.BADWIDTH
)

var (
	// maximum size variable which we will allocate on the stack.
	// This limit is for explicit variable declarations like "var x T" or "x := ...".
	// Note: the flag smallframes can update this value.
	maxStackVarSize = int64(10 * 1024 * 1024)

	// maximum size of implicit variables that we will allocate on the stack.
	//   p := new(T)          allocating T on the stack
	//   p := &T{}            allocating T on the stack
	//   s := make([]T, n)    allocating [n]T on the stack
	//   s := []byte("...")   allocating [n]byte on the stack
	// Note: the flag smallframes can update this value.
	maxImplicitStackVarSize = int64(64 * 1024)

	// smallArrayBytes is the maximum size of an array which is considered small.
	// Small arrays will be initialized directly with a sequence of constant stores.
	// Large arrays will be initialized by copying from a static temp.
	// 256 bytes was chosen to minimize generated code + statictmp size.
	smallArrayBytes = int64(256)
)

// isRuntimePkg reports whether p is package runtime.
func isRuntimePkg(p *types.Pkg) bool {
	if compiling_runtime && p == localpkg {
		return true
	}
	return p.Path == "runtime"
}

// The Class of a variable/function describes the "storage class"
// of a variable or function. During parsing, storage classes are
// called declaration contexts.
type Class uint8

//go:generate stringer -type=Class
const (
	Pxxx      Class = iota // no class; used during ssa conversion to indicate pseudo-variables
	PEXTERN                // global variable
	PAUTO                  // local variables
	PAUTOHEAP              // local variable or parameter moved to heap
	PPARAM                 // input arguments
	PPARAMOUT              // output results
	PFUNC                  // global function

	// Careful: Class is stored in three bits in Node.flags.
	_ = uint((1 << 3) - iota) // static assert for iota <= (1 << 3)
)

// note this is the runtime representation
// of the compilers slices.
//
// typedef	struct
// {				// must not move anything
// 	uchar	array[8];	// pointer to data
// 	uchar	nel[4];		// number of elements
// 	uchar	cap[4];		// allocated number of elements
// } Slice;
var slice_array int // runtime offsetof(Slice,array) - same for String

var slice_nel int // runtime offsetof(Slice,nel) - same for String

var slice_cap int // runtime offsetof(Slice,cap)

var sizeof_Slice int // runtime sizeof(Slice)

// note this is the runtime representation
// of the compilers strings.
//
// typedef	struct
// {				// must not move anything
// 	uchar	array[8];	// pointer to data
// 	uchar	nel[4];		// number of elements
// } String;
var sizeof_String int // runtime sizeof(String)

var pragcgobuf [][]string

var outfile string
var linkobj string

// nerrors is the number of compiler errors reported
// since the last call to saveerrors.
var nerrors int

// nsavederrors is the total number of compiler errors
// reported before the last call to saveerrors.
var nsavederrors int

var nsyntaxerrors int

var decldepth int32

var nolocalimports bool

var Debug [256]int

var debugstr string

var Debug_checknil int
var Debug_typeassert int

var localpkg *types.Pkg // package being compiled

var inimport bool // set during import

var itabpkg *types.Pkg // fake pkg for itab entries

var itablinkpkg *types.Pkg // fake package for runtime itab entries

var Runtimepkg *types.Pkg // fake package runtime

var racepkg *types.Pkg // package runtime/race

var msanpkg *types.Pkg // package runtime/msan

var unsafepkg *types.Pkg // package unsafe

var trackpkg *types.Pkg // fake package for field tracking

var mappkg *types.Pkg // fake package for map zero value

var gopkg *types.Pkg // pseudo-package for method symbols on anonymous receiver types

var zerosize int64

var myimportpath string

var localimport string

var asmhdr string

var simtype [NTYPE]types.EType

var (
	isInt     [NTYPE]bool
	isFloat   [NTYPE]bool
	isComplex [NTYPE]bool
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

var minintval [NTYPE]*Mpint

var maxintval [NTYPE]*Mpint

var minfltval [NTYPE]*Mpflt

var maxfltval [NTYPE]*Mpflt

var xtop []*Node

var exportlist []*Node

var importlist []*Node // imported functions and methods with inlinable bodies

var (
	funcsymsmu sync.Mutex // protects funcsyms and associated package lookups (see func funcsym)
	funcsyms   []*types.Sym
)

var dclcontext Class // PEXTERN/PAUTO

var Curfn *Node

var Widthptr int

var Widthreg int

var nblank *Node

var typecheckok bool

var compiling_runtime bool

// Compiling the standard library
var compiling_std bool

var use_writebarrier bool

var pure_go bool

var flag_installsuffix string

var flag_race bool

var flag_msan bool

var flagDWARF bool

// Whether we are adding any sort of code instrumentation, such as
// when the race detector is enabled.
var instrumenting bool

// Whether we are tracking lexical scopes for DWARF.
var trackScopes bool

// Controls generation of DWARF inlined instance records. Zero
// disables, 1 emits inlined routines but suppresses var info,
// and 2 emits inlined routines with tracking of formals/locals.
var genDwarfInline int

var debuglive int

var Ctxt *obj.Link

var writearchive bool

var nodfp *Node

var disable_checknil int

var autogeneratedPos src.XPos

// interface to back end

type Arch struct {
	LinkArch *obj.LinkArch

	REGSP     int
	MAXWIDTH  int64
	Use387    bool // should 386 backend use 387 FP instructions instead of sse2.
	SoftFloat bool

	PadFrame func(int64) int64

	// ZeroRange zeroes a range of memory on stack. It is only inserted
	// at function entry, and it is ok to clobber registers.
	ZeroRange func(*Progs, *obj.Prog, int64, int64, *uint32) *obj.Prog

	Ginsnop      func(*Progs) *obj.Prog
	Ginsnopdefer func(*Progs) *obj.Prog // special ginsnop for deferreturn

	// SSAMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
	SSAMarkMoves func(*SSAGenState, *ssa.Block)

	// SSAGenValue emits Prog(s) for the Value.
	SSAGenValue func(*SSAGenState, *ssa.Value)

	// SSAGenBlock emits end-of-block Progs. SSAGenValue should be called
	// for all values in the block before SSAGenBlock.
	SSAGenBlock func(s *SSAGenState, b, next *ssa.Block)
}

var thearch Arch

var (
	staticuint64s,
	zerobase *Node

	assertE2I,
	assertE2I2,
	assertI2I,
	assertI2I2,
	deferproc,
	deferprocStack,
	Deferreturn,
	Duffcopy,
	Duffzero,
	gcWriteBarrier,
	goschedguarded,
	growslice,
	msanread,
	msanwrite,
	newobject,
	newproc,
	panicdivide,
	panicshift,
	panicdottypeE,
	panicdottypeI,
	panicnildottype,
	panicoverflow,
	raceread,
	racereadrange,
	racewrite,
	racewriterange,
	x86HasPOPCNT,
	x86HasSSE41,
	x86HasFMA,
	armHasVFPv4,
	arm64HasATOMICS,
	typedmemclr,
	typedmemmove,
	Udiv,
	writeBarrier,
	zerobaseSym *obj.LSym

	BoundsCheckFunc [ssa.BoundsKindCount]*obj.LSym
	ExtendCheckFunc [ssa.BoundsKindCount]*obj.LSym

	// GO386=387
	ControlWord64trunc,
	ControlWord32 *obj.LSym

	// Wasm
	WasmMove,
	WasmZero,
	WasmDiv,
	WasmTruncS,
	WasmTruncU,
	SigPanic *obj.LSym
)
