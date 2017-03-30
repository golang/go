// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/internal/bio"
	"cmd/internal/obj"
	"cmd/internal/src"
)

const (
	BADWIDTH        = -1000000000
	MaxStackVarSize = 10 * 1024 * 1024
)

type Pkg struct {
	Name     string // package name, e.g. "sys"
	Path     string // string literal used in import statement, e.g. "runtime/internal/sys"
	Pathsym  *obj.LSym
	Prefix   string // escaped path for use in symbol table
	Imported bool   // export data of this package was parsed
	Direct   bool   // imported directly
	Syms     map[string]*Sym
}

// isRuntime reports whether p is package runtime.
func (p *Pkg) isRuntime() bool {
	if compiling_runtime && p == localpkg {
		return true
	}
	return p.Path == "runtime"
}

// Sym represents an object name. Most commonly, this is a Go identifier naming
// an object declared within a package, but Syms are also used to name internal
// synthesized objects.
//
// As an exception, field and method names that are exported use the Sym
// associated with localpkg instead of the package that declared them. This
// allows using Sym pointer equality to test for Go identifier uniqueness when
// handling selector expressions.
type Sym struct {
	Link      *Sym
	Importdef *Pkg   // where imported definition was found
	Linkname  string // link name

	// saved and restored by dcopy
	Pkg        *Pkg
	Name       string   // object name
	Def        *Node    // definition: ONAME OTYPE OPACK or OLITERAL
	Lastlineno src.XPos // last declaration for diagnostic
	Block      int32    // blocknumber to catch redeclaration

	flags   bitset8
	Label   *Node // corresponding label (ephemeral)
	Origpkg *Pkg  // original package for . import
	Lsym    *obj.LSym
}

const (
	symExport = 1 << iota // added to exportlist (no need to add again)
	symPackage
	symExported // already written out by export
	symUniq
	symSiggen
	symAsm
	symAlgGen
)

func (sym *Sym) Export() bool   { return sym.flags&symExport != 0 }
func (sym *Sym) Package() bool  { return sym.flags&symPackage != 0 }
func (sym *Sym) Exported() bool { return sym.flags&symExported != 0 }
func (sym *Sym) Uniq() bool     { return sym.flags&symUniq != 0 }
func (sym *Sym) Siggen() bool   { return sym.flags&symSiggen != 0 }
func (sym *Sym) Asm() bool      { return sym.flags&symAsm != 0 }
func (sym *Sym) AlgGen() bool   { return sym.flags&symAlgGen != 0 }

func (sym *Sym) SetExport(b bool)   { sym.flags.set(symExport, b) }
func (sym *Sym) SetPackage(b bool)  { sym.flags.set(symPackage, b) }
func (sym *Sym) SetExported(b bool) { sym.flags.set(symExported, b) }
func (sym *Sym) SetUniq(b bool)     { sym.flags.set(symUniq, b) }
func (sym *Sym) SetSiggen(b bool)   { sym.flags.set(symSiggen, b) }
func (sym *Sym) SetAsm(b bool)      { sym.flags.set(symAsm, b) }
func (sym *Sym) SetAlgGen(b bool)   { sym.flags.set(symAlgGen, b) }

func (sym *Sym) isAlias() bool {
	return sym.Def != nil && sym.Def.Sym != sym
}

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
// {				// must not move anything
// 	uchar	array[8];	// pointer to data
// 	uchar	nel[4];		// number of elements
// 	uchar	cap[4];		// allocated number of elements
// } Array;
var array_array int // runtime offsetof(Array,array) - same for String

var array_nel int // runtime offsetof(Array,nel) - same for String

var array_cap int // runtime offsetof(Array,cap)

var sizeof_Array int // runtime sizeof(Array)

// note this is the runtime representation
// of the compilers strings.
//
// typedef	struct
// {				// must not move anything
// 	uchar	array[8];	// pointer to data
// 	uchar	nel[4];		// number of elements
// } String;
var sizeof_String int // runtime sizeof(String)

var pragcgobuf string

var outfile string
var linkobj string
var dolinkobj bool

var bout *bio.Writer

// nerrors is the number of compiler errors reported
// since the last call to saveerrors.
var nerrors int

// nsavederrors is the total number of compiler errors
// reported before the last call to saveerrors.
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

var inimport bool // set during import

var itabpkg *Pkg // fake pkg for itab entries

var itablinkpkg *Pkg // fake package for runtime itab entries

var Runtimepkg *Pkg // fake package runtime

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

var simtype [NTYPE]EType

var (
	isforw    [NTYPE]bool
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

var funcsyms []*Sym

var dclcontext Class // PEXTERN/PAUTO

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

var Nacl bool

var nodfp *Node

var disable_checknil int

var autogeneratedPos src.XPos

// interface to back end

type Arch struct {
	LinkArch *obj.LinkArch

	REGSP    int
	MAXWIDTH int64
	Use387   bool // should 386 backend use 387 FP instructions instead of sse2.

	Defframe func(*Progs, *Node, int64)
	Ginsnop  func(*Progs)

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
	staticbytes,
	zerobase *Node

	Newproc,
	Deferproc,
	Deferreturn,
	Duffcopy,
	Duffzero,
	panicindex,
	panicslice,
	panicdivide,
	growslice,
	panicdottypeE,
	panicdottypeI,
	panicnildottype,
	assertE2I,
	assertE2I2,
	assertI2I,
	assertI2I2 *obj.LSym
)
