// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"cmd/internal/obj"
)

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// avoid <ctype.h>

// The parser's maximum stack size.
// We have to use a #define macro here since yacc
// or bison will check for its definition and use
// a potentially smaller value if it is undefined.
const (
	NHUNK    = 50000
	BUFSIZ   = 8192
	NSYMB    = 500
	NHASH    = 1024
	STRINGSZ = 200
	MAXALIGN = 7
	UINF     = 100
	PRIME1   = 3
	AUNK     = 100
	AMEM     = 0 + iota - 9
	AMEM0
	AMEM8
	AMEM16
	AMEM32
	AMEM64
	AMEM128
	ANOEQ
	ANOEQ0
	ANOEQ8
	ANOEQ16
	ANOEQ32
	ANOEQ64
	ANOEQ128
	ASTRING
	AINTER
	ANILINTER
	ASLICE
	AFLOAT32
	AFLOAT64
	ACPLX64
	ACPLX128
	BADWIDTH        = -1000000000
	MaxStackVarSize = 10 * 1024 * 1024
)

/*
 * note this is the representation
 * of the compilers string literals,
 * it is not the runtime representation
 */
type Strlit struct {
	S string
}

const (
	Mpscale = 29
	Mpprec  = 16
	Mpnorm  = Mpprec - 1
	Mpbase  = 1 << Mpscale
	Mpsign  = Mpbase >> 1
	Mpmask  = Mpbase - 1
	Mpdebug = 0
)

type Mpint struct {
	A   [Mpprec]int
	Neg uint8
	Ovf uint8
}

type Mpflt struct {
	Val Mpint
	Exp int16
}

type Mpcplx struct {
	Real Mpflt
	Imag Mpflt
}

type Val struct {
	Ctype int16
	U     struct {
		Reg  int16
		Bval int16
		Xval *Mpint
		Fval *Mpflt
		Cval *Mpcplx
		Sval *Strlit
	}
}

type Array struct {
	length   int32
	size     int32
	capacity int32
	data     string
}

type Bvec struct {
	n int32
	b []uint32
}

type Pkg struct {
	Name     string
	Path     *Strlit
	Pathsym  *Sym
	Prefix   string
	Link     *Pkg
	Imported uint8
	Exported int8
	Direct   int8
	Safe     bool
}

type Sym struct {
	Lexical    uint16
	Flags      uint8
	Sym        uint8
	Link       *Sym
	Npkg       int32
	Uniqgen    uint32
	Importdef  *Pkg
	Linkname   string
	Pkg        *Pkg
	Name       string
	Def        *Node
	Label      *Label
	Block      int32
	Lastlineno int32
	Origpkg    *Pkg
	Lsym       *obj.LSym
}

type Node struct {
	Left           *Node
	Right          *Node
	Ntest          *Node
	Nincr          *Node
	Ninit          *NodeList
	Nbody          *NodeList
	Nelse          *NodeList
	List           *NodeList
	Rlist          *NodeList
	Op             uint8
	Nointerface    bool
	Ullman         uint8
	Addable        uint8
	Trecur         uint8
	Etype          uint8
	Bounded        bool
	Class          uint8
	Method         uint8
	Embedded       uint8
	Colas          uint8
	Diag           uint8
	Noescape       bool
	Nosplit        bool
	Builtin        uint8
	Nowritebarrier bool
	Walkdef        uint8
	Typecheck      uint8
	Local          uint8
	Dodata         uint8
	Initorder      uint8
	Used           uint8
	Isddd          uint8
	Readonly       uint8
	Implicit       uint8
	Addrtaken      uint8
	Assigned       uint8
	Captured       uint8
	Byval          uint8
	Dupok          uint8
	Wrapper        uint8
	Reslice        uint8
	Likely         int8
	Hasbreak       uint8
	Needzero       uint8
	Needctxt       bool
	Esc            uint
	Funcdepth      int
	Type           *Type
	Orig           *Node
	Nname          *Node
	Shortname      *Node
	Enter          *NodeList
	Exit           *NodeList
	Cvars          *NodeList
	Dcl            *NodeList
	Inl            *NodeList
	Inldcl         *NodeList
	Closgen        int
	Outerfunc      *Node
	Val            Val
	Ntype          *Node
	Defn           *Node
	Pack           *Node
	Curfn          *Node
	Paramfld       *Type
	Decldepth      int
	Heapaddr       *Node
	Outerexpr      *Node
	Stackparam     *Node
	Alloc          *Node
	Outer          *Node
	Closure        *Node
	Top            int
	Inlvar         *Node
	Pkg            *Pkg
	Initplan       *InitPlan
	Escflowsrc     *NodeList
	Escretval      *NodeList
	Escloopdepth   int
	Sym            *Sym
	Vargen         int32
	Lineno         int32
	Endlineno      int32
	Xoffset        int64
	Stkdelta       int64
	Ostk           int32
	Iota           int32
	Walkgen        uint32
	Esclevel       int32
	Opt            interface{}
}

type NodeList struct {
	N    *Node
	Next *NodeList
	End  *NodeList
}

type Type struct {
	Etype       uint8
	Nointerface bool
	Noalg       uint8
	Chan        uint8
	Trecur      uint8
	Printed     uint8
	Embedded    uint8
	Siggen      uint8
	Funarg      uint8
	Copyany     uint8
	Local       uint8
	Deferwidth  uint8
	Broke       uint8
	Isddd       uint8
	Align       uint8
	Haspointers uint8
	Nod         *Node
	Orig        *Type
	Lineno      int
	Thistuple   int
	Outtuple    int
	Intuple     int
	Outnamed    uint8
	Method      *Type
	Xmethod     *Type
	Sym         *Sym
	Vargen      int32
	Nname       *Node
	Argwid      int64
	Type        *Type
	Width       int64
	Down        *Type
	Outer       *Type
	Note        *Strlit
	Bound       int64
	Bucket      *Type
	Hmap        *Type
	Hiter       *Type
	Map         *Type
	Maplineno   int32
	Embedlineno int32
	Copyto      *NodeList
	Lastfn      *Node
}

type Label struct {
	Used     uint8
	Sym      *Sym
	Def      *Node
	Use      *NodeList
	Link     *Label
	Gotopc   *obj.Prog
	Labelpc  *obj.Prog
	Breakpc  *obj.Prog
	Continpc *obj.Prog
}

type InitEntry struct {
	Xoffset int64
	Key     *Node
	Expr    *Node
}

type InitPlan struct {
	Lit  int64
	Zero int64
	Expr int64
	E    []InitEntry
}

const (
	EscUnknown = iota
	EscHeap
	EscScope
	EscNone
	EscReturn
	EscNever
	EscBits           = 3
	EscMask           = (1 << EscBits) - 1
	EscContentEscapes = 1 << EscBits
	EscReturnBits     = EscBits + 1
)

/*
 * Every node has a walkgen field.
 * If you want to do a traversal of a node graph that
 * might contain duplicates and want to avoid
 * visiting the same nodes twice, increment walkgen
 * before starting.  Then before processing a node, do
 *
 *	if(n->walkgen == walkgen)
 *		return;
 *	n->walkgen = walkgen;
 *
 * Such a walk cannot call another such walk recursively,
 * because of the use of the global walkgen.
 */
var walkgen uint32

const (
	SymExport   = 1 << 0
	SymPackage  = 1 << 1
	SymExported = 1 << 2
	SymUniq     = 1 << 3
	SymSiggen   = 1 << 4
	SymAsm      = 1 << 5
	SymAlgGen   = 1 << 6
)

var dclstack *Sym

type Iter struct {
	Done  int
	Tfunc *Type
	T     *Type
	An    **Node
	N     *Node
}

// Node ops.
const (
	OXXX = iota
	ONAME
	ONONAME
	OTYPE
	OPACK
	OLITERAL
	OADD
	OSUB
	OOR
	OXOR
	OADDSTR
	OADDR
	OANDAND
	OAPPEND
	OARRAYBYTESTR
	OARRAYBYTESTRTMP
	OARRAYRUNESTR
	OSTRARRAYBYTE
	OSTRARRAYBYTETMP
	OSTRARRAYRUNE
	OAS
	OAS2
	OAS2FUNC
	OAS2RECV
	OAS2MAPR
	OAS2DOTTYPE
	OASOP
	OCALL
	OCALLFUNC
	OCALLMETH
	OCALLINTER
	OCALLPART
	OCAP
	OCLOSE
	OCLOSURE
	OCMPIFACE
	OCMPSTR
	OCOMPLIT
	OMAPLIT
	OSTRUCTLIT
	OARRAYLIT
	OPTRLIT
	OCONV
	OCONVIFACE
	OCONVNOP
	OCOPY
	ODCL
	ODCLFUNC
	ODCLFIELD
	ODCLCONST
	ODCLTYPE
	ODELETE
	ODOT
	ODOTPTR
	ODOTMETH
	ODOTINTER
	OXDOT
	ODOTTYPE
	ODOTTYPE2
	OEQ
	ONE
	OLT
	OLE
	OGE
	OGT
	OIND
	OINDEX
	OINDEXMAP
	OKEY
	OPARAM
	OLEN
	OMAKE
	OMAKECHAN
	OMAKEMAP
	OMAKESLICE
	OMUL
	ODIV
	OMOD
	OLSH
	ORSH
	OAND
	OANDNOT
	ONEW
	ONOT
	OCOM
	OPLUS
	OMINUS
	OOROR
	OPANIC
	OPRINT
	OPRINTN
	OPAREN
	OSEND
	OSLICE
	OSLICEARR
	OSLICESTR
	OSLICE3
	OSLICE3ARR
	ORECOVER
	ORECV
	ORUNESTR
	OSELRECV
	OSELRECV2
	OIOTA
	OREAL
	OIMAG
	OCOMPLEX
	OBLOCK
	OBREAK
	OCASE
	OXCASE
	OCONTINUE
	ODEFER
	OEMPTY
	OFALL
	OXFALL
	OFOR
	OGOTO
	OIF
	OLABEL
	OPROC
	ORANGE
	ORETURN
	OSELECT
	OSWITCH
	OTYPESW
	OTCHAN
	OTMAP
	OTSTRUCT
	OTINTER
	OTFUNC
	OTARRAY
	ODDD
	ODDDARG
	OINLCALL
	OEFACE
	OITAB
	OSPTR
	OCLOSUREVAR
	OCFUNC
	OCHECKNIL
	OVARKILL
	OREGISTER
	OINDREG
	OCMP
	ODEC
	OINC
	OEXTEND
	OHMUL
	OLROT
	ORROTC
	ORETJMP
	OEND
)

const (
	Txxx = iota
	TINT8
	TUINT8
	TINT16
	TUINT16
	TINT32
	TUINT32
	TINT64
	TUINT64
	TINT
	TUINT
	TUINTPTR
	TCOMPLEX64
	TCOMPLEX128
	TFLOAT32
	TFLOAT64
	TBOOL
	TPTR32
	TPTR64
	TFUNC
	TARRAY
	T_old_DARRAY
	TSTRUCT
	TCHAN
	TMAP
	TINTER
	TFORW
	TFIELD
	TANY
	TSTRING
	TUNSAFEPTR
	TIDEAL
	TNIL
	TBLANK
	TFUNCARGS
	TCHANARGS
	TINTERMETH
	NTYPE
)

const (
	CTxxx = iota
	CTINT
	CTRUNE
	CTFLT
	CTCPLX
	CTSTR
	CTBOOL
	CTNIL
)

const (
	Cxxx  = 0
	Crecv = 1 << 0
	Csend = 1 << 1
	Cboth = Crecv | Csend
)

// declaration context
const (
	Pxxx = iota
	PEXTERN
	PAUTO
	PPARAM
	PPARAMOUT
	PPARAMREF
	PFUNC
	PDISCARD
	PHEAP = 1 << 7
)

const (
	Etop      = 1 << 1
	Erv       = 1 << 2
	Etype     = 1 << 3
	Ecall     = 1 << 4
	Efnstruct = 1 << 5
	Eiota     = 1 << 6
	Easgn     = 1 << 7
	Eindir    = 1 << 8
	Eaddr     = 1 << 9
	Eproc     = 1 << 10
	Ecomplit  = 1 << 11
)

const (
	BITS = 3
	NVAR = BITS * 64
)

type Bits struct {
	b [BITS]uint64
}

var zbits Bits

type Var struct {
	offset     int64
	node       *Node
	nextinnode *Var
	width      int
	id         int
	name       int8
	etype      int8
	addr       int8
}

var var_ [NVAR]Var

type Typedef struct {
	Name   string
	Etype  int
	Sameas int
}

type Sig struct {
	name   string
	pkg    *Pkg
	isym   *Sym
	tsym   *Sym
	type_  *Type
	mtype  *Type
	offset int32
	link   *Sig
}

type Io struct {
	infile     string
	bin        *obj.Biobuf
	ilineno    int32
	nlsemi     int
	eofnl      int
	last       int
	peekc      int
	peekc1     int
	cp         string
	importsafe bool
}

type Dlist struct {
	field *Type
}

type Idir struct {
	link *Idir
	dir  string
}

/*
 * argument passing to/from
 * smagic and umagic
 */
type Magic struct {
	W   int
	S   int
	Bad int
	Sd  int64
	Sm  int64
	Ud  uint64
	Um  uint64
	Ua  int
}

/*
 * note this is the runtime representation
 * of the compilers arrays.
 *
 * typedef	struct
 * {				// must not move anything
 *	uchar	array[8];	// pointer to data
 *	uchar	nel[4];		// number of elements
 *	uchar	cap[4];		// allocated number of elements
 * } Array;
 */
var Array_array int // runtime offsetof(Array,array) - same for String

var Array_nel int // runtime offsetof(Array,nel) - same for String

var Array_cap int // runtime offsetof(Array,cap)

var sizeof_Array int // runtime sizeof(Array)

/*
 * note this is the runtime representation
 * of the compilers strings.
 *
 * typedef	struct
 * {				// must not move anything
 *	uchar	array[8];	// pointer to data
 *	uchar	nel[4];		// number of elements
 * } String;
 */
var sizeof_String int // runtime sizeof(String)

var dotlist [10]Dlist // size is max depth of embeddeds

var curio Io

var pushedio Io

var lexlineno int32

var lineno int32

var prevlineno int32

var pragcgobuf string

var infile string

var outfile string

var bout *obj.Biobuf

var nerrors int

var nsavederrors int

var nsyntaxerrors int

var decldepth int

var safemode int

var nolocalimports int

var namebuf string

var lexbuf bytes.Buffer
var strbuf bytes.Buffer

func DBG(...interface{}) {}

var litbuf string

var Debug [256]int

var debugstr string

var Debug_checknil int

var hash [NHASH]*Sym

var importmyname *Sym // my name for package

var localpkg *Pkg // package being compiled

var importpkg *Pkg // package being imported

var structpkg *Pkg // package that declared struct, during import

var builtinpkg *Pkg // fake package for builtins

var gostringpkg *Pkg // fake pkg for Go strings

var itabpkg *Pkg // fake pkg for itab cache

var Runtimepkg *Pkg // package runtime

var racepkg *Pkg // package runtime/race

var stringpkg *Pkg // fake package for C strings

var typepkg *Pkg // fake package for runtime type info (headers)

var typelinkpkg *Pkg // fake package for runtime type info (data)

var weaktypepkg *Pkg // weak references to runtime type info

var unsafepkg *Pkg // package unsafe

var trackpkg *Pkg // fake package for field tracking

var rawpkg *Pkg // fake package for raw symbol names

var phash [128]*Pkg

var Tptr int // either TPTR32 or TPTR64

var myimportpath string

var idirs *Idir

var localimport string

var asmhdr string

var Types [NTYPE]*Type

var idealstring *Type

var idealbool *Type

var bytetype *Type

var runetype *Type

var errortype *Type

var Simtype [NTYPE]uint8

var Isptr [NTYPE]uint8

var isforw [NTYPE]uint8

var Isint [NTYPE]uint8

var Isfloat [NTYPE]uint8

var Iscomplex [NTYPE]uint8

var Issigned [NTYPE]uint8

var issimple [NTYPE]uint8

var okforeq [NTYPE]uint8

var okforadd [NTYPE]uint8

var okforand [NTYPE]uint8

var okfornone [NTYPE]uint8

var okforcmp [NTYPE]uint8

var okforbool [NTYPE]uint8

var okforcap [NTYPE]uint8

var okforlen [NTYPE]uint8

var okforarith [NTYPE]uint8

var okforconst [NTYPE]uint8

var okfor [OEND][]byte

var iscmp [OEND]uint8

var Minintval [NTYPE]*Mpint

var Maxintval [NTYPE]*Mpint

var minfltval [NTYPE]*Mpflt

var maxfltval [NTYPE]*Mpflt

var xtop *NodeList

var externdcl *NodeList

var exportlist *NodeList

var importlist *NodeList // imported functions and methods with inlinable bodies

var funcsyms *NodeList

var dclcontext int // PEXTERN/PAUTO

var incannedimport int

var statuniqgen int // name generator for static temps

var loophack int

var iota_ int32

var lastconst *NodeList

var lasttype *Node

var Maxarg int64

var Stksize int64 // stack size for current frame

var stkptrsize int64 // prefix of stack containing pointers

var blockgen int32 // max block number

var block int32 // current block number

var Hasdefer int // flag that curfn has defer statetment

var Curfn *Node

var Widthptr int

var Widthint int

var Widthreg int

var typesw *Node

var nblank *Node

var Use_sse int

var hunk string

var nhunk int32

var thunk int32

var Funcdepth int

var typecheckok int

var compiling_runtime int

var compiling_wrappers int

var inl_nonlocal int

var use_writebarrier int

var pure_go int

var flag_installsuffix string

var flag_race int

var flag_largemodel int

var noescape bool

var nosplit bool

var nowritebarrier bool

var debuglive int

var Ctxt *obj.Link

var nointerface bool

var writearchive int

var bstdout obj.Biobuf

var Nacl bool

/*
 *	y.tab.c
 */

/*
 *	align.c
 */

/*
 *	array.c
 */

/*
 *	bits.c
 */

/*
 *	mparith1.c
 */

/*
 *	mparith2.c
 */

/*
 *	mparith3.c
 */

/*
 *	obj.c
 */

/*
 *	order.c
 */

/*
 *	range.c
 */

/*
 *	reflect.c
 */

/*
 *	select.c
 */

/*
 *	sinit.c
 */

/*
 *	subr.c
 */

/*
 *	swt.c
 */

/*
 *	typecheck.c
 */

/*
 *	unsafe.c
 */

/*
 *	walk.c
 */

/*
 *	thearch-specific ggen.c/gsubr.c/gobj.c/pgen.c/plive.c
 */
var continpc *obj.Prog

var breakpc *obj.Prog

var Pc *obj.Prog

var firstpc *obj.Prog

var nodfp *Node

var Disable_checknil int

var zerosize int64

/*
 *	racewalk.c
 */

/*
 *	flow.c
 */
type Flow struct {
	Prog   *obj.Prog
	P1     *Flow
	P2     *Flow
	P2link *Flow
	S1     *Flow
	S2     *Flow
	Link   *Flow
	Active int32
	Id     int32
	Rpo    int32
	Loop   uint16
	Refset uint8
	Data   interface{}
}

type Graph struct {
	Start *Flow
	Num   int
	Rpo   []*Flow
}

/*
 *	interface to back end
 */
type ProgInfo struct {
	Flags    uint32
	Reguse   uint64
	Regset   uint64
	Regindex uint64
}

const (
	Pseudo     = 1 << 1
	OK         = 1 << 2
	SizeB      = 1 << 3
	SizeW      = 1 << 4
	SizeL      = 1 << 5
	SizeQ      = 1 << 6
	SizeF      = 1 << 7
	SizeD      = 1 << 8
	LeftAddr   = 1 << 9
	LeftRead   = 1 << 10
	LeftWrite  = 1 << 11
	RegRead    = 1 << 12
	CanRegRead = 1 << 13
	RightAddr  = 1 << 14
	RightRead  = 1 << 15
	RightWrite = 1 << 16
	Move       = 1 << 17
	Conv       = 1 << 18
	Cjmp       = 1 << 19
	Break      = 1 << 20
	Call       = 1 << 21
	Jump       = 1 << 22
	Skip       = 1 << 23
	SetCarry   = 1 << 24
	UseCarry   = 1 << 25
	KillCarry  = 1 << 26
	ShiftCX    = 1 << 27
	ImulAXDX   = 1 << 28
	PostInc    = 1 << 29
)

type Arch struct {
	Thechar        int
	Thestring      string
	Thelinkarch    *obj.LinkArch
	Typedefs       []Typedef
	REGSP          int
	REGCTXT        int
	MAXWIDTH       int64
	Anyregalloc    func() bool
	Betypeinit     func()
	Bgen           func(*Node, bool, int, *obj.Prog)
	Cgen           func(*Node, *Node)
	Cgen_call      func(*Node, int)
	Cgen_callinter func(*Node, *Node, int)
	Cgen_ret       func(*Node)
	Clearfat       func(*Node)
	Defframe       func(*obj.Prog)
	Excise         func(*Flow)
	Expandchecks   func(*obj.Prog)
	Gclean         func()
	Ginit          func()
	Gins           func(int, *Node, *Node) *obj.Prog
	Ginscall       func(*Node, int)
	Igen           func(*Node, *Node, *Node)
	Linkarchinit   func()
	Peep           func(*obj.Prog)
	Proginfo       func(*ProgInfo, *obj.Prog)
	Regalloc       func(*Node, *Type, *Node)
	Regfree        func(*Node)
	Regtyp         func(*obj.Addr) bool
	Sameaddr       func(*obj.Addr, *obj.Addr) bool
	Smallindir     func(*obj.Addr, *obj.Addr) bool
	Stackaddr      func(*obj.Addr) bool
	Excludedregs   func() uint64
	RtoB           func(int) uint64
	FtoB           func(int) uint64
	BtoR           func(uint64) int
	BtoF           func(uint64) int
	Optoas         func(int, *Type) int
	Doregbits      func(int) uint64
	Regnames       func(*int) []string
}

var pcloc int32

var Thearch Arch

var Newproc *Node

var Deferproc *Node

var Deferreturn *Node

var Panicindex *Node

var panicslice *Node

var throwreturn *Node
