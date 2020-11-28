// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// “Abstract” syntax representation.

package ir

import (
	"fmt"
	"go/constant"
	"sort"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
)

// A Node is the abstract interface to an IR node.
type Node interface {
	// Formatting
	Format(s fmt.State, verb rune)
	String() string

	// Source position.
	Pos() src.XPos
	SetPos(x src.XPos)

	// For making copies. Mainly used by Copy and SepCopy.
	RawCopy() Node

	// Abstract graph structure, for generic traversals.
	Op() Op
	SetOp(x Op)
	SubOp() Op
	SetSubOp(x Op)
	Left() Node
	SetLeft(x Node)
	Right() Node
	SetRight(x Node)
	Init() Nodes
	PtrInit() *Nodes
	SetInit(x Nodes)
	Body() Nodes
	PtrBody() *Nodes
	SetBody(x Nodes)
	List() Nodes
	SetList(x Nodes)
	PtrList() *Nodes
	Rlist() Nodes
	SetRlist(x Nodes)
	PtrRlist() *Nodes

	// Fields specific to certain Ops only.
	Type() *types.Type
	SetType(t *types.Type)
	Func() *Func
	SetFunc(x *Func)
	Name() *Name
	SetName(x *Name)
	Sym() *types.Sym
	SetSym(x *types.Sym)
	Offset() int64
	SetOffset(x int64)
	Class() Class
	SetClass(x Class)
	Likely() bool
	SetLikely(x bool)
	SliceBounds() (low, high, max Node)
	SetSliceBounds(low, high, max Node)
	Iota() int64
	SetIota(x int64)
	Colas() bool
	SetColas(x bool)
	NoInline() bool
	SetNoInline(x bool)
	Transient() bool
	SetTransient(x bool)
	Implicit() bool
	SetImplicit(x bool)
	IsDDD() bool
	SetIsDDD(x bool)
	Embedded() bool
	SetEmbedded(x bool)
	IndexMapLValue() bool
	SetIndexMapLValue(x bool)
	TChanDir() types.ChanDir
	SetTChanDir(x types.ChanDir)
	ResetAux()
	HasBreak() bool
	SetHasBreak(x bool)
	MarkReadonly()
	Val() constant.Value
	HasVal() bool
	SetVal(v constant.Value)
	Int64Val() int64
	Uint64Val() uint64
	CanInt64() bool
	BoolVal() bool
	StringVal() string

	// Storage for analysis passes.
	Esc() uint16
	SetEsc(x uint16)
	Walkdef() uint8
	SetWalkdef(x uint8)
	Opt() interface{}
	SetOpt(x interface{})
	Diag() bool
	SetDiag(x bool)
	Bounded() bool
	SetBounded(x bool)
	Typecheck() uint8
	SetTypecheck(x uint8)
	Initorder() uint8
	SetInitorder(x uint8)
	NonNil() bool
	MarkNonNil()
	HasCall() bool
	SetHasCall(x bool)

	// Only for SSA and should be removed when SSA starts
	// using a more specific type than Node.
	CanBeAnSSASym()
}

var _ Node = (*node)(nil)

// A Node is a single node in the syntax tree.
// Actually the syntax tree is a syntax DAG, because there is only one
// node with Op=ONAME for a given instance of a variable x.
// The same is true for Op=OTYPE and Op=OLITERAL. See Node.mayBeShared.
type node struct {
	// Tree structure.
	// Generic recursive walks should follow these fields.
	left  Node
	right Node
	init  Nodes
	body  Nodes
	list  Nodes
	rlist Nodes

	// most nodes
	typ  *types.Type
	orig Node // original form, for printing, and tracking copies of ONAMEs

	// func
	fn *Func

	// ONAME, OTYPE, OPACK, OLABEL, some OLITERAL
	name *Name

	sym *types.Sym  // various
	e   interface{} // Opt or Val, see methods below

	// Various. Usually an offset into a struct. For example:
	// - ONAME nodes that refer to local variables use it to identify their stack frame position.
	// - ODOT, ODOTPTR, and ORESULT use it to indicate offset relative to their base address.
	// - OSTRUCTKEY uses it to store the named field's offset.
	// - Named OLITERALs use it to store their ambient iota value.
	// - OINLMARK stores an index into the inlTree data structure.
	// - OCLOSURE uses it to store ambient iota value, if any.
	// Possibly still more uses. If you find any, document them.
	offset int64

	pos src.XPos

	flags bitset32

	esc uint16 // EscXXX

	op  Op
	aux uint8
}

func (n *node) Left() Node            { return n.left }
func (n *node) SetLeft(x Node)        { n.left = x }
func (n *node) Right() Node           { return n.right }
func (n *node) SetRight(x Node)       { n.right = x }
func (n *node) Orig() Node            { return n.orig }
func (n *node) SetOrig(x Node)        { n.orig = x }
func (n *node) Type() *types.Type     { return n.typ }
func (n *node) SetType(x *types.Type) { n.typ = x }
func (n *node) Func() *Func           { return n.fn }
func (n *node) SetFunc(x *Func)       { n.fn = x }
func (n *node) Name() *Name           { return n.name }
func (n *node) SetName(x *Name)       { n.name = x }
func (n *node) Sym() *types.Sym       { return n.sym }
func (n *node) SetSym(x *types.Sym)   { n.sym = x }
func (n *node) Pos() src.XPos         { return n.pos }
func (n *node) SetPos(x src.XPos)     { n.pos = x }
func (n *node) Offset() int64         { return n.offset }
func (n *node) SetOffset(x int64)     { n.offset = x }
func (n *node) Esc() uint16           { return n.esc }
func (n *node) SetEsc(x uint16)       { n.esc = x }
func (n *node) Op() Op                { return n.op }
func (n *node) Init() Nodes           { return n.init }
func (n *node) SetInit(x Nodes)       { n.init = x }
func (n *node) PtrInit() *Nodes       { return &n.init }
func (n *node) Body() Nodes           { return n.body }
func (n *node) SetBody(x Nodes)       { n.body = x }
func (n *node) PtrBody() *Nodes       { return &n.body }
func (n *node) List() Nodes           { return n.list }
func (n *node) SetList(x Nodes)       { n.list = x }
func (n *node) PtrList() *Nodes       { return &n.list }
func (n *node) Rlist() Nodes          { return n.rlist }
func (n *node) SetRlist(x Nodes)      { n.rlist = x }
func (n *node) PtrRlist() *Nodes      { return &n.rlist }

func (n *node) SetOp(op Op) {
	if !okForNod[op] {
		panic("cannot node.SetOp " + op.String())
	}
	n.op = op
}

func (n *node) ResetAux() {
	n.aux = 0
}

func (n *node) SubOp() Op {
	switch n.Op() {
	case OASOP, ONAME:
	default:
		base.Fatalf("unexpected op: %v", n.Op())
	}
	return Op(n.aux)
}

func (n *node) SetSubOp(op Op) {
	switch n.Op() {
	case OASOP, ONAME:
	default:
		base.Fatalf("unexpected op: %v", n.Op())
	}
	n.aux = uint8(op)
}

func (n *node) IndexMapLValue() bool {
	if n.Op() != OINDEXMAP {
		base.Fatalf("unexpected op: %v", n.Op())
	}
	return n.aux != 0
}

func (n *node) SetIndexMapLValue(b bool) {
	if n.Op() != OINDEXMAP {
		base.Fatalf("unexpected op: %v", n.Op())
	}
	if b {
		n.aux = 1
	} else {
		n.aux = 0
	}
}

func (n *node) TChanDir() types.ChanDir {
	if n.Op() != OTCHAN {
		base.Fatalf("unexpected op: %v", n.Op())
	}
	return types.ChanDir(n.aux)
}

func (n *node) SetTChanDir(dir types.ChanDir) {
	if n.Op() != OTCHAN {
		base.Fatalf("unexpected op: %v", n.Op())
	}
	n.aux = uint8(dir)
}

func IsSynthetic(n Node) bool {
	name := n.Sym().Name
	return name[0] == '.' || name[0] == '~'
}

// IsAutoTmp indicates if n was created by the compiler as a temporary,
// based on the setting of the .AutoTemp flag in n's Name.
func IsAutoTmp(n Node) bool {
	if n == nil || n.Op() != ONAME {
		return false
	}
	return n.Name().AutoTemp()
}

const (
	nodeClass, _     = iota, 1 << iota // PPARAM, PAUTO, PEXTERN, etc; three bits; first in the list because frequently accessed
	_, _                               // second nodeClass bit
	_, _                               // third nodeClass bit
	nodeWalkdef, _                     // tracks state during typecheckdef; 2 == loop detected; two bits
	_, _                               // second nodeWalkdef bit
	nodeTypecheck, _                   // tracks state during typechecking; 2 == loop detected; two bits
	_, _                               // second nodeTypecheck bit
	nodeInitorder, _                   // tracks state during init1; two bits
	_, _                               // second nodeInitorder bit
	_, nodeHasBreak
	_, nodeNoInline  // used internally by inliner to indicate that a function call should not be inlined; set for OCALLFUNC and OCALLMETH only
	_, nodeImplicit  // implicit OADDR or ODEREF; ++/-- statement represented as OASOP
	_, nodeIsDDD     // is the argument variadic
	_, nodeDiag      // already printed error about this
	_, nodeColas     // OAS resulting from :=
	_, nodeNonNil    // guaranteed to be non-nil
	_, nodeTransient // storage can be reused immediately after this statement
	_, nodeBounded   // bounds check unnecessary
	_, nodeHasCall   // expression contains a function call
	_, nodeLikely    // if statement condition likely
	_, nodeHasVal    // node.E contains a Val
	_, nodeHasOpt    // node.E contains an Opt
	_, nodeEmbedded  // ODCLFIELD embedded type
)

func (n *node) Class() Class     { return Class(n.flags.get3(nodeClass)) }
func (n *node) Walkdef() uint8   { return n.flags.get2(nodeWalkdef) }
func (n *node) Typecheck() uint8 { return n.flags.get2(nodeTypecheck) }
func (n *node) Initorder() uint8 { return n.flags.get2(nodeInitorder) }

func (n *node) HasBreak() bool  { return n.flags&nodeHasBreak != 0 }
func (n *node) NoInline() bool  { return n.flags&nodeNoInline != 0 }
func (n *node) Implicit() bool  { return n.flags&nodeImplicit != 0 }
func (n *node) IsDDD() bool     { return n.flags&nodeIsDDD != 0 }
func (n *node) Diag() bool      { return n.flags&nodeDiag != 0 }
func (n *node) Colas() bool     { return n.flags&nodeColas != 0 }
func (n *node) NonNil() bool    { return n.flags&nodeNonNil != 0 }
func (n *node) Transient() bool { return n.flags&nodeTransient != 0 }
func (n *node) Bounded() bool   { return n.flags&nodeBounded != 0 }
func (n *node) HasCall() bool   { return n.flags&nodeHasCall != 0 }
func (n *node) Likely() bool    { return n.flags&nodeLikely != 0 }
func (n *node) HasVal() bool    { return n.flags&nodeHasVal != 0 }
func (n *node) hasOpt() bool    { return n.flags&nodeHasOpt != 0 }
func (n *node) Embedded() bool  { return n.flags&nodeEmbedded != 0 }

func (n *node) SetClass(b Class)     { n.flags.set3(nodeClass, uint8(b)) }
func (n *node) SetWalkdef(b uint8)   { n.flags.set2(nodeWalkdef, b) }
func (n *node) SetTypecheck(b uint8) { n.flags.set2(nodeTypecheck, b) }
func (n *node) SetInitorder(b uint8) { n.flags.set2(nodeInitorder, b) }

func (n *node) SetHasBreak(b bool)  { n.flags.set(nodeHasBreak, b) }
func (n *node) SetNoInline(b bool)  { n.flags.set(nodeNoInline, b) }
func (n *node) SetImplicit(b bool)  { n.flags.set(nodeImplicit, b) }
func (n *node) SetIsDDD(b bool)     { n.flags.set(nodeIsDDD, b) }
func (n *node) SetDiag(b bool)      { n.flags.set(nodeDiag, b) }
func (n *node) SetColas(b bool)     { n.flags.set(nodeColas, b) }
func (n *node) SetTransient(b bool) { n.flags.set(nodeTransient, b) }
func (n *node) SetHasCall(b bool)   { n.flags.set(nodeHasCall, b) }
func (n *node) SetLikely(b bool)    { n.flags.set(nodeLikely, b) }
func (n *node) setHasVal(b bool)    { n.flags.set(nodeHasVal, b) }
func (n *node) setHasOpt(b bool)    { n.flags.set(nodeHasOpt, b) }
func (n *node) SetEmbedded(b bool)  { n.flags.set(nodeEmbedded, b) }

// MarkNonNil marks a pointer n as being guaranteed non-nil,
// on all code paths, at all times.
// During conversion to SSA, non-nil pointers won't have nil checks
// inserted before dereferencing. See state.exprPtr.
func (n *node) MarkNonNil() {
	if !n.Type().IsPtr() && !n.Type().IsUnsafePtr() {
		base.Fatalf("MarkNonNil(%v), type %v", n, n.Type())
	}
	n.flags.set(nodeNonNil, true)
}

// SetBounded indicates whether operation n does not need safety checks.
// When n is an index or slice operation, n does not need bounds checks.
// When n is a dereferencing operation, n does not need nil checks.
// When n is a makeslice+copy operation, n does not need length and cap checks.
func (n *node) SetBounded(b bool) {
	switch n.Op() {
	case OINDEX, OSLICE, OSLICEARR, OSLICE3, OSLICE3ARR, OSLICESTR:
		// No bounds checks needed.
	case ODOTPTR, ODEREF:
		// No nil check needed.
	case OMAKESLICECOPY:
		// No length and cap checks needed
		// since new slice and copied over slice data have same length.
	default:
		base.Fatalf("SetBounded(%v)", n)
	}
	n.flags.set(nodeBounded, b)
}

// Opt returns the optimizer data for the node.
func (n *node) Opt() interface{} {
	if !n.hasOpt() {
		return nil
	}
	return n.e
}

// SetOpt sets the optimizer data for the node, which must not have been used with SetVal.
// SetOpt(nil) is ignored for Vals to simplify call sites that are clearing Opts.
func (n *node) SetOpt(x interface{}) {
	if x == nil {
		if n.hasOpt() {
			n.setHasOpt(false)
			n.e = nil
		}
		return
	}
	if n.HasVal() {
		base.Flag.LowerH = 1
		Dump("have Val", n)
		base.Fatalf("have Val")
	}
	n.setHasOpt(true)
	n.e = x
}

func (n *node) Iota() int64 {
	return n.Offset()
}

func (n *node) SetIota(x int64) {
	n.SetOffset(x)
}

// mayBeShared reports whether n may occur in multiple places in the AST.
// Extra care must be taken when mutating such a node.
func MayBeShared(n Node) bool {
	switch n.Op() {
	case ONAME, OLITERAL, ONIL, OTYPE:
		return true
	}
	return false
}

// funcname returns the name (without the package) of the function n.
func FuncName(n Node) string {
	if n == nil || n.Func() == nil || n.Func().Nname == nil {
		return "<nil>"
	}
	return n.Func().Nname.Sym().Name
}

// pkgFuncName returns the name of the function referenced by n, with package prepended.
// This differs from the compiler's internal convention where local functions lack a package
// because the ultimate consumer of this is a human looking at an IDE; package is only empty
// if the compilation package is actually the empty string.
func PkgFuncName(n Node) string {
	var s *types.Sym
	if n == nil {
		return "<nil>"
	}
	if n.Op() == ONAME {
		s = n.Sym()
	} else {
		if n.Func() == nil || n.Func().Nname == nil {
			return "<nil>"
		}
		s = n.Func().Nname.Sym()
	}
	pkg := s.Pkg

	p := base.Ctxt.Pkgpath
	if pkg != nil && pkg.Path != "" {
		p = pkg.Path
	}
	if p == "" {
		return s.Name
	}
	return p + "." + s.Name
}

// The compiler needs *Node to be assignable to cmd/compile/internal/ssa.Sym.
func (n *node) CanBeAnSSASym() {
}

// A Func corresponds to a single function in a Go program
// (and vice versa: each function is denoted by exactly one *Func).
//
// There are multiple nodes that represent a Func in the IR.
//
// The ONAME node (Func.Name) is used for plain references to it.
// The ODCLFUNC node (Func.Decl) is used for its declaration code.
// The OCLOSURE node (Func.Closure) is used for a reference to a
// function literal.
//
// A Func for an imported function will have only an ONAME node.
// A declared function or method has an ONAME and an ODCLFUNC.
// A function literal is represented directly by an OCLOSURE, but it also
// has an ODCLFUNC (and a matching ONAME) representing the compiled
// underlying form of the closure, which accesses the captured variables
// using a special data structure passed in a register.
//
// A method declaration is represented like functions, except f.Sym
// will be the qualified method name (e.g., "T.m") and
// f.Func.Shortname is the bare method name (e.g., "m").
//
// A method expression (T.M) is represented as an OMETHEXPR node,
// in which n.Left and n.Right point to the type and method, respectively.
// Each distinct mention of a method expression in the source code
// constructs a fresh node.
//
// A method value (t.M) is represented by ODOTMETH/ODOTINTER
// when it is called directly and by OCALLPART otherwise.
// These are like method expressions, except that for ODOTMETH/ODOTINTER,
// the method name is stored in Sym instead of Right.
// Each OCALLPART ends up being implemented as a new
// function, a bit like a closure, with its own ODCLFUNC.
// The OCALLPART has uses n.Func to record the linkage to
// the generated ODCLFUNC (as n.Func.Decl), but there is no
// pointer from the Func back to the OCALLPART.
type Func struct {
	Nname    Node // ONAME node
	Decl     Node // ODCLFUNC node
	OClosure Node // OCLOSURE node

	Shortname *types.Sym

	// Extra entry code for the function. For example, allocate and initialize
	// memory for escaping parameters.
	Enter Nodes
	Exit  Nodes
	// ONAME nodes for all params/locals for this func/closure, does NOT
	// include closurevars until transformclosure runs.
	Dcl []Node

	ClosureEnter  Nodes // list of ONAME nodes of captured variables
	ClosureType   Node  // closure representation type
	ClosureCalled bool  // closure is only immediately called
	ClosureVars   Nodes // closure params; each has closurevar set

	// Parents records the parent scope of each scope within a
	// function. The root scope (0) has no parent, so the i'th
	// scope's parent is stored at Parents[i-1].
	Parents []ScopeID

	// Marks records scope boundary changes.
	Marks []Mark

	// Closgen tracks how many closures have been generated within
	// this function. Used by closurename for creating unique
	// function names.
	Closgen int

	FieldTrack map[*types.Sym]struct{}
	DebugInfo  interface{}
	LSym       *obj.LSym

	Inl *Inline

	Label int32 // largest auto-generated label in this function

	Endlineno src.XPos
	WBPos     src.XPos // position of first write barrier; see SetWBPos

	Pragma PragmaFlag // go:xxx function annotations

	flags      bitset16
	NumDefers  int // number of defer calls in the function
	NumReturns int // number of explicit returns in the function

	// nwbrCalls records the LSyms of functions called by this
	// function for go:nowritebarrierrec analysis. Only filled in
	// if nowritebarrierrecCheck != nil.
	NWBRCalls *[]SymAndPos
}

// An Inline holds fields used for function bodies that can be inlined.
type Inline struct {
	Cost int32 // heuristic cost of inlining this function

	// Copies of Func.Dcl and Nbody for use during inlining.
	Dcl  []Node
	Body []Node
}

// A Mark represents a scope boundary.
type Mark struct {
	// Pos is the position of the token that marks the scope
	// change.
	Pos src.XPos

	// Scope identifies the innermost scope to the right of Pos.
	Scope ScopeID
}

// A ScopeID represents a lexical scope within a function.
type ScopeID int32

const (
	funcDupok         = 1 << iota // duplicate definitions ok
	funcWrapper                   // is method wrapper
	funcNeedctxt                  // function uses context register (has closure variables)
	funcReflectMethod             // function calls reflect.Type.Method or MethodByName
	// true if closure inside a function; false if a simple function or a
	// closure in a global variable initialization
	funcIsHiddenClosure
	funcHasDefer                 // contains a defer statement
	funcNilCheckDisabled         // disable nil checks when compiling this function
	funcInlinabilityChecked      // inliner has already determined whether the function is inlinable
	funcExportInline             // include inline body in export data
	funcInstrumentBody           // add race/msan instrumentation during SSA construction
	funcOpenCodedDeferDisallowed // can't do open-coded defers
)

func (f *Func) Dupok() bool                    { return f.flags&funcDupok != 0 }
func (f *Func) Wrapper() bool                  { return f.flags&funcWrapper != 0 }
func (f *Func) Needctxt() bool                 { return f.flags&funcNeedctxt != 0 }
func (f *Func) ReflectMethod() bool            { return f.flags&funcReflectMethod != 0 }
func (f *Func) IsHiddenClosure() bool          { return f.flags&funcIsHiddenClosure != 0 }
func (f *Func) HasDefer() bool                 { return f.flags&funcHasDefer != 0 }
func (f *Func) NilCheckDisabled() bool         { return f.flags&funcNilCheckDisabled != 0 }
func (f *Func) InlinabilityChecked() bool      { return f.flags&funcInlinabilityChecked != 0 }
func (f *Func) ExportInline() bool             { return f.flags&funcExportInline != 0 }
func (f *Func) InstrumentBody() bool           { return f.flags&funcInstrumentBody != 0 }
func (f *Func) OpenCodedDeferDisallowed() bool { return f.flags&funcOpenCodedDeferDisallowed != 0 }

func (f *Func) SetDupok(b bool)                    { f.flags.set(funcDupok, b) }
func (f *Func) SetWrapper(b bool)                  { f.flags.set(funcWrapper, b) }
func (f *Func) SetNeedctxt(b bool)                 { f.flags.set(funcNeedctxt, b) }
func (f *Func) SetReflectMethod(b bool)            { f.flags.set(funcReflectMethod, b) }
func (f *Func) SetIsHiddenClosure(b bool)          { f.flags.set(funcIsHiddenClosure, b) }
func (f *Func) SetHasDefer(b bool)                 { f.flags.set(funcHasDefer, b) }
func (f *Func) SetNilCheckDisabled(b bool)         { f.flags.set(funcNilCheckDisabled, b) }
func (f *Func) SetInlinabilityChecked(b bool)      { f.flags.set(funcInlinabilityChecked, b) }
func (f *Func) SetExportInline(b bool)             { f.flags.set(funcExportInline, b) }
func (f *Func) SetInstrumentBody(b bool)           { f.flags.set(funcInstrumentBody, b) }
func (f *Func) SetOpenCodedDeferDisallowed(b bool) { f.flags.set(funcOpenCodedDeferDisallowed, b) }

func (f *Func) SetWBPos(pos src.XPos) {
	if base.Debug.WB != 0 {
		base.WarnfAt(pos, "write barrier")
	}
	if !f.WBPos.IsKnown() {
		f.WBPos = pos
	}
}

//go:generate stringer -type=Op -trimprefix=O

type Op uint8

// Node ops.
const (
	OXXX Op = iota

	// names
	ONAME // var or func name
	// Unnamed arg or return value: f(int, string) (int, error) { etc }
	// Also used for a qualified package identifier that hasn't been resolved yet.
	ONONAME
	OTYPE    // type name
	OPACK    // import
	OLITERAL // literal
	ONIL     // nil

	// expressions
	OADD          // Left + Right
	OSUB          // Left - Right
	OOR           // Left | Right
	OXOR          // Left ^ Right
	OADDSTR       // +{List} (string addition, list elements are strings)
	OADDR         // &Left
	OANDAND       // Left && Right
	OAPPEND       // append(List); after walk, Left may contain elem type descriptor
	OBYTES2STR    // Type(Left) (Type is string, Left is a []byte)
	OBYTES2STRTMP // Type(Left) (Type is string, Left is a []byte, ephemeral)
	ORUNES2STR    // Type(Left) (Type is string, Left is a []rune)
	OSTR2BYTES    // Type(Left) (Type is []byte, Left is a string)
	OSTR2BYTESTMP // Type(Left) (Type is []byte, Left is a string, ephemeral)
	OSTR2RUNES    // Type(Left) (Type is []rune, Left is a string)
	// Left = Right or (if Colas=true) Left := Right
	// If Colas, then Ninit includes a DCL node for Left.
	OAS
	// List = Rlist (x, y, z = a, b, c) or (if Colas=true) List := Rlist
	// If Colas, then Ninit includes DCL nodes for List
	OAS2
	OAS2DOTTYPE // List = Right (x, ok = I.(int))
	OAS2FUNC    // List = Right (x, y = f())
	OAS2MAPR    // List = Right (x, ok = m["foo"])
	OAS2RECV    // List = Right (x, ok = <-c)
	OASOP       // Left Etype= Right (x += y)
	OCALL       // Left(List) (function call, method call or type conversion)

	// OCALLFUNC, OCALLMETH, and OCALLINTER have the same structure.
	// Prior to walk, they are: Left(List), where List is all regular arguments.
	// After walk, List is a series of assignments to temporaries,
	// and Rlist is an updated set of arguments.
	// Nbody is all OVARLIVE nodes that are attached to OCALLxxx.
	// TODO(josharian/khr): Use Ninit instead of List for the assignments to temporaries. See CL 114797.
	OCALLFUNC  // Left(List/Rlist) (function call f(args))
	OCALLMETH  // Left(List/Rlist) (direct method call x.Method(args))
	OCALLINTER // Left(List/Rlist) (interface method call x.Method(args))
	OCALLPART  // Left.Right (method expression x.Method, not called)
	OCAP       // cap(Left)
	OCLOSE     // close(Left)
	OCLOSURE   // func Type { Func.Closure.Nbody } (func literal)
	OCOMPLIT   // Right{List} (composite literal, not yet lowered to specific form)
	OMAPLIT    // Type{List} (composite literal, Type is map)
	OSTRUCTLIT // Type{List} (composite literal, Type is struct)
	OARRAYLIT  // Type{List} (composite literal, Type is array)
	OSLICELIT  // Type{List} (composite literal, Type is slice) Right.Int64() = slice length.
	OPTRLIT    // &Left (left is composite literal)
	OCONV      // Type(Left) (type conversion)
	OCONVIFACE // Type(Left) (type conversion, to interface)
	OCONVNOP   // Type(Left) (type conversion, no effect)
	OCOPY      // copy(Left, Right)
	ODCL       // var Left (declares Left of type Left.Type)

	// Used during parsing but don't last.
	ODCLFUNC  // func f() or func (r) f()
	ODCLFIELD // struct field, interface field, or func/method argument/return value.
	ODCLCONST // const pi = 3.14
	ODCLTYPE  // type Int int or type Int = int

	ODELETE        // delete(List)
	ODOT           // Left.Sym (Left is of struct type)
	ODOTPTR        // Left.Sym (Left is of pointer to struct type)
	ODOTMETH       // Left.Sym (Left is non-interface, Right is method name)
	ODOTINTER      // Left.Sym (Left is interface, Right is method name)
	OXDOT          // Left.Sym (before rewrite to one of the preceding)
	ODOTTYPE       // Left.Right or Left.Type (.Right during parsing, .Type once resolved); after walk, .Right contains address of interface type descriptor and .Right.Right contains address of concrete type descriptor
	ODOTTYPE2      // Left.Right or Left.Type (.Right during parsing, .Type once resolved; on rhs of OAS2DOTTYPE); after walk, .Right contains address of interface type descriptor
	OEQ            // Left == Right
	ONE            // Left != Right
	OLT            // Left < Right
	OLE            // Left <= Right
	OGE            // Left >= Right
	OGT            // Left > Right
	ODEREF         // *Left
	OINDEX         // Left[Right] (index of array or slice)
	OINDEXMAP      // Left[Right] (index of map)
	OKEY           // Left:Right (key:value in struct/array/map literal)
	OSTRUCTKEY     // Sym:Left (key:value in struct literal, after type checking)
	OLEN           // len(Left)
	OMAKE          // make(List) (before type checking converts to one of the following)
	OMAKECHAN      // make(Type, Left) (type is chan)
	OMAKEMAP       // make(Type, Left) (type is map)
	OMAKESLICE     // make(Type, Left, Right) (type is slice)
	OMAKESLICECOPY // makeslicecopy(Type, Left, Right) (type is slice; Left is length and Right is the copied from slice)
	// OMAKESLICECOPY is created by the order pass and corresponds to:
	//  s = make(Type, Left); copy(s, Right)
	//
	// Bounded can be set on the node when Left == len(Right) is known at compile time.
	//
	// This node is created so the walk pass can optimize this pattern which would
	// otherwise be hard to detect after the order pass.
	OMUL         // Left * Right
	ODIV         // Left / Right
	OMOD         // Left % Right
	OLSH         // Left << Right
	ORSH         // Left >> Right
	OAND         // Left & Right
	OANDNOT      // Left &^ Right
	ONEW         // new(Left); corresponds to calls to new in source code
	ONEWOBJ      // runtime.newobject(n.Type); introduced by walk; Left is type descriptor
	ONOT         // !Left
	OBITNOT      // ^Left
	OPLUS        // +Left
	ONEG         // -Left
	OOROR        // Left || Right
	OPANIC       // panic(Left)
	OPRINT       // print(List)
	OPRINTN      // println(List)
	OPAREN       // (Left)
	OSEND        // Left <- Right
	OSLICE       // Left[List[0] : List[1]] (Left is untypechecked or slice)
	OSLICEARR    // Left[List[0] : List[1]] (Left is array)
	OSLICESTR    // Left[List[0] : List[1]] (Left is string)
	OSLICE3      // Left[List[0] : List[1] : List[2]] (Left is untypedchecked or slice)
	OSLICE3ARR   // Left[List[0] : List[1] : List[2]] (Left is array)
	OSLICEHEADER // sliceheader{Left, List[0], List[1]} (Left is unsafe.Pointer, List[0] is length, List[1] is capacity)
	ORECOVER     // recover()
	ORECV        // <-Left
	ORUNESTR     // Type(Left) (Type is string, Left is rune)
	OSELRECV     // Left = <-Right.Left: (appears as .Left of OCASE; Right.Op == ORECV)
	OSELRECV2    // List = <-Right.Left: (appears as .Left of OCASE; count(List) == 2, Right.Op == ORECV)
	OIOTA        // iota
	OREAL        // real(Left)
	OIMAG        // imag(Left)
	OCOMPLEX     // complex(Left, Right) or complex(List[0]) where List[0] is a 2-result function call
	OALIGNOF     // unsafe.Alignof(Left)
	OOFFSETOF    // unsafe.Offsetof(Left)
	OSIZEOF      // unsafe.Sizeof(Left)
	OMETHEXPR    // method expression

	// statements
	OBLOCK // { List } (block of code)
	OBREAK // break [Sym]
	// OCASE:  case List: Nbody (List==nil means default)
	//   For OTYPESW, List is a OTYPE node for the specified type (or OLITERAL
	//   for nil), and, if a type-switch variable is specified, Rlist is an
	//   ONAME for the version of the type-switch variable with the specified
	//   type.
	OCASE
	OCONTINUE // continue [Sym]
	ODEFER    // defer Left (Left must be call)
	OEMPTY    // no-op (empty statement)
	OFALL     // fallthrough
	OFOR      // for Ninit; Left; Right { Nbody }
	// OFORUNTIL is like OFOR, but the test (Left) is applied after the body:
	// 	Ninit
	// 	top: { Nbody }   // Execute the body at least once
	// 	cont: Right
	// 	if Left {        // And then test the loop condition
	// 		List     // Before looping to top, execute List
	// 		goto top
	// 	}
	// OFORUNTIL is created by walk. There's no way to write this in Go code.
	OFORUNTIL
	OGOTO   // goto Sym
	OIF     // if Ninit; Left { Nbody } else { Rlist }
	OLABEL  // Sym:
	OGO     // go Left (Left must be call)
	ORANGE  // for List = range Right { Nbody }
	ORETURN // return List
	OSELECT // select { List } (List is list of OCASE)
	OSWITCH // switch Ninit; Left { List } (List is a list of OCASE)
	// OTYPESW:  Left := Right.(type) (appears as .Left of OSWITCH)
	//   Left is nil if there is no type-switch variable
	OTYPESW

	// types
	OTCHAN   // chan int
	OTMAP    // map[string]int
	OTSTRUCT // struct{}
	OTINTER  // interface{}
	// OTFUNC: func() - Left is receiver field, List is list of param fields, Rlist is
	// list of result fields.
	OTFUNC
	OTARRAY // []int, [8]int, [N]int or [...]int

	// misc
	ODDD        // func f(args ...int) or f(l...) or var a = [...]int{0, 1, 2}.
	OINLCALL    // intermediary representation of an inlined call.
	OEFACE      // itable and data words of an empty-interface value.
	OITAB       // itable word of an interface value.
	OIDATA      // data word of an interface value in Left
	OSPTR       // base pointer of a slice or string.
	OCLOSUREVAR // variable reference at beginning of closure function
	OCFUNC      // reference to c function pointer (not go func value)
	OCHECKNIL   // emit code to ensure pointer/interface not nil
	OVARDEF     // variable is about to be fully initialized
	OVARKILL    // variable is dead
	OVARLIVE    // variable is alive
	ORESULT     // result of a function call; Xoffset is stack offset
	OINLMARK    // start of an inlined body, with file/line of caller. Xoffset is an index into the inline tree.

	// arch-specific opcodes
	ORETJMP // return to other function
	OGETG   // runtime.getg() (read g pointer)

	OEND
)

// Nodes is a pointer to a slice of *Node.
// For fields that are not used in most nodes, this is used instead of
// a slice to save space.
type Nodes struct{ slice *[]Node }

// immutableEmptyNodes is an immutable, empty Nodes list.
// The methods that would modify it panic instead.
var immutableEmptyNodes = Nodes{}

// asNodes returns a slice of *Node as a Nodes value.
func AsNodes(s []Node) Nodes {
	return Nodes{&s}
}

// Slice returns the entries in Nodes as a slice.
// Changes to the slice entries (as in s[i] = n) will be reflected in
// the Nodes.
func (n Nodes) Slice() []Node {
	if n.slice == nil {
		return nil
	}
	return *n.slice
}

// Len returns the number of entries in Nodes.
func (n Nodes) Len() int {
	if n.slice == nil {
		return 0
	}
	return len(*n.slice)
}

// Index returns the i'th element of Nodes.
// It panics if n does not have at least i+1 elements.
func (n Nodes) Index(i int) Node {
	return (*n.slice)[i]
}

// First returns the first element of Nodes (same as n.Index(0)).
// It panics if n has no elements.
func (n Nodes) First() Node {
	return (*n.slice)[0]
}

// Second returns the second element of Nodes (same as n.Index(1)).
// It panics if n has fewer than two elements.
func (n Nodes) Second() Node {
	return (*n.slice)[1]
}

func (n *Nodes) mutate() {
	if n == &immutableEmptyNodes {
		panic("immutable Nodes.Set")
	}
}

// Set sets n to a slice.
// This takes ownership of the slice.
func (n *Nodes) Set(s []Node) {
	if n == &immutableEmptyNodes {
		if len(s) == 0 {
			// Allow immutableEmptyNodes.Set(nil) (a no-op).
			return
		}
		n.mutate()
	}
	if len(s) == 0 {
		n.slice = nil
	} else {
		// Copy s and take address of t rather than s to avoid
		// allocation in the case where len(s) == 0 (which is
		// over 3x more common, dynamically, for make.bash).
		t := s
		n.slice = &t
	}
}

// Set1 sets n to a slice containing a single node.
func (n *Nodes) Set1(n1 Node) {
	n.mutate()
	n.slice = &[]Node{n1}
}

// Set2 sets n to a slice containing two nodes.
func (n *Nodes) Set2(n1, n2 Node) {
	n.mutate()
	n.slice = &[]Node{n1, n2}
}

// Set3 sets n to a slice containing three nodes.
func (n *Nodes) Set3(n1, n2, n3 Node) {
	n.mutate()
	n.slice = &[]Node{n1, n2, n3}
}

// MoveNodes sets n to the contents of n2, then clears n2.
func (n *Nodes) MoveNodes(n2 *Nodes) {
	n.mutate()
	n.slice = n2.slice
	n2.slice = nil
}

// SetIndex sets the i'th element of Nodes to node.
// It panics if n does not have at least i+1 elements.
func (n Nodes) SetIndex(i int, node Node) {
	(*n.slice)[i] = node
}

// SetFirst sets the first element of Nodes to node.
// It panics if n does not have at least one elements.
func (n Nodes) SetFirst(node Node) {
	(*n.slice)[0] = node
}

// SetSecond sets the second element of Nodes to node.
// It panics if n does not have at least two elements.
func (n Nodes) SetSecond(node Node) {
	(*n.slice)[1] = node
}

// Addr returns the address of the i'th element of Nodes.
// It panics if n does not have at least i+1 elements.
func (n Nodes) Addr(i int) *Node {
	return &(*n.slice)[i]
}

// Append appends entries to Nodes.
func (n *Nodes) Append(a ...Node) {
	if len(a) == 0 {
		return
	}
	n.mutate()
	if n.slice == nil {
		s := make([]Node, len(a))
		copy(s, a)
		n.slice = &s
		return
	}
	*n.slice = append(*n.slice, a...)
}

// Prepend prepends entries to Nodes.
// If a slice is passed in, this will take ownership of it.
func (n *Nodes) Prepend(a ...Node) {
	if len(a) == 0 {
		return
	}
	n.mutate()
	if n.slice == nil {
		n.slice = &a
	} else {
		*n.slice = append(a, *n.slice...)
	}
}

// AppendNodes appends the contents of *n2 to n, then clears n2.
func (n *Nodes) AppendNodes(n2 *Nodes) {
	n.mutate()
	switch {
	case n2.slice == nil:
	case n.slice == nil:
		n.slice = n2.slice
	default:
		*n.slice = append(*n.slice, *n2.slice...)
	}
	n2.slice = nil
}

// inspect invokes f on each node in an AST in depth-first order.
// If f(n) returns false, inspect skips visiting n's children.
func Inspect(n Node, f func(Node) bool) {
	if n == nil || !f(n) {
		return
	}
	InspectList(n.Init(), f)
	Inspect(n.Left(), f)
	Inspect(n.Right(), f)
	InspectList(n.List(), f)
	InspectList(n.Body(), f)
	InspectList(n.Rlist(), f)
}

func InspectList(l Nodes, f func(Node) bool) {
	for _, n := range l.Slice() {
		Inspect(n, f)
	}
}

// nodeQueue is a FIFO queue of *Node. The zero value of nodeQueue is
// a ready-to-use empty queue.
type NodeQueue struct {
	ring       []Node
	head, tail int
}

// empty reports whether q contains no Nodes.
func (q *NodeQueue) Empty() bool {
	return q.head == q.tail
}

// pushRight appends n to the right of the queue.
func (q *NodeQueue) PushRight(n Node) {
	if len(q.ring) == 0 {
		q.ring = make([]Node, 16)
	} else if q.head+len(q.ring) == q.tail {
		// Grow the ring.
		nring := make([]Node, len(q.ring)*2)
		// Copy the old elements.
		part := q.ring[q.head%len(q.ring):]
		if q.tail-q.head <= len(part) {
			part = part[:q.tail-q.head]
			copy(nring, part)
		} else {
			pos := copy(nring, part)
			copy(nring[pos:], q.ring[:q.tail%len(q.ring)])
		}
		q.ring, q.head, q.tail = nring, 0, q.tail-q.head
	}

	q.ring[q.tail%len(q.ring)] = n
	q.tail++
}

// popLeft pops a node from the left of the queue. It panics if q is
// empty.
func (q *NodeQueue) PopLeft() Node {
	if q.Empty() {
		panic("dequeue empty")
	}
	n := q.ring[q.head%len(q.ring)]
	q.head++
	return n
}

// NodeSet is a set of Nodes.
type NodeSet map[Node]struct{}

// Has reports whether s contains n.
func (s NodeSet) Has(n Node) bool {
	_, isPresent := s[n]
	return isPresent
}

// Add adds n to s.
func (s *NodeSet) Add(n Node) {
	if *s == nil {
		*s = make(map[Node]struct{})
	}
	(*s)[n] = struct{}{}
}

// Sorted returns s sorted according to less.
func (s NodeSet) Sorted(less func(Node, Node) bool) []Node {
	var res []Node
	for n := range s {
		res = append(res, n)
	}
	sort.Slice(res, func(i, j int) bool { return less(res[i], res[j]) })
	return res
}

type PragmaFlag int16

const (
	// Func pragmas.
	Nointerface    PragmaFlag = 1 << iota
	Noescape                  // func parameters don't escape
	Norace                    // func must not have race detector annotations
	Nosplit                   // func should not execute on separate stack
	Noinline                  // func should not be inlined
	NoCheckPtr                // func should not be instrumented by checkptr
	CgoUnsafeArgs             // treat a pointer to one arg as a pointer to them all
	UintptrEscapes            // pointers converted to uintptr escape

	// Runtime-only func pragmas.
	// See ../../../../runtime/README.md for detailed descriptions.
	Systemstack        // func must run on system stack
	Nowritebarrier     // emit compiler error instead of write barrier
	Nowritebarrierrec  // error on write barrier in this or recursive callees
	Yeswritebarrierrec // cancels Nowritebarrierrec in this function and callees

	// Runtime and cgo type pragmas
	NotInHeap // values of this type must not be heap allocated

	// Go command pragmas
	GoBuildPragma
)

type SymAndPos struct {
	Sym *obj.LSym // LSym of callee
	Pos src.XPos  // line of call
}

func AsNode(n types.IRNode) Node {
	if n == nil {
		return nil
	}
	return n.(Node)
}

var BlankNode Node

// origSym returns the original symbol written by the user.
func OrigSym(s *types.Sym) *types.Sym {
	if s == nil {
		return nil
	}

	if len(s.Name) > 1 && s.Name[0] == '~' {
		switch s.Name[1] {
		case 'r': // originally an unnamed result
			return nil
		case 'b': // originally the blank identifier _
			// TODO(mdempsky): Does s.Pkg matter here?
			return BlankNode.Sym()
		}
		return s
	}

	if strings.HasPrefix(s.Name, ".anon") {
		// originally an unnamed or _ name (see subr.go: structargs)
		return nil
	}

	return s
}

// SliceBounds returns n's slice bounds: low, high, and max in expr[low:high:max].
// n must be a slice expression. max is nil if n is a simple slice expression.
func (n *node) SliceBounds() (low, high, max Node) {
	if n.List().Len() == 0 {
		return nil, nil, nil
	}

	switch n.Op() {
	case OSLICE, OSLICEARR, OSLICESTR:
		s := n.List().Slice()
		return s[0], s[1], nil
	case OSLICE3, OSLICE3ARR:
		s := n.List().Slice()
		return s[0], s[1], s[2]
	}
	base.Fatalf("SliceBounds op %v: %v", n.Op(), n)
	return nil, nil, nil
}

// SetSliceBounds sets n's slice bounds, where n is a slice expression.
// n must be a slice expression. If max is non-nil, n must be a full slice expression.
func (n *node) SetSliceBounds(low, high, max Node) {
	switch n.Op() {
	case OSLICE, OSLICEARR, OSLICESTR:
		if max != nil {
			base.Fatalf("SetSliceBounds %v given three bounds", n.Op())
		}
		s := n.List().Slice()
		if s == nil {
			if low == nil && high == nil {
				return
			}
			n.PtrList().Set2(low, high)
			return
		}
		s[0] = low
		s[1] = high
		return
	case OSLICE3, OSLICE3ARR:
		s := n.List().Slice()
		if s == nil {
			if low == nil && high == nil && max == nil {
				return
			}
			n.PtrList().Set3(low, high, max)
			return
		}
		s[0] = low
		s[1] = high
		s[2] = max
		return
	}
	base.Fatalf("SetSliceBounds op %v: %v", n.Op(), n)
}

// IsSlice3 reports whether o is a slice3 op (OSLICE3, OSLICE3ARR).
// o must be a slicing op.
func (o Op) IsSlice3() bool {
	switch o {
	case OSLICE, OSLICEARR, OSLICESTR:
		return false
	case OSLICE3, OSLICE3ARR:
		return true
	}
	base.Fatalf("IsSlice3 op %v", o)
	return false
}

func IsConst(n Node, ct constant.Kind) bool {
	return ConstType(n) == ct
}

// rawcopy returns a shallow copy of n.
// Note: copy or sepcopy (rather than rawcopy) is usually the
//       correct choice (see comment with Node.copy, below).
func (n *node) RawCopy() Node {
	copy := *n
	return &copy
}

// A Node may implement the Orig and SetOrig method to
// maintain a pointer to the "unrewritten" form of a Node.
// If a Node does not implement OrigNode, it is its own Orig.
//
// Note that both SepCopy and Copy have definitions compatible
// with a Node that does not implement OrigNode: such a Node
// is its own Orig, and in that case, that's what both want to return
// anyway (SepCopy unconditionally, and Copy only when the input
// is its own Orig as well, but if the output does not implement
// OrigNode, then neither does the input, making the condition true).
type OrigNode interface {
	Node
	Orig() Node
	SetOrig(Node)
}

func Orig(n Node) Node {
	if n, ok := n.(OrigNode); ok {
		o := n.Orig()
		if o == nil {
			Dump("Orig nil", n)
			base.Fatalf("Orig returned nil")
		}
		return o
	}
	return n
}

// sepcopy returns a separate shallow copy of n, with the copy's
// Orig pointing to itself.
func SepCopy(n Node) Node {
	n = n.RawCopy()
	if n, ok := n.(OrigNode); ok {
		n.SetOrig(n)
	}
	return n
}

// copy returns shallow copy of n and adjusts the copy's Orig if
// necessary: In general, if n.Orig points to itself, the copy's
// Orig should point to itself as well. Otherwise, if n is modified,
// the copy's Orig node appears modified, too, and then doesn't
// represent the original node anymore.
// (This caused the wrong complit Op to be used when printing error
// messages; see issues #26855, #27765).
func Copy(n Node) Node {
	copy := n.RawCopy()
	if n, ok := n.(OrigNode); ok && n.Orig() == n {
		copy.(OrigNode).SetOrig(copy)
	}
	return copy
}

// isNil reports whether n represents the universal untyped zero value "nil".
func IsNil(n Node) bool {
	// Check n.Orig because constant propagation may produce typed nil constants,
	// which don't exist in the Go spec.
	return Orig(n).Op() == ONIL
}

func IsBlank(n Node) bool {
	if n == nil {
		return false
	}
	return n.Sym().IsBlank()
}

// IsMethod reports whether n is a method.
// n must be a function or a method.
func IsMethod(n Node) bool {
	return n.Type().Recv() != nil
}

func Nod(op Op, nleft, nright Node) Node {
	return NodAt(base.Pos, op, nleft, nright)
}

func NodAt(pos src.XPos, op Op, nleft, nright Node) Node {
	var n *node
	switch op {
	case ODCLFUNC:
		var x struct {
			n node
			f Func
		}
		n = &x.n
		n.SetFunc(&x.f)
		n.Func().Decl = n
	case OPACK:
		var x struct {
			n node
			m Name
		}
		n = &x.n
		n.SetName(&x.m)
	case OEMPTY:
		return NewEmptyStmt(pos)
	case OBREAK, OCONTINUE, OFALL, OGOTO:
		return NewBranchStmt(pos, op, nil)
	case OLABEL:
		return NewLabelStmt(pos, nil)
	default:
		n = new(node)
	}
	n.SetOp(op)
	n.SetLeft(nleft)
	n.SetRight(nright)
	n.SetPos(pos)
	n.SetOffset(types.BADWIDTH)
	n.SetOrig(n)
	return n
}

var okForNod = [OEND]bool{
	OADD:           true,
	OADDR:          true,
	OADDSTR:        true,
	OALIGNOF:       true,
	OAND:           true,
	OANDAND:        true,
	OANDNOT:        true,
	OAPPEND:        true,
	OARRAYLIT:      true,
	OAS:            true,
	OAS2:           true,
	OAS2DOTTYPE:    true,
	OAS2FUNC:       true,
	OAS2MAPR:       true,
	OAS2RECV:       true,
	OASOP:          true,
	OBITNOT:        true,
	OBLOCK:         true,
	OBYTES2STR:     true,
	OBYTES2STRTMP:  true,
	OCALL:          true,
	OCALLFUNC:      true,
	OCALLINTER:     true,
	OCALLMETH:      true,
	OCALLPART:      true,
	OCAP:           true,
	OCASE:          true,
	OCFUNC:         true,
	OCHECKNIL:      true,
	OCLOSE:         true,
	OCLOSURE:       true,
	OCLOSUREVAR:    true,
	OCOMPLEX:       true,
	OCOMPLIT:       true,
	OCONV:          true,
	OCONVIFACE:     true,
	OCONVNOP:       true,
	OCOPY:          true,
	ODCL:           true,
	ODCLCONST:      true,
	ODCLFIELD:      true,
	ODCLFUNC:       true,
	ODCLTYPE:       true,
	ODDD:           true,
	ODEFER:         true,
	ODELETE:        true,
	ODEREF:         true,
	ODIV:           true,
	ODOT:           true,
	ODOTINTER:      true,
	ODOTMETH:       true,
	ODOTPTR:        true,
	ODOTTYPE:       true,
	ODOTTYPE2:      true,
	OEFACE:         true,
	OEQ:            true,
	OFOR:           true,
	OFORUNTIL:      true,
	OGE:            true,
	OGETG:          true,
	OGO:            true,
	OGT:            true,
	OIDATA:         true,
	OIF:            true,
	OIMAG:          true,
	OINDEX:         true,
	OINDEXMAP:      true,
	OINLCALL:       true,
	OINLMARK:       true,
	OIOTA:          true,
	OITAB:          true,
	OKEY:           true,
	OLABEL:         true,
	OLE:            true,
	OLEN:           true,
	OLITERAL:       true,
	OLSH:           true,
	OLT:            true,
	OMAKE:          true,
	OMAKECHAN:      true,
	OMAKEMAP:       true,
	OMAKESLICE:     true,
	OMAKESLICECOPY: true,
	OMAPLIT:        true,
	OMETHEXPR:      true,
	OMOD:           true,
	OMUL:           true,
	ONAME:          true,
	ONE:            true,
	ONEG:           true,
	ONEW:           true,
	ONEWOBJ:        true,
	ONIL:           true,
	ONONAME:        true,
	ONOT:           true,
	OOFFSETOF:      true,
	OOR:            true,
	OOROR:          true,
	OPACK:          true,
	OPANIC:         true,
	OPAREN:         true,
	OPLUS:          true,
	OPRINT:         true,
	OPRINTN:        true,
	OPTRLIT:        true,
	ORANGE:         true,
	OREAL:          true,
	ORECOVER:       true,
	ORECV:          true,
	ORESULT:        true,
	ORETJMP:        true,
	ORETURN:        true,
	ORSH:           true,
	ORUNES2STR:     true,
	ORUNESTR:       true,
	OSELECT:        true,
	OSELRECV:       true,
	OSELRECV2:      true,
	OSEND:          true,
	OSIZEOF:        true,
	OSLICE:         true,
	OSLICE3:        true,
	OSLICE3ARR:     true,
	OSLICEARR:      true,
	OSLICEHEADER:   true,
	OSLICELIT:      true,
	OSLICESTR:      true,
	OSPTR:          true,
	OSTR2BYTES:     true,
	OSTR2BYTESTMP:  true,
	OSTR2RUNES:     true,
	OSTRUCTKEY:     true,
	OSTRUCTLIT:     true,
	OSUB:           true,
	OSWITCH:        true,
	OTARRAY:        true,
	OTCHAN:         true,
	OTFUNC:         true,
	OTINTER:        true,
	OTMAP:          true,
	OTSTRUCT:       true,
	OTYPE:          true,
	OTYPESW:        true,
	OVARDEF:        true,
	OVARKILL:       true,
	OVARLIVE:       true,
	OXDOT:          true,
	OXOR:           true,
}
