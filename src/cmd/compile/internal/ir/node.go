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
	"cmd/internal/objabi"
	"cmd/internal/src"
)

// A Node is the abstract interface to an IR node.
type INode interface {
	// Formatting
	Format(s fmt.State, verb rune)
	String() string

	// Source position.
	Pos() src.XPos
	SetPos(x src.XPos)

	// For making copies. Mainly used by Copy and SepCopy.
	RawCopy() *Node

	// Abstract graph structure, for generic traversals.
	Op() Op
	SetOp(x Op)
	Orig() *Node
	SetOrig(x *Node)
	SubOp() Op
	SetSubOp(x Op)
	Left() *Node
	SetLeft(x *Node)
	Right() *Node
	SetRight(x *Node)
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
	SliceBounds() (low, high, max *Node)
	SetSliceBounds(low, high, max *Node)
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
	HasOpt() bool
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

var _ INode = (*Node)(nil)

// A Node is a single node in the syntax tree.
// Actually the syntax tree is a syntax DAG, because there is only one
// node with Op=ONAME for a given instance of a variable x.
// The same is true for Op=OTYPE and Op=OLITERAL. See Node.mayBeShared.
type Node struct {
	// Tree structure.
	// Generic recursive walks should follow these fields.
	left  *Node
	right *Node
	init  Nodes
	body  Nodes
	list  Nodes
	rlist Nodes

	// most nodes
	typ  *types.Type
	orig *Node // original form, for printing, and tracking copies of ONAMEs

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

func (n *Node) Left() *Node           { return n.left }
func (n *Node) SetLeft(x *Node)       { n.left = x }
func (n *Node) Right() *Node          { return n.right }
func (n *Node) SetRight(x *Node)      { n.right = x }
func (n *Node) Orig() *Node           { return n.orig }
func (n *Node) SetOrig(x *Node)       { n.orig = x }
func (n *Node) Type() *types.Type     { return n.typ }
func (n *Node) SetType(x *types.Type) { n.typ = x }
func (n *Node) Func() *Func           { return n.fn }
func (n *Node) SetFunc(x *Func)       { n.fn = x }
func (n *Node) Name() *Name           { return n.name }
func (n *Node) SetName(x *Name)       { n.name = x }
func (n *Node) Sym() *types.Sym       { return n.sym }
func (n *Node) SetSym(x *types.Sym)   { n.sym = x }
func (n *Node) Pos() src.XPos         { return n.pos }
func (n *Node) SetPos(x src.XPos)     { n.pos = x }
func (n *Node) Offset() int64         { return n.offset }
func (n *Node) SetOffset(x int64)     { n.offset = x }
func (n *Node) Esc() uint16           { return n.esc }
func (n *Node) SetEsc(x uint16)       { n.esc = x }
func (n *Node) Op() Op                { return n.op }
func (n *Node) SetOp(x Op)            { n.op = x }
func (n *Node) Init() Nodes           { return n.init }
func (n *Node) SetInit(x Nodes)       { n.init = x }
func (n *Node) PtrInit() *Nodes       { return &n.init }
func (n *Node) Body() Nodes           { return n.body }
func (n *Node) SetBody(x Nodes)       { n.body = x }
func (n *Node) PtrBody() *Nodes       { return &n.body }
func (n *Node) List() Nodes           { return n.list }
func (n *Node) SetList(x Nodes)       { n.list = x }
func (n *Node) PtrList() *Nodes       { return &n.list }
func (n *Node) Rlist() Nodes          { return n.rlist }
func (n *Node) SetRlist(x Nodes)      { n.rlist = x }
func (n *Node) PtrRlist() *Nodes      { return &n.rlist }

func (n *Node) ResetAux() {
	n.aux = 0
}

func (n *Node) SubOp() Op {
	switch n.Op() {
	case OASOP, ONAME:
	default:
		base.Fatalf("unexpected op: %v", n.Op())
	}
	return Op(n.aux)
}

func (n *Node) SetSubOp(op Op) {
	switch n.Op() {
	case OASOP, ONAME:
	default:
		base.Fatalf("unexpected op: %v", n.Op())
	}
	n.aux = uint8(op)
}

func (n *Node) IndexMapLValue() bool {
	if n.Op() != OINDEXMAP {
		base.Fatalf("unexpected op: %v", n.Op())
	}
	return n.aux != 0
}

func (n *Node) SetIndexMapLValue(b bool) {
	if n.Op() != OINDEXMAP {
		base.Fatalf("unexpected op: %v", n.Op())
	}
	if b {
		n.aux = 1
	} else {
		n.aux = 0
	}
}

func (n *Node) TChanDir() types.ChanDir {
	if n.Op() != OTCHAN {
		base.Fatalf("unexpected op: %v", n.Op())
	}
	return types.ChanDir(n.aux)
}

func (n *Node) SetTChanDir(dir types.ChanDir) {
	if n.Op() != OTCHAN {
		base.Fatalf("unexpected op: %v", n.Op())
	}
	n.aux = uint8(dir)
}

func IsSynthetic(n *Node) bool {
	name := n.Sym().Name
	return name[0] == '.' || name[0] == '~'
}

// IsAutoTmp indicates if n was created by the compiler as a temporary,
// based on the setting of the .AutoTemp flag in n's Name.
func IsAutoTmp(n *Node) bool {
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

func (n *Node) Class() Class     { return Class(n.flags.get3(nodeClass)) }
func (n *Node) Walkdef() uint8   { return n.flags.get2(nodeWalkdef) }
func (n *Node) Typecheck() uint8 { return n.flags.get2(nodeTypecheck) }
func (n *Node) Initorder() uint8 { return n.flags.get2(nodeInitorder) }

func (n *Node) HasBreak() bool  { return n.flags&nodeHasBreak != 0 }
func (n *Node) NoInline() bool  { return n.flags&nodeNoInline != 0 }
func (n *Node) Implicit() bool  { return n.flags&nodeImplicit != 0 }
func (n *Node) IsDDD() bool     { return n.flags&nodeIsDDD != 0 }
func (n *Node) Diag() bool      { return n.flags&nodeDiag != 0 }
func (n *Node) Colas() bool     { return n.flags&nodeColas != 0 }
func (n *Node) NonNil() bool    { return n.flags&nodeNonNil != 0 }
func (n *Node) Transient() bool { return n.flags&nodeTransient != 0 }
func (n *Node) Bounded() bool   { return n.flags&nodeBounded != 0 }
func (n *Node) HasCall() bool   { return n.flags&nodeHasCall != 0 }
func (n *Node) Likely() bool    { return n.flags&nodeLikely != 0 }
func (n *Node) HasVal() bool    { return n.flags&nodeHasVal != 0 }
func (n *Node) HasOpt() bool    { return n.flags&nodeHasOpt != 0 }
func (n *Node) Embedded() bool  { return n.flags&nodeEmbedded != 0 }

func (n *Node) SetClass(b Class)     { n.flags.set3(nodeClass, uint8(b)) }
func (n *Node) SetWalkdef(b uint8)   { n.flags.set2(nodeWalkdef, b) }
func (n *Node) SetTypecheck(b uint8) { n.flags.set2(nodeTypecheck, b) }
func (n *Node) SetInitorder(b uint8) { n.flags.set2(nodeInitorder, b) }

func (n *Node) SetHasBreak(b bool)  { n.flags.set(nodeHasBreak, b) }
func (n *Node) SetNoInline(b bool)  { n.flags.set(nodeNoInline, b) }
func (n *Node) SetImplicit(b bool)  { n.flags.set(nodeImplicit, b) }
func (n *Node) SetIsDDD(b bool)     { n.flags.set(nodeIsDDD, b) }
func (n *Node) SetDiag(b bool)      { n.flags.set(nodeDiag, b) }
func (n *Node) SetColas(b bool)     { n.flags.set(nodeColas, b) }
func (n *Node) SetTransient(b bool) { n.flags.set(nodeTransient, b) }
func (n *Node) SetHasCall(b bool)   { n.flags.set(nodeHasCall, b) }
func (n *Node) SetLikely(b bool)    { n.flags.set(nodeLikely, b) }
func (n *Node) setHasVal(b bool)    { n.flags.set(nodeHasVal, b) }
func (n *Node) setHasOpt(b bool)    { n.flags.set(nodeHasOpt, b) }
func (n *Node) SetEmbedded(b bool)  { n.flags.set(nodeEmbedded, b) }

// MarkNonNil marks a pointer n as being guaranteed non-nil,
// on all code paths, at all times.
// During conversion to SSA, non-nil pointers won't have nil checks
// inserted before dereferencing. See state.exprPtr.
func (n *Node) MarkNonNil() {
	if !n.Type().IsPtr() && !n.Type().IsUnsafePtr() {
		base.Fatalf("MarkNonNil(%v), type %v", n, n.Type())
	}
	n.flags.set(nodeNonNil, true)
}

// SetBounded indicates whether operation n does not need safety checks.
// When n is an index or slice operation, n does not need bounds checks.
// When n is a dereferencing operation, n does not need nil checks.
// When n is a makeslice+copy operation, n does not need length and cap checks.
func (n *Node) SetBounded(b bool) {
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

// MarkReadonly indicates that n is an ONAME with readonly contents.
func (n *Node) MarkReadonly() {
	if n.Op() != ONAME {
		base.Fatalf("Node.MarkReadonly %v", n.Op())
	}
	n.Name().SetReadonly(true)
	// Mark the linksym as readonly immediately
	// so that the SSA backend can use this information.
	// It will be overridden later during dumpglobls.
	n.Sym().Linksym().Type = objabi.SRODATA
}

// Val returns the constant.Value for the node.
func (n *Node) Val() constant.Value {
	if !n.HasVal() {
		return constant.MakeUnknown()
	}
	return *n.e.(*constant.Value)
}

// SetVal sets the constant.Value for the node,
// which must not have been used with SetOpt.
func (n *Node) SetVal(v constant.Value) {
	if n.HasOpt() {
		base.Flag.LowerH = 1
		Dump("have Opt", n)
		base.Fatalf("have Opt")
	}
	if n.Op() == OLITERAL {
		AssertValidTypeForConst(n.Type(), v)
	}
	n.setHasVal(true)
	n.e = &v
}

// Opt returns the optimizer data for the node.
func (n *Node) Opt() interface{} {
	if !n.HasOpt() {
		return nil
	}
	return n.e
}

// SetOpt sets the optimizer data for the node, which must not have been used with SetVal.
// SetOpt(nil) is ignored for Vals to simplify call sites that are clearing Opts.
func (n *Node) SetOpt(x interface{}) {
	if x == nil {
		if n.HasOpt() {
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

func (n *Node) Iota() int64 {
	return n.Offset()
}

func (n *Node) SetIota(x int64) {
	n.SetOffset(x)
}

// mayBeShared reports whether n may occur in multiple places in the AST.
// Extra care must be taken when mutating such a node.
func MayBeShared(n *Node) bool {
	switch n.Op() {
	case ONAME, OLITERAL, ONIL, OTYPE:
		return true
	}
	return false
}

// funcname returns the name (without the package) of the function n.
func FuncName(n *Node) string {
	if n == nil || n.Func() == nil || n.Func().Nname == nil {
		return "<nil>"
	}
	return n.Func().Nname.Sym().Name
}

// pkgFuncName returns the name of the function referenced by n, with package prepended.
// This differs from the compiler's internal convention where local functions lack a package
// because the ultimate consumer of this is a human looking at an IDE; package is only empty
// if the compilation package is actually the empty string.
func PkgFuncName(n *Node) string {
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
func (n *Node) CanBeAnSSASym() {
}

// Name holds Node fields used only by named nodes (ONAME, OTYPE, OPACK, OLABEL, some OLITERAL).
type Name struct {
	Pack *Node      // real package for import . names
	Pkg  *types.Pkg // pkg for OPACK nodes
	// For a local variable (not param) or extern, the initializing assignment (OAS or OAS2).
	// For a closure var, the ONAME node of the outer captured variable
	Defn *Node
	// The ODCLFUNC node (for a static function/method or a closure) in which
	// local variable or param is declared.
	Curfn     *Node
	Param     *Param // additional fields for ONAME, OTYPE
	Decldepth int32  // declaration loop depth, increased for every loop or label
	// Unique number for ONAME nodes within a function. Function outputs
	// (results) are numbered starting at one, followed by function inputs
	// (parameters), and then local variables. Vargen is used to distinguish
	// local variables/params with the same name.
	Vargen int32
	flags  bitset16
}

const (
	nameCaptured = 1 << iota // is the variable captured by a closure
	nameReadonly
	nameByval                 // is the variable captured by value or by reference
	nameNeedzero              // if it contains pointers, needs to be zeroed on function entry
	nameAutoTemp              // is the variable a temporary (implies no dwarf info. reset if escapes to heap)
	nameUsed                  // for variable declared and not used error
	nameIsClosureVar          // PAUTOHEAP closure pseudo-variable; original at n.Name.Defn
	nameIsOutputParamHeapAddr // pointer to a result parameter's heap copy
	nameAssigned              // is the variable ever assigned to
	nameAddrtaken             // address taken, even if not moved to heap
	nameInlFormal             // PAUTO created by inliner, derived from callee formal
	nameInlLocal              // PAUTO created by inliner, derived from callee local
	nameOpenDeferSlot         // if temporary var storing info for open-coded defers
	nameLibfuzzerExtraCounter // if PEXTERN should be assigned to __libfuzzer_extra_counters section
)

func (n *Name) Captured() bool              { return n.flags&nameCaptured != 0 }
func (n *Name) Readonly() bool              { return n.flags&nameReadonly != 0 }
func (n *Name) Byval() bool                 { return n.flags&nameByval != 0 }
func (n *Name) Needzero() bool              { return n.flags&nameNeedzero != 0 }
func (n *Name) AutoTemp() bool              { return n.flags&nameAutoTemp != 0 }
func (n *Name) Used() bool                  { return n.flags&nameUsed != 0 }
func (n *Name) IsClosureVar() bool          { return n.flags&nameIsClosureVar != 0 }
func (n *Name) IsOutputParamHeapAddr() bool { return n.flags&nameIsOutputParamHeapAddr != 0 }
func (n *Name) Assigned() bool              { return n.flags&nameAssigned != 0 }
func (n *Name) Addrtaken() bool             { return n.flags&nameAddrtaken != 0 }
func (n *Name) InlFormal() bool             { return n.flags&nameInlFormal != 0 }
func (n *Name) InlLocal() bool              { return n.flags&nameInlLocal != 0 }
func (n *Name) OpenDeferSlot() bool         { return n.flags&nameOpenDeferSlot != 0 }
func (n *Name) LibfuzzerExtraCounter() bool { return n.flags&nameLibfuzzerExtraCounter != 0 }

func (n *Name) SetCaptured(b bool)              { n.flags.set(nameCaptured, b) }
func (n *Name) SetReadonly(b bool)              { n.flags.set(nameReadonly, b) }
func (n *Name) SetByval(b bool)                 { n.flags.set(nameByval, b) }
func (n *Name) SetNeedzero(b bool)              { n.flags.set(nameNeedzero, b) }
func (n *Name) SetAutoTemp(b bool)              { n.flags.set(nameAutoTemp, b) }
func (n *Name) SetUsed(b bool)                  { n.flags.set(nameUsed, b) }
func (n *Name) SetIsClosureVar(b bool)          { n.flags.set(nameIsClosureVar, b) }
func (n *Name) SetIsOutputParamHeapAddr(b bool) { n.flags.set(nameIsOutputParamHeapAddr, b) }
func (n *Name) SetAssigned(b bool)              { n.flags.set(nameAssigned, b) }
func (n *Name) SetAddrtaken(b bool)             { n.flags.set(nameAddrtaken, b) }
func (n *Name) SetInlFormal(b bool)             { n.flags.set(nameInlFormal, b) }
func (n *Name) SetInlLocal(b bool)              { n.flags.set(nameInlLocal, b) }
func (n *Name) SetOpenDeferSlot(b bool)         { n.flags.set(nameOpenDeferSlot, b) }
func (n *Name) SetLibfuzzerExtraCounter(b bool) { n.flags.set(nameLibfuzzerExtraCounter, b) }

type Param struct {
	Ntype    *Node
	Heapaddr *Node // temp holding heap address of param

	// ONAME PAUTOHEAP
	Stackcopy *Node // the PPARAM/PPARAMOUT on-stack slot (moved func params only)

	// ONAME closure linkage
	// Consider:
	//
	//	func f() {
	//		x := 1 // x1
	//		func() {
	//			use(x) // x2
	//			func() {
	//				use(x) // x3
	//				--- parser is here ---
	//			}()
	//		}()
	//	}
	//
	// There is an original declaration of x and then a chain of mentions of x
	// leading into the current function. Each time x is mentioned in a new closure,
	// we create a variable representing x for use in that specific closure,
	// since the way you get to x is different in each closure.
	//
	// Let's number the specific variables as shown in the code:
	// x1 is the original x, x2 is when mentioned in the closure,
	// and x3 is when mentioned in the closure in the closure.
	//
	// We keep these linked (assume N > 1):
	//
	//   - x1.Defn = original declaration statement for x (like most variables)
	//   - x1.Innermost = current innermost closure x (in this case x3), or nil for none
	//   - x1.IsClosureVar() = false
	//
	//   - xN.Defn = x1, N > 1
	//   - xN.IsClosureVar() = true, N > 1
	//   - x2.Outer = nil
	//   - xN.Outer = x(N-1), N > 2
	//
	//
	// When we look up x in the symbol table, we always get x1.
	// Then we can use x1.Innermost (if not nil) to get the x
	// for the innermost known closure function,
	// but the first reference in a closure will find either no x1.Innermost
	// or an x1.Innermost with .Funcdepth < Funcdepth.
	// In that case, a new xN must be created, linked in with:
	//
	//     xN.Defn = x1
	//     xN.Outer = x1.Innermost
	//     x1.Innermost = xN
	//
	// When we finish the function, we'll process its closure variables
	// and find xN and pop it off the list using:
	//
	//     x1 := xN.Defn
	//     x1.Innermost = xN.Outer
	//
	// We leave x1.Innermost set so that we can still get to the original
	// variable quickly. Not shown here, but once we're
	// done parsing a function and no longer need xN.Outer for the
	// lexical x reference links as described above, funcLit
	// recomputes xN.Outer as the semantic x reference link tree,
	// even filling in x in intermediate closures that might not
	// have mentioned it along the way to inner closures that did.
	// See funcLit for details.
	//
	// During the eventual compilation, then, for closure variables we have:
	//
	//     xN.Defn = original variable
	//     xN.Outer = variable captured in next outward scope
	//                to make closure where xN appears
	//
	// Because of the sharding of pieces of the node, x.Defn means x.Name.Defn
	// and x.Innermost/Outer means x.Name.Param.Innermost/Outer.
	Innermost *Node
	Outer     *Node

	// OTYPE & ONAME //go:embed info,
	// sharing storage to reduce gc.Param size.
	// Extra is nil, or else *Extra is a *paramType or an *embedFileList.
	Extra *interface{}
}

type paramType struct {
	flag  PragmaFlag
	alias bool
}

type embedFileList []string

// Pragma returns the PragmaFlag for p, which must be for an OTYPE.
func (p *Param) Pragma() PragmaFlag {
	if p.Extra == nil {
		return 0
	}
	return (*p.Extra).(*paramType).flag
}

// SetPragma sets the PragmaFlag for p, which must be for an OTYPE.
func (p *Param) SetPragma(flag PragmaFlag) {
	if p.Extra == nil {
		if flag == 0 {
			return
		}
		p.Extra = new(interface{})
		*p.Extra = &paramType{flag: flag}
		return
	}
	(*p.Extra).(*paramType).flag = flag
}

// Alias reports whether p, which must be for an OTYPE, is a type alias.
func (p *Param) Alias() bool {
	if p.Extra == nil {
		return false
	}
	t, ok := (*p.Extra).(*paramType)
	if !ok {
		return false
	}
	return t.alias
}

// SetAlias sets whether p, which must be for an OTYPE, is a type alias.
func (p *Param) SetAlias(alias bool) {
	if p.Extra == nil {
		if !alias {
			return
		}
		p.Extra = new(interface{})
		*p.Extra = &paramType{alias: alias}
		return
	}
	(*p.Extra).(*paramType).alias = alias
}

// EmbedFiles returns the list of embedded files for p,
// which must be for an ONAME var.
func (p *Param) EmbedFiles() []string {
	if p.Extra == nil {
		return nil
	}
	return *(*p.Extra).(*embedFileList)
}

// SetEmbedFiles sets the list of embedded files for p,
// which must be for an ONAME var.
func (p *Param) SetEmbedFiles(list []string) {
	if p.Extra == nil {
		if len(list) == 0 {
			return
		}
		f := embedFileList(list)
		p.Extra = new(interface{})
		*p.Extra = &f
		return
	}
	*(*p.Extra).(*embedFileList) = list
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
	Nname    *Node // ONAME node
	Decl     *Node // ODCLFUNC node
	OClosure *Node // OCLOSURE node

	Shortname *types.Sym

	// Extra entry code for the function. For example, allocate and initialize
	// memory for escaping parameters.
	Enter Nodes
	Exit  Nodes
	// ONAME nodes for all params/locals for this func/closure, does NOT
	// include closurevars until transformclosure runs.
	Dcl []*Node

	ClosureEnter  Nodes // list of ONAME nodes of captured variables
	ClosureType   *Node // closure representation type
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
	Dcl  []*Node
	Body []*Node
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
type Nodes struct{ slice *[]*Node }

// asNodes returns a slice of *Node as a Nodes value.
func AsNodes(s []*Node) Nodes {
	return Nodes{&s}
}

// Slice returns the entries in Nodes as a slice.
// Changes to the slice entries (as in s[i] = n) will be reflected in
// the Nodes.
func (n Nodes) Slice() []*Node {
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
func (n Nodes) Index(i int) *Node {
	return (*n.slice)[i]
}

// First returns the first element of Nodes (same as n.Index(0)).
// It panics if n has no elements.
func (n Nodes) First() *Node {
	return (*n.slice)[0]
}

// Second returns the second element of Nodes (same as n.Index(1)).
// It panics if n has fewer than two elements.
func (n Nodes) Second() *Node {
	return (*n.slice)[1]
}

// Set sets n to a slice.
// This takes ownership of the slice.
func (n *Nodes) Set(s []*Node) {
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
func (n *Nodes) Set1(n1 *Node) {
	n.slice = &[]*Node{n1}
}

// Set2 sets n to a slice containing two nodes.
func (n *Nodes) Set2(n1, n2 *Node) {
	n.slice = &[]*Node{n1, n2}
}

// Set3 sets n to a slice containing three nodes.
func (n *Nodes) Set3(n1, n2, n3 *Node) {
	n.slice = &[]*Node{n1, n2, n3}
}

// MoveNodes sets n to the contents of n2, then clears n2.
func (n *Nodes) MoveNodes(n2 *Nodes) {
	n.slice = n2.slice
	n2.slice = nil
}

// SetIndex sets the i'th element of Nodes to node.
// It panics if n does not have at least i+1 elements.
func (n Nodes) SetIndex(i int, node *Node) {
	(*n.slice)[i] = node
}

// SetFirst sets the first element of Nodes to node.
// It panics if n does not have at least one elements.
func (n Nodes) SetFirst(node *Node) {
	(*n.slice)[0] = node
}

// SetSecond sets the second element of Nodes to node.
// It panics if n does not have at least two elements.
func (n Nodes) SetSecond(node *Node) {
	(*n.slice)[1] = node
}

// Addr returns the address of the i'th element of Nodes.
// It panics if n does not have at least i+1 elements.
func (n Nodes) Addr(i int) **Node {
	return &(*n.slice)[i]
}

// Append appends entries to Nodes.
func (n *Nodes) Append(a ...*Node) {
	if len(a) == 0 {
		return
	}
	if n.slice == nil {
		s := make([]*Node, len(a))
		copy(s, a)
		n.slice = &s
		return
	}
	*n.slice = append(*n.slice, a...)
}

// Prepend prepends entries to Nodes.
// If a slice is passed in, this will take ownership of it.
func (n *Nodes) Prepend(a ...*Node) {
	if len(a) == 0 {
		return
	}
	if n.slice == nil {
		n.slice = &a
	} else {
		*n.slice = append(a, *n.slice...)
	}
}

// AppendNodes appends the contents of *n2 to n, then clears n2.
func (n *Nodes) AppendNodes(n2 *Nodes) {
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
func Inspect(n *Node, f func(*Node) bool) {
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

func InspectList(l Nodes, f func(*Node) bool) {
	for _, n := range l.Slice() {
		Inspect(n, f)
	}
}

// nodeQueue is a FIFO queue of *Node. The zero value of nodeQueue is
// a ready-to-use empty queue.
type NodeQueue struct {
	ring       []*Node
	head, tail int
}

// empty reports whether q contains no Nodes.
func (q *NodeQueue) Empty() bool {
	return q.head == q.tail
}

// pushRight appends n to the right of the queue.
func (q *NodeQueue) PushRight(n *Node) {
	if len(q.ring) == 0 {
		q.ring = make([]*Node, 16)
	} else if q.head+len(q.ring) == q.tail {
		// Grow the ring.
		nring := make([]*Node, len(q.ring)*2)
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
func (q *NodeQueue) PopLeft() *Node {
	if q.Empty() {
		panic("dequeue empty")
	}
	n := q.ring[q.head%len(q.ring)]
	q.head++
	return n
}

// NodeSet is a set of Nodes.
type NodeSet map[*Node]struct{}

// Has reports whether s contains n.
func (s NodeSet) Has(n *Node) bool {
	_, isPresent := s[n]
	return isPresent
}

// Add adds n to s.
func (s *NodeSet) Add(n *Node) {
	if *s == nil {
		*s = make(map[*Node]struct{})
	}
	(*s)[n] = struct{}{}
}

// Sorted returns s sorted according to less.
func (s NodeSet) Sorted(less func(*Node, *Node) bool) []*Node {
	var res []*Node
	for n := range s {
		res = append(res, n)
	}
	sort.Slice(res, func(i, j int) bool { return less(res[i], res[j]) })
	return res
}

func Nod(op Op, nleft, nright *Node) *Node {
	return NodAt(base.Pos, op, nleft, nright)
}

func NodAt(pos src.XPos, op Op, nleft, nright *Node) *Node {
	var n *Node
	switch op {
	case ODCLFUNC:
		var x struct {
			n Node
			f Func
		}
		n = &x.n
		n.SetFunc(&x.f)
		n.Func().Decl = n
	case ONAME:
		base.Fatalf("use newname instead")
	case OLABEL, OPACK:
		var x struct {
			n Node
			m Name
		}
		n = &x.n
		n.SetName(&x.m)
	default:
		n = new(Node)
	}
	n.SetOp(op)
	n.SetLeft(nleft)
	n.SetRight(nright)
	n.SetPos(pos)
	n.SetOffset(types.BADWIDTH)
	n.SetOrig(n)
	return n
}

// newnamel returns a new ONAME Node associated with symbol s at position pos.
// The caller is responsible for setting n.Name.Curfn.
func NewNameAt(pos src.XPos, s *types.Sym) *Node {
	if s == nil {
		base.Fatalf("newnamel nil")
	}

	var x struct {
		n Node
		m Name
		p Param
	}
	n := &x.n
	n.SetName(&x.m)
	n.Name().Param = &x.p

	n.SetOp(ONAME)
	n.SetPos(pos)
	n.SetOrig(n)

	n.SetSym(s)
	return n
}

// The Class of a variable/function describes the "storage class"
// of a variable or function. During parsing, storage classes are
// called declaration contexts.
type Class uint8

//go:generate stringer -type=Class
const (
	Pxxx      Class = iota // no class; used during ssa conversion to indicate pseudo-variables
	PEXTERN                // global variables
	PAUTO                  // local variables
	PAUTOHEAP              // local variables or parameters moved to heap
	PPARAM                 // input arguments
	PPARAMOUT              // output results
	PFUNC                  // global functions

	// Careful: Class is stored in three bits in Node.flags.
	_ = uint((1 << 3) - iota) // static assert for iota <= (1 << 3)
)

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

func AsNode(n types.IRNode) *Node {
	if n == nil {
		return nil
	}
	return n.(*Node)
}

var BlankNode *Node

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
func (n *Node) SliceBounds() (low, high, max *Node) {
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
func (n *Node) SetSliceBounds(low, high, max *Node) {
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

func IsConst(n *Node, ct constant.Kind) bool {
	return ConstType(n) == ct
}

// Int64Val returns n as an int64.
// n must be an integer or rune constant.
func (n *Node) Int64Val() int64 {
	if !IsConst(n, constant.Int) {
		base.Fatalf("Int64Val(%v)", n)
	}
	x, ok := constant.Int64Val(n.Val())
	if !ok {
		base.Fatalf("Int64Val(%v)", n)
	}
	return x
}

// CanInt64 reports whether it is safe to call Int64Val() on n.
func (n *Node) CanInt64() bool {
	if !IsConst(n, constant.Int) {
		return false
	}

	// if the value inside n cannot be represented as an int64, the
	// return value of Int64 is undefined
	_, ok := constant.Int64Val(n.Val())
	return ok
}

// Uint64Val returns n as an uint64.
// n must be an integer or rune constant.
func (n *Node) Uint64Val() uint64 {
	if !IsConst(n, constant.Int) {
		base.Fatalf("Uint64Val(%v)", n)
	}
	x, ok := constant.Uint64Val(n.Val())
	if !ok {
		base.Fatalf("Uint64Val(%v)", n)
	}
	return x
}

// BoolVal returns n as a bool.
// n must be a boolean constant.
func (n *Node) BoolVal() bool {
	if !IsConst(n, constant.Bool) {
		base.Fatalf("BoolVal(%v)", n)
	}
	return constant.BoolVal(n.Val())
}

// StringVal returns the value of a literal string Node as a string.
// n must be a string constant.
func (n *Node) StringVal() string {
	if !IsConst(n, constant.String) {
		base.Fatalf("StringVal(%v)", n)
	}
	return constant.StringVal(n.Val())
}

// rawcopy returns a shallow copy of n.
// Note: copy or sepcopy (rather than rawcopy) is usually the
//       correct choice (see comment with Node.copy, below).
func (n *Node) RawCopy() *Node {
	copy := *n
	return &copy
}

// sepcopy returns a separate shallow copy of n, with the copy's
// Orig pointing to itself.
func SepCopy(n *Node) *Node {
	n = n.RawCopy()
	n.SetOrig(n)
	return n
}

// copy returns shallow copy of n and adjusts the copy's Orig if
// necessary: In general, if n.Orig points to itself, the copy's
// Orig should point to itself as well. Otherwise, if n is modified,
// the copy's Orig node appears modified, too, and then doesn't
// represent the original node anymore.
// (This caused the wrong complit Op to be used when printing error
// messages; see issues #26855, #27765).
func Copy(n *Node) *Node {
	copy := n.RawCopy()
	if n.Orig() == n {
		copy.SetOrig(copy)
	}
	return copy
}

// isNil reports whether n represents the universal untyped zero value "nil".
func IsNil(n *Node) bool {
	// Check n.Orig because constant propagation may produce typed nil constants,
	// which don't exist in the Go spec.
	return n.Orig().Op() == ONIL
}

func IsBlank(n *Node) bool {
	if n == nil {
		return false
	}
	return n.Sym().IsBlank()
}

// IsMethod reports whether n is a method.
// n must be a function or a method.
func IsMethod(n *Node) bool {
	return n.Type().Recv() != nil
}
