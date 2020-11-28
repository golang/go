// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
)

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
	miniNode
	typ  *types.Type
	body Nodes
	iota int64

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

func NewFunc(pos src.XPos) *Func {
	f := new(Func)
	f.pos = pos
	f.op = ODCLFUNC
	f.Decl = f
	f.iota = -1
	return f
}

func (f *Func) String() string                { return fmt.Sprint(f) }
func (f *Func) Format(s fmt.State, verb rune) { FmtNode(f, s, verb) }
func (f *Func) RawCopy() Node                 { panic(f.no("RawCopy")) }
func (f *Func) Func() *Func                   { return f }
func (f *Func) Body() Nodes                   { return f.body }
func (f *Func) PtrBody() *Nodes               { return &f.body }
func (f *Func) SetBody(x Nodes)               { f.body = x }
func (f *Func) Type() *types.Type             { return f.typ }
func (f *Func) SetType(x *types.Type)         { f.typ = x }
func (f *Func) Iota() int64                   { return f.iota }
func (f *Func) SetIota(x int64)               { f.iota = x }

func (f *Func) Sym() *types.Sym {
	if f.Nname != nil {
		return f.Nname.Sym()
	}
	return nil
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

type SymAndPos struct {
	Sym *obj.LSym // LSym of callee
	Pos src.XPos  // line of call
}

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
