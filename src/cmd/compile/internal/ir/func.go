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
// The ONAME node (Func.Nname) is used for plain references to it.
// The ODCLFUNC node (the Func itself) is used for its declaration code.
// The OCLOSURE node (Func.OClosure) is used for a reference to a
// function literal.
//
// An imported function will have an ONAME node which points to a Func
// with an empty body.
// A declared function or method has an ODCLFUNC (the Func itself) and an ONAME.
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
// when it is called directly and by OMETHVALUE otherwise.
// These are like method expressions, except that for ODOTMETH/ODOTINTER,
// the method name is stored in Sym instead of Right.
// Each OMETHVALUE ends up being implemented as a new
// function, a bit like a closure, with its own ODCLFUNC.
// The OMETHVALUE uses n.Func to record the linkage to
// the generated ODCLFUNC, but there is no
// pointer from the Func back to the OMETHVALUE.
type Func struct {
	miniNode
	Body Nodes
	Iota int64

	Nname    *Name        // ONAME node
	OClosure *ClosureExpr // OCLOSURE node

	Shortname *types.Sym

	// Extra entry code for the function. For example, allocate and initialize
	// memory for escaping parameters.
	Enter Nodes
	Exit  Nodes

	// ONAME nodes for all params/locals for this func/closure, does NOT
	// include closurevars until transforming closures during walk.
	// Names must be listed PPARAMs, PPARAMOUTs, then PAUTOs,
	// with PPARAMs and PPARAMOUTs in order corresponding to the function signature.
	// However, as anonymous or blank PPARAMs are not actually declared,
	// they are omitted from Dcl.
	// Anonymous and blank PPARAMOUTs are declared as ~rNN and ~bNN Names, respectively.
	Dcl []*Name

	// ClosureVars lists the free variables that are used within a
	// function literal, but formally declared in an enclosing
	// function. The variables in this slice are the closure function's
	// own copy of the variables, which are used within its function
	// body. They will also each have IsClosureVar set, and will have
	// Byval set if they're captured by value.
	ClosureVars []*Name

	// Enclosed functions that need to be compiled.
	// Populated during walk.
	Closures []*Func

	// Parents records the parent scope of each scope within a
	// function. The root scope (0) has no parent, so the i'th
	// scope's parent is stored at Parents[i-1].
	Parents []ScopeID

	// Marks records scope boundary changes.
	Marks []Mark

	FieldTrack map[*obj.LSym]struct{}
	DebugInfo  interface{}
	LSym       *obj.LSym // Linker object in this function's native ABI (Func.ABI)

	Inl *Inline

	// Closgen tracks how many closures have been generated within
	// this function. Used by closurename for creating unique
	// function names.
	Closgen int32

	Label int32 // largest auto-generated label in this function

	Endlineno src.XPos
	WBPos     src.XPos // position of first write barrier; see SetWBPos

	Pragma PragmaFlag // go:xxx function annotations

	flags bitset16

	// ABI is a function's "definition" ABI. This is the ABI that
	// this function's generated code is expecting to be called by.
	//
	// For most functions, this will be obj.ABIInternal. It may be
	// a different ABI for functions defined in assembly or ABI wrappers.
	//
	// This is included in the export data and tracked across packages.
	ABI obj.ABI
	// ABIRefs is the set of ABIs by which this function is referenced.
	// For ABIs other than this function's definition ABI, the
	// compiler generates ABI wrapper functions. This is only tracked
	// within a package.
	ABIRefs obj.ABISet

	NumDefers  int32 // number of defer calls in the function
	NumReturns int32 // number of explicit returns in the function

	// nwbrCalls records the LSyms of functions called by this
	// function for go:nowritebarrierrec analysis. Only filled in
	// if nowritebarrierrecCheck != nil.
	NWBRCalls *[]SymAndPos

	// For wrapper functions, WrappedFunc point to the original Func.
	// Currently only used for go/defer wrappers.
	WrappedFunc *Func
}

func NewFunc(pos src.XPos) *Func {
	f := new(Func)
	f.pos = pos
	f.op = ODCLFUNC
	f.Iota = -1
	// Most functions are ABIInternal. The importer or symabis
	// pass may override this.
	f.ABI = obj.ABIInternal
	return f
}

func (f *Func) isStmt() {}

func (n *Func) copy() Node                         { panic(n.no("copy")) }
func (n *Func) doChildren(do func(Node) bool) bool { return doNodes(n.Body, do) }
func (n *Func) editChildren(edit func(Node) Node)  { editNodes(n.Body, edit) }

func (f *Func) Type() *types.Type                { return f.Nname.Type() }
func (f *Func) Sym() *types.Sym                  { return f.Nname.Sym() }
func (f *Func) Linksym() *obj.LSym               { return f.Nname.Linksym() }
func (f *Func) LinksymABI(abi obj.ABI) *obj.LSym { return f.Nname.LinksymABI(abi) }

// An Inline holds fields used for function bodies that can be inlined.
type Inline struct {
	Cost int32 // heuristic cost of inlining this function

	// Copies of Func.Dcl and Func.Body for use during inlining. Copies are
	// needed because the function's dcl/body may be changed by later compiler
	// transformations. These fields are also populated when a function from
	// another package is imported.
	Dcl  []*Name
	Body []Node

	// CanDelayResults reports whether it's safe for the inliner to delay
	// initializing the result parameters until immediately before the
	// "return" statement.
	CanDelayResults bool
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
	funcWrapper                   // hide frame from users (elide in tracebacks, don't count as a frame for recover())
	funcABIWrapper                // is an ABI wrapper (also set flagWrapper)
	funcNeedctxt                  // function uses context register (has closure variables)
	funcReflectMethod             // function calls reflect.Type.Method or MethodByName
	// true if closure inside a function; false if a simple function or a
	// closure in a global variable initialization
	funcIsHiddenClosure
	funcIsDeadcodeClosure        // true if closure is deadcode
	funcHasDefer                 // contains a defer statement
	funcNilCheckDisabled         // disable nil checks when compiling this function
	funcInlinabilityChecked      // inliner has already determined whether the function is inlinable
	funcExportInline             // include inline body in export data
	funcInstrumentBody           // add race/msan/asan instrumentation during SSA construction
	funcOpenCodedDeferDisallowed // can't do open-coded defers
	funcClosureCalled            // closure is only immediately called; used by escape analysis
)

type SymAndPos struct {
	Sym *obj.LSym // LSym of callee
	Pos src.XPos  // line of call
}

func (f *Func) Dupok() bool                    { return f.flags&funcDupok != 0 }
func (f *Func) Wrapper() bool                  { return f.flags&funcWrapper != 0 }
func (f *Func) ABIWrapper() bool               { return f.flags&funcABIWrapper != 0 }
func (f *Func) Needctxt() bool                 { return f.flags&funcNeedctxt != 0 }
func (f *Func) ReflectMethod() bool            { return f.flags&funcReflectMethod != 0 }
func (f *Func) IsHiddenClosure() bool          { return f.flags&funcIsHiddenClosure != 0 }
func (f *Func) IsDeadcodeClosure() bool        { return f.flags&funcIsDeadcodeClosure != 0 }
func (f *Func) HasDefer() bool                 { return f.flags&funcHasDefer != 0 }
func (f *Func) NilCheckDisabled() bool         { return f.flags&funcNilCheckDisabled != 0 }
func (f *Func) InlinabilityChecked() bool      { return f.flags&funcInlinabilityChecked != 0 }
func (f *Func) ExportInline() bool             { return f.flags&funcExportInline != 0 }
func (f *Func) InstrumentBody() bool           { return f.flags&funcInstrumentBody != 0 }
func (f *Func) OpenCodedDeferDisallowed() bool { return f.flags&funcOpenCodedDeferDisallowed != 0 }
func (f *Func) ClosureCalled() bool            { return f.flags&funcClosureCalled != 0 }

func (f *Func) SetDupok(b bool)                    { f.flags.set(funcDupok, b) }
func (f *Func) SetWrapper(b bool)                  { f.flags.set(funcWrapper, b) }
func (f *Func) SetABIWrapper(b bool)               { f.flags.set(funcABIWrapper, b) }
func (f *Func) SetNeedctxt(b bool)                 { f.flags.set(funcNeedctxt, b) }
func (f *Func) SetReflectMethod(b bool)            { f.flags.set(funcReflectMethod, b) }
func (f *Func) SetIsHiddenClosure(b bool)          { f.flags.set(funcIsHiddenClosure, b) }
func (f *Func) SetIsDeadcodeClosure(b bool)        { f.flags.set(funcIsDeadcodeClosure, b) }
func (f *Func) SetHasDefer(b bool)                 { f.flags.set(funcHasDefer, b) }
func (f *Func) SetNilCheckDisabled(b bool)         { f.flags.set(funcNilCheckDisabled, b) }
func (f *Func) SetInlinabilityChecked(b bool)      { f.flags.set(funcInlinabilityChecked, b) }
func (f *Func) SetExportInline(b bool)             { f.flags.set(funcExportInline, b) }
func (f *Func) SetInstrumentBody(b bool)           { f.flags.set(funcInstrumentBody, b) }
func (f *Func) SetOpenCodedDeferDisallowed(b bool) { f.flags.set(funcOpenCodedDeferDisallowed, b) }
func (f *Func) SetClosureCalled(b bool)            { f.flags.set(funcClosureCalled, b) }

func (f *Func) SetWBPos(pos src.XPos) {
	if base.Debug.WB != 0 {
		base.WarnfAt(pos, "write barrier")
	}
	if !f.WBPos.IsKnown() {
		f.WBPos = pos
	}
}

// FuncName returns the name (without the package) of the function n.
func FuncName(f *Func) string {
	if f == nil || f.Nname == nil {
		return "<nil>"
	}
	return f.Sym().Name
}

// PkgFuncName returns the name of the function referenced by n, with package prepended.
// This differs from the compiler's internal convention where local functions lack a package
// because the ultimate consumer of this is a human looking at an IDE; package is only empty
// if the compilation package is actually the empty string.
func PkgFuncName(f *Func) string {
	if f == nil || f.Nname == nil {
		return "<nil>"
	}
	s := f.Sym()
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

var CurFunc *Func

// WithFunc invokes do with CurFunc and base.Pos set to curfn and
// curfn.Pos(), respectively, and then restores their previous values
// before returning.
func WithFunc(curfn *Func, do func()) {
	oldfn, oldpos := CurFunc, base.Pos
	defer func() { CurFunc, base.Pos = oldfn, oldpos }()

	CurFunc, base.Pos = curfn, curfn.Pos()
	do()
}

func FuncSymName(s *types.Sym) string {
	return s.Name + "·f"
}

// MarkFunc marks a node as a function.
func MarkFunc(n *Name) {
	if n.Op() != ONAME || n.Class != Pxxx {
		base.FatalfAt(n.Pos(), "expected ONAME/Pxxx node, got %v (%v/%v)", n, n.Op(), n.Class)
	}

	n.Class = PFUNC
	n.Sym().SetFunc(true)
}

// ClosureDebugRuntimeCheck applies boilerplate checks for debug flags
// and compiling runtime
func ClosureDebugRuntimeCheck(clo *ClosureExpr) {
	if base.Debug.Closure > 0 {
		if clo.Esc() == EscHeap {
			base.WarnfAt(clo.Pos(), "heap closure, captured vars = %v", clo.Func.ClosureVars)
		} else {
			base.WarnfAt(clo.Pos(), "stack closure, captured vars = %v", clo.Func.ClosureVars)
		}
	}
	if base.Flag.CompilingRuntime && clo.Esc() == EscHeap && !clo.IsGoWrap {
		base.ErrorfAt(clo.Pos(), "heap-allocated closure %s, not allowed in runtime", FuncName(clo.Func))
	}
}

// IsTrivialClosure reports whether closure clo has an
// empty list of captured vars.
func IsTrivialClosure(clo *ClosureExpr) bool {
	return len(clo.Func.ClosureVars) == 0
}

// globClosgen is like Func.Closgen, but for the global scope.
var globClosgen int32

// closureName generates a new unique name for a closure within outerfn.
func closureName(outerfn *Func) *types.Sym {
	pkg := types.LocalPkg
	outer := "glob."
	prefix := "func"
	gen := &globClosgen

	if outerfn != nil {
		if outerfn.OClosure != nil {
			prefix = ""
		}

		pkg = outerfn.Sym().Pkg
		outer = FuncName(outerfn)

		// There may be multiple functions named "_". In those
		// cases, we can't use their individual Closgens as it
		// would lead to name clashes.
		if !IsBlank(outerfn.Nname) {
			gen = &outerfn.Closgen
		}
	}

	*gen++
	return pkg.Lookup(fmt.Sprintf("%s.%s%d", outer, prefix, *gen))
}

// NewClosureFunc creates a new Func to represent a function literal.
// If hidden is true, then the closure is marked hidden (i.e., as a
// function literal contained within another function, rather than a
// package-scope variable initialization expression).
func NewClosureFunc(pos src.XPos, hidden bool) *Func {
	fn := NewFunc(pos)
	fn.SetIsHiddenClosure(hidden)

	fn.Nname = NewNameAt(pos, BlankNode.Sym())
	fn.Nname.Func = fn
	fn.Nname.Defn = fn

	fn.OClosure = NewClosureExpr(pos, fn)

	return fn
}

// NameClosure generates a unique for the given function literal,
// which must have appeared within outerfn.
func NameClosure(clo *ClosureExpr, outerfn *Func) {
	fn := clo.Func
	if fn.IsHiddenClosure() != (outerfn != nil) {
		base.FatalfAt(clo.Pos(), "closure naming inconsistency: hidden %v, but outer %v", fn.IsHiddenClosure(), outerfn)
	}

	name := fn.Nname
	if !IsBlank(name) {
		base.FatalfAt(clo.Pos(), "closure already named: %v", name)
	}

	name.SetSym(closureName(outerfn))
	MarkFunc(name)
}

// UseClosure checks that the ginen function literal has been setup
// correctly, and then returns it as an expression.
// It must be called after clo.Func.ClosureVars has been set.
func UseClosure(clo *ClosureExpr, pkg *Package) Node {
	fn := clo.Func
	name := fn.Nname

	if IsBlank(name) {
		base.FatalfAt(fn.Pos(), "unnamed closure func: %v", fn)
	}
	// Caution: clo.Typecheck() is still 0 when UseClosure is called by
	// tcClosure.
	if fn.Typecheck() != 1 || name.Typecheck() != 1 {
		base.FatalfAt(fn.Pos(), "missed typecheck: %v", fn)
	}
	if clo.Type() == nil || name.Type() == nil {
		base.FatalfAt(fn.Pos(), "missing types: %v", fn)
	}
	if !types.Identical(clo.Type(), name.Type()) {
		base.FatalfAt(fn.Pos(), "mismatched types: %v", fn)
	}

	if base.Flag.W > 1 {
		s := fmt.Sprintf("new closure func: %v", fn)
		Dump(s, fn)
	}

	if pkg != nil {
		pkg.Decls = append(pkg.Decls, fn)
	}

	if false && IsTrivialClosure(clo) {
		// TODO(mdempsky): Investigate if we can/should optimize this
		// case. walkClosure already handles it later, but it could be
		// useful to recognize earlier (e.g., it might allow multiple
		// inlined calls to a function to share a common trivial closure
		// func, rather than cloning it for each inlined call).
	}

	return clo
}
