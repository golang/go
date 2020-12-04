// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"cmd/internal/src"

	"go/constant"
)

// Name holds Node fields used only by named nodes (ONAME, OTYPE, some OLITERAL).
type Name struct {
	miniExpr
	subOp      Op    // uint8
	class      Class // uint8
	flags      bitset16
	pragma     PragmaFlag // int16
	sym        *types.Sym
	fn         *Func
	offset     int64
	val        constant.Value
	orig       Node
	embedFiles *[]string // list of embedded files, for ONAME var

	PkgName *PkgName // real package for import . names
	// For a local variable (not param) or extern, the initializing assignment (OAS or OAS2).
	// For a closure var, the ONAME node of the outer captured variable
	Defn Node

	// The function, method, or closure in which local variable or param is declared.
	Curfn *Func

	// Unique number for ONAME nodes within a function. Function outputs
	// (results) are numbered starting at one, followed by function inputs
	// (parameters), and then local variables. Vargen is used to distinguish
	// local variables/params with the same name.
	Vargen    int32
	Decldepth int32 // declaration loop depth, increased for every loop or label

	Ntype    Ntype
	Heapaddr *Name // temp holding heap address of param

	// ONAME PAUTOHEAP
	Stackcopy *Name // the PPARAM/PPARAMOUT on-stack slot (moved func params only)

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
	Innermost *Name
	Outer     *Name
}

// NewNameAt returns a new ONAME Node associated with symbol s at position pos.
// The caller is responsible for setting Curfn.
func NewNameAt(pos src.XPos, sym *types.Sym) *Name {
	if sym == nil {
		base.Fatalf("NewNameAt nil")
	}
	return newNameAt(pos, ONAME, sym)
}

// NewDeclNameAt returns a new ONONAME Node associated with symbol s at position pos.
// The caller is responsible for setting Curfn.
func NewDeclNameAt(pos src.XPos, sym *types.Sym) *Name {
	if sym == nil {
		base.Fatalf("NewDeclNameAt nil")
	}
	return newNameAt(pos, ONONAME, sym)
}

// newNameAt is like NewNameAt but allows sym == nil.
func newNameAt(pos src.XPos, op Op, sym *types.Sym) *Name {
	n := new(Name)
	n.op = op
	n.pos = pos
	n.orig = n
	n.sym = sym
	return n
}

func (n *Name) Name() *Name         { return n }
func (n *Name) Sym() *types.Sym     { return n.sym }
func (n *Name) SetSym(x *types.Sym) { n.sym = x }
func (n *Name) SubOp() Op           { return n.subOp }
func (n *Name) SetSubOp(x Op)       { n.subOp = x }
func (n *Name) Class() Class        { return n.class }
func (n *Name) SetClass(x Class)    { n.class = x }
func (n *Name) Func() *Func         { return n.fn }
func (n *Name) SetFunc(x *Func)     { n.fn = x }
func (n *Name) Offset() int64       { return n.offset }
func (n *Name) SetOffset(x int64)   { n.offset = x }
func (n *Name) Iota() int64         { return n.offset }
func (n *Name) SetIota(x int64)     { n.offset = x }

func (*Name) CanBeNtype() {}

func (n *Name) SetOp(op Op) {
	if n.op != ONONAME {
		base.Fatalf("%v already has Op %v", n, n.op)
	}
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OLITERAL, ONAME, OTYPE, OIOTA:
		n.op = op
	}
}

// Pragma returns the PragmaFlag for p, which must be for an OTYPE.
func (n *Name) Pragma() PragmaFlag { return n.pragma }

// SetPragma sets the PragmaFlag for p, which must be for an OTYPE.
func (n *Name) SetPragma(flag PragmaFlag) { n.pragma = flag }

// Alias reports whether p, which must be for an OTYPE, is a type alias.
func (n *Name) Alias() bool { return n.flags&nameAlias != 0 }

// SetAlias sets whether p, which must be for an OTYPE, is a type alias.
func (n *Name) SetAlias(alias bool) { n.flags.set(nameAlias, alias) }

// EmbedFiles returns the list of embedded files for p,
// which must be for an ONAME var.
func (n *Name) EmbedFiles() []string {
	if n.embedFiles == nil {
		return nil
	}
	return *n.embedFiles
}

// SetEmbedFiles sets the list of embedded files for p,
// which must be for an ONAME var.
func (n *Name) SetEmbedFiles(list []string) {
	if n.embedFiles == nil && list == nil {
		return
	}
	if n.embedFiles == nil {
		n.embedFiles = new([]string)
	}
	*n.embedFiles = list
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
	nameIsDDD                 // is function argument a ...
	nameAlias                 // is type name an alias
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
func (n *Name) IsDDD() bool                 { return n.flags&nameIsDDD != 0 }

func (n *Name) SetCaptured(b bool)              { n.flags.set(nameCaptured, b) }
func (n *Name) setReadonly(b bool)              { n.flags.set(nameReadonly, b) }
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
func (n *Name) SetIsDDD(b bool)                 { n.flags.set(nameIsDDD, b) }

// MarkReadonly indicates that n is an ONAME with readonly contents.
func (n *Name) MarkReadonly() {
	if n.Op() != ONAME {
		base.Fatalf("Node.MarkReadonly %v", n.Op())
	}
	n.Name().setReadonly(true)
	// Mark the linksym as readonly immediately
	// so that the SSA backend can use this information.
	// It will be overridden later during dumpglobls.
	n.Sym().Linksym().Type = objabi.SRODATA
}

// Val returns the constant.Value for the node.
func (n *Name) Val() constant.Value {
	if n.val == nil {
		return constant.MakeUnknown()
	}
	return n.val
}

// SetVal sets the constant.Value for the node,
// which must not have been used with SetOpt.
func (n *Name) SetVal(v constant.Value) {
	if n.op != OLITERAL {
		panic(n.no("SetVal"))
	}
	AssertValidTypeForConst(n.Type(), v)
	n.val = v
}

// SameSource reports whether two nodes refer to the same source
// element.
//
// It exists to help incrementally migrate the compiler towards
// allowing the introduction of IdentExpr (#42990). Once we have
// IdentExpr, it will no longer be safe to directly compare Node
// values to tell if they refer to the same Name. Instead, code will
// need to explicitly get references to the underlying Name object(s),
// and compare those instead.
//
// It will still be safe to compare Nodes directly for checking if two
// nodes are syntactically the same. The SameSource function exists to
// indicate code that intentionally compares Nodes for syntactic
// equality as opposed to code that has yet to be updated in
// preparation for IdentExpr.
func SameSource(n1, n2 Node) bool {
	return n1 == n2
}

// Uses reports whether expression x is a (direct) use of the given
// variable.
func Uses(x Node, v *Name) bool {
	if v == nil || v.Op() != ONAME {
		base.Fatalf("RefersTo bad Name: %v", v)
	}
	return x.Op() == ONAME && x.Name() == v
}

// DeclaredBy reports whether expression x refers (directly) to a
// variable that was declared by the given statement.
func DeclaredBy(x, stmt Node) bool {
	if stmt == nil {
		base.Fatalf("DeclaredBy nil")
	}
	return x.Op() == ONAME && SameSource(x.Name().Defn, stmt)
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

// A Pack is an identifier referring to an imported package.
type PkgName struct {
	miniNode
	sym  *types.Sym
	Pkg  *types.Pkg
	Used bool
}

func (p *PkgName) Sym() *types.Sym { return p.sym }

func (*PkgName) CanBeNtype() {}

func NewPkgName(pos src.XPos, sym *types.Sym, pkg *types.Pkg) *PkgName {
	p := &PkgName{sym: sym, Pkg: pkg}
	p.op = OPACK
	p.pos = pos
	return p
}
