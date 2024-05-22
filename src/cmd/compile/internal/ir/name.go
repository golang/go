// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"fmt"

	"go/constant"
)

// An Ident is an identifier, possibly qualified.
type Ident struct {
	miniExpr
	sym *types.Sym
}

func NewIdent(pos src.XPos, sym *types.Sym) *Ident {
	n := new(Ident)
	n.op = ONONAME
	n.pos = pos
	n.sym = sym
	return n
}

func (n *Ident) Sym() *types.Sym { return n.sym }

// Name holds Node fields used only by named nodes (ONAME, OTYPE, some OLITERAL).
type Name struct {
	miniExpr
	BuiltinOp Op         // uint8
	Class     Class      // uint8
	pragma    PragmaFlag // int16
	flags     bitset16
	DictIndex uint16 // index of the dictionary entry describing the type of this variable declaration plus 1
	sym       *types.Sym
	Func      *Func // TODO(austin): nil for I.M
	Offset_   int64
	val       constant.Value
	Opt       interface{} // for use by escape analysis
	Embed     *[]Embed    // list of embedded files, for ONAME var

	// For a local variable (not param) or extern, the initializing assignment (OAS or OAS2).
	// For a closure var, the ONAME node of the original (outermost) captured variable.
	// For the case-local variables of a type switch, the type switch guard (OTYPESW).
	// For a range variable, the range statement (ORANGE)
	// For a recv variable in a case of a select statement, the receive assignment (OSELRECV2)
	// For the name of a function, points to corresponding Func node.
	Defn Node

	// The function, method, or closure in which local variable or param is declared.
	Curfn *Func

	Heapaddr *Name // temp holding heap address of param

	// Outer points to the immediately enclosing function's copy of this
	// closure variable. If not a closure variable, then Outer is nil.
	Outer *Name
}

func (n *Name) isExpr() {}

func (n *Name) copy() Node                                   { panic(n.no("copy")) }
func (n *Name) doChildren(do func(Node) bool) bool           { return false }
func (n *Name) doChildrenWithHidden(do func(Node) bool) bool { return false }
func (n *Name) editChildren(edit func(Node) Node)            {}
func (n *Name) editChildrenWithHidden(edit func(Node) Node)  {}

// RecordFrameOffset records the frame offset for the name.
// It is used by package types when laying out function arguments.
func (n *Name) RecordFrameOffset(offset int64) {
	n.SetFrameOffset(offset)
}

// NewNameAt returns a new ONAME Node associated with symbol s at position pos.
// The caller is responsible for setting Curfn.
func NewNameAt(pos src.XPos, sym *types.Sym, typ *types.Type) *Name {
	if sym == nil {
		base.Fatalf("NewNameAt nil")
	}
	n := newNameAt(pos, ONAME, sym)
	if typ != nil {
		n.SetType(typ)
		n.SetTypecheck(1)
	}
	return n
}

// NewBuiltin returns a new Name representing a builtin function,
// either predeclared or from package unsafe.
func NewBuiltin(sym *types.Sym, op Op) *Name {
	n := newNameAt(src.NoXPos, ONAME, sym)
	n.BuiltinOp = op
	n.SetTypecheck(1)
	sym.Def = n
	return n
}

// NewLocal returns a new function-local variable with the given name and type.
func (fn *Func) NewLocal(pos src.XPos, sym *types.Sym, typ *types.Type) *Name {
	if fn.Dcl == nil {
		base.FatalfAt(pos, "must call DeclParams on %v first", fn)
	}

	n := NewNameAt(pos, sym, typ)
	n.Class = PAUTO
	n.Curfn = fn
	fn.Dcl = append(fn.Dcl, n)
	return n
}

// NewDeclNameAt returns a new Name associated with symbol s at position pos.
// The caller is responsible for setting Curfn.
func NewDeclNameAt(pos src.XPos, op Op, sym *types.Sym) *Name {
	if sym == nil {
		base.Fatalf("NewDeclNameAt nil")
	}
	switch op {
	case ONAME, OTYPE, OLITERAL:
		// ok
	default:
		base.Fatalf("NewDeclNameAt op %v", op)
	}
	return newNameAt(pos, op, sym)
}

// NewConstAt returns a new OLITERAL Node associated with symbol s at position pos.
func NewConstAt(pos src.XPos, sym *types.Sym, typ *types.Type, val constant.Value) *Name {
	if sym == nil {
		base.Fatalf("NewConstAt nil")
	}
	n := newNameAt(pos, OLITERAL, sym)
	n.SetType(typ)
	n.SetTypecheck(1)
	n.SetVal(val)
	return n
}

// newNameAt is like NewNameAt but allows sym == nil.
func newNameAt(pos src.XPos, op Op, sym *types.Sym) *Name {
	n := new(Name)
	n.op = op
	n.pos = pos
	n.sym = sym
	return n
}

func (n *Name) Name() *Name            { return n }
func (n *Name) Sym() *types.Sym        { return n.sym }
func (n *Name) SetSym(x *types.Sym)    { n.sym = x }
func (n *Name) SubOp() Op              { return n.BuiltinOp }
func (n *Name) SetSubOp(x Op)          { n.BuiltinOp = x }
func (n *Name) SetFunc(x *Func)        { n.Func = x }
func (n *Name) FrameOffset() int64     { return n.Offset_ }
func (n *Name) SetFrameOffset(x int64) { n.Offset_ = x }

func (n *Name) Linksym() *obj.LSym               { return n.sym.Linksym() }
func (n *Name) LinksymABI(abi obj.ABI) *obj.LSym { return n.sym.LinksymABI(abi) }

func (*Name) CanBeNtype()    {}
func (*Name) CanBeAnSSASym() {}
func (*Name) CanBeAnSSAAux() {}

// Pragma returns the PragmaFlag for p, which must be for an OTYPE.
func (n *Name) Pragma() PragmaFlag { return n.pragma }

// SetPragma sets the PragmaFlag for p, which must be for an OTYPE.
func (n *Name) SetPragma(flag PragmaFlag) { n.pragma = flag }

// Alias reports whether p, which must be for an OTYPE, is a type alias.
func (n *Name) Alias() bool { return n.flags&nameAlias != 0 }

// SetAlias sets whether p, which must be for an OTYPE, is a type alias.
func (n *Name) SetAlias(alias bool) { n.flags.set(nameAlias, alias) }

const (
	nameReadonly                 = 1 << iota
	nameByval                    // is the variable captured by value or by reference
	nameNeedzero                 // if it contains pointers, needs to be zeroed on function entry
	nameAutoTemp                 // is the variable a temporary (implies no dwarf info. reset if escapes to heap)
	nameUsed                     // for variable declared and not used error
	nameIsClosureVar             // PAUTOHEAP closure pseudo-variable; original (if any) at n.Defn
	nameIsOutputParamHeapAddr    // pointer to a result parameter's heap copy
	nameIsOutputParamInRegisters // output parameter in registers spills as an auto
	nameAddrtaken                // address taken, even if not moved to heap
	nameInlFormal                // PAUTO created by inliner, derived from callee formal
	nameInlLocal                 // PAUTO created by inliner, derived from callee local
	nameOpenDeferSlot            // if temporary var storing info for open-coded defers
	nameLibfuzzer8BitCounter     // if PEXTERN should be assigned to __sancov_cntrs section
	nameCoverageAuxVar           // instrumentation counter var or pkg ID for cmd/cover
	nameAlias                    // is type name an alias
	nameNonMergeable             // not a candidate for stack slot merging
)

func (n *Name) Readonly() bool                 { return n.flags&nameReadonly != 0 }
func (n *Name) Needzero() bool                 { return n.flags&nameNeedzero != 0 }
func (n *Name) AutoTemp() bool                 { return n.flags&nameAutoTemp != 0 }
func (n *Name) Used() bool                     { return n.flags&nameUsed != 0 }
func (n *Name) IsClosureVar() bool             { return n.flags&nameIsClosureVar != 0 }
func (n *Name) IsOutputParamHeapAddr() bool    { return n.flags&nameIsOutputParamHeapAddr != 0 }
func (n *Name) IsOutputParamInRegisters() bool { return n.flags&nameIsOutputParamInRegisters != 0 }
func (n *Name) Addrtaken() bool                { return n.flags&nameAddrtaken != 0 }
func (n *Name) InlFormal() bool                { return n.flags&nameInlFormal != 0 }
func (n *Name) InlLocal() bool                 { return n.flags&nameInlLocal != 0 }
func (n *Name) OpenDeferSlot() bool            { return n.flags&nameOpenDeferSlot != 0 }
func (n *Name) Libfuzzer8BitCounter() bool     { return n.flags&nameLibfuzzer8BitCounter != 0 }
func (n *Name) CoverageAuxVar() bool           { return n.flags&nameCoverageAuxVar != 0 }
func (n *Name) NonMergeable() bool             { return n.flags&nameNonMergeable != 0 }

func (n *Name) setReadonly(b bool)                 { n.flags.set(nameReadonly, b) }
func (n *Name) SetNeedzero(b bool)                 { n.flags.set(nameNeedzero, b) }
func (n *Name) SetAutoTemp(b bool)                 { n.flags.set(nameAutoTemp, b) }
func (n *Name) SetUsed(b bool)                     { n.flags.set(nameUsed, b) }
func (n *Name) SetIsClosureVar(b bool)             { n.flags.set(nameIsClosureVar, b) }
func (n *Name) SetIsOutputParamHeapAddr(b bool)    { n.flags.set(nameIsOutputParamHeapAddr, b) }
func (n *Name) SetIsOutputParamInRegisters(b bool) { n.flags.set(nameIsOutputParamInRegisters, b) }
func (n *Name) SetAddrtaken(b bool)                { n.flags.set(nameAddrtaken, b) }
func (n *Name) SetInlFormal(b bool)                { n.flags.set(nameInlFormal, b) }
func (n *Name) SetInlLocal(b bool)                 { n.flags.set(nameInlLocal, b) }
func (n *Name) SetOpenDeferSlot(b bool)            { n.flags.set(nameOpenDeferSlot, b) }
func (n *Name) SetLibfuzzer8BitCounter(b bool)     { n.flags.set(nameLibfuzzer8BitCounter, b) }
func (n *Name) SetCoverageAuxVar(b bool)           { n.flags.set(nameCoverageAuxVar, b) }
func (n *Name) SetNonMergeable(b bool)             { n.flags.set(nameNonMergeable, b) }

// OnStack reports whether variable n may reside on the stack.
func (n *Name) OnStack() bool {
	if n.Op() == ONAME {
		switch n.Class {
		case PPARAM, PPARAMOUT, PAUTO:
			return n.Esc() != EscHeap
		case PEXTERN, PAUTOHEAP:
			return false
		}
	}
	// Note: fmt.go:dumpNodeHeader calls all "func() bool"-typed
	// methods, but it can only recover from panics, not Fatalf.
	panic(fmt.Sprintf("%v: not a variable: %v", base.FmtPos(n.Pos()), n))
}

// MarkReadonly indicates that n is an ONAME with readonly contents.
func (n *Name) MarkReadonly() {
	if n.Op() != ONAME {
		base.Fatalf("Node.MarkReadonly %v", n.Op())
	}
	n.setReadonly(true)
	// Mark the linksym as readonly immediately
	// so that the SSA backend can use this information.
	// It will be overridden later during dumpglobls.
	n.Linksym().Type = objabi.SRODATA
}

// Val returns the constant.Value for the node.
func (n *Name) Val() constant.Value {
	if n.val == nil {
		return constant.MakeUnknown()
	}
	return n.val
}

// SetVal sets the constant.Value for the node.
func (n *Name) SetVal(v constant.Value) {
	if n.op != OLITERAL {
		panic(n.no("SetVal"))
	}
	AssertValidTypeForConst(n.Type(), v)
	n.val = v
}

// Canonical returns the logical declaration that n represents. If n
// is a closure variable, then Canonical returns the original Name as
// it appears in the function that immediately contains the
// declaration. Otherwise, Canonical simply returns n itself.
func (n *Name) Canonical() *Name {
	if n.IsClosureVar() && n.Defn != nil {
		n = n.Defn.(*Name)
	}
	return n
}

func (n *Name) SetByval(b bool) {
	if n.Canonical() != n {
		base.Fatalf("SetByval called on non-canonical variable: %v", n)
	}
	n.flags.set(nameByval, b)
}

func (n *Name) Byval() bool {
	// We require byval to be set on the canonical variable, but we
	// allow it to be accessed from any instance.
	return n.Canonical().flags&nameByval != 0
}

// NewClosureVar returns a new closure variable for fn to refer to
// outer variable n.
func NewClosureVar(pos src.XPos, fn *Func, n *Name) *Name {
	switch n.Class {
	case PAUTO, PPARAM, PPARAMOUT, PAUTOHEAP:
		// ok
	default:
		// Prevent mistaken capture of global variables.
		base.Fatalf("NewClosureVar: %+v", n)
	}

	c := NewNameAt(pos, n.Sym(), n.Type())
	c.Curfn = fn
	c.Class = PAUTOHEAP
	c.SetIsClosureVar(true)
	c.Defn = n.Canonical()
	c.Outer = n

	fn.ClosureVars = append(fn.ClosureVars, c)

	return c
}

// NewHiddenParam returns a new hidden parameter for fn with the given
// name and type.
func NewHiddenParam(pos src.XPos, fn *Func, sym *types.Sym, typ *types.Type) *Name {
	if fn.OClosure != nil {
		base.FatalfAt(fn.Pos(), "cannot add hidden parameters to closures")
	}

	fn.SetNeedctxt(true)

	// Create a fake parameter, disassociated from any real function, to
	// pretend to capture.
	fake := NewNameAt(pos, sym, typ)
	fake.Class = PPARAM
	fake.SetByval(true)

	return NewClosureVar(pos, fn, fake)
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

//go:generate stringer -type=Class name.go
const (
	Pxxx       Class = iota // no class; used during ssa conversion to indicate pseudo-variables
	PEXTERN                 // global variables
	PAUTO                   // local variables
	PAUTOHEAP               // local variables or parameters moved to heap
	PPARAM                  // input arguments
	PPARAMOUT               // output results
	PTYPEPARAM              // type params
	PFUNC                   // global functions

	// Careful: Class is stored in three bits in Node.flags.
	_ = uint((1 << 3) - iota) // static assert for iota <= (1 << 3)
)

type Embed struct {
	Pos      src.XPos
	Patterns []string
}
