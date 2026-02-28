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

func (*Ident) CanBeNtype() {}

// Name holds Node fields used only by named nodes (ONAME, OTYPE, some OLITERAL).
type Name struct {
	miniExpr
	BuiltinOp Op         // uint8
	Class     Class      // uint8
	pragma    PragmaFlag // int16
	flags     bitset16
	DictIndex uint16 // index of the dictionary entry describing the type of this variable declaration plus 1
	sym       *types.Sym
	Func      *Func // TODO(austin): nil for I.M, eqFor, hashfor, and hashmem
	Offset_   int64
	val       constant.Value
	Opt       interface{} // for use by escape analysis
	Embed     *[]Embed    // list of embedded files, for ONAME var

	// For a local variable (not param) or extern, the initializing assignment (OAS or OAS2).
	// For a closure var, the ONAME node of the outer captured variable.
	// For the case-local variables of a type switch, the type switch guard (OTYPESW).
	// For a range variable, the range statement (ORANGE)
	// For a recv variable in a case of a select statement, the receive assignment (OSELRECV2)
	// For the name of a function, points to corresponding Func node.
	Defn Node

	// The function, method, or closure in which local variable or param is declared.
	Curfn *Func

	Ntype    Ntype
	Heapaddr *Name // temp holding heap address of param

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

func (n *Name) isExpr() {}

func (n *Name) copy() Node                         { panic(n.no("copy")) }
func (n *Name) doChildren(do func(Node) bool) bool { return false }
func (n *Name) editChildren(edit func(Node) Node)  {}

// TypeDefn returns the type definition for a named OTYPE.
// That is, given "type T Defn", it returns Defn.
// It is used by package types.
func (n *Name) TypeDefn() *types.Type {
	if n.Ntype != nil {
		return n.Ntype.Type()
	}
	return n.Type()
}

// RecordFrameOffset records the frame offset for the name.
// It is used by package types when laying out function arguments.
func (n *Name) RecordFrameOffset(offset int64) {
	n.SetFrameOffset(offset)
}

// NewNameAt returns a new ONAME Node associated with symbol s at position pos.
// The caller is responsible for setting Curfn.
func NewNameAt(pos src.XPos, sym *types.Sym) *Name {
	if sym == nil {
		base.Fatalf("NewNameAt nil")
	}
	return newNameAt(pos, ONAME, sym)
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

func (n *Name) Name() *Name         { return n }
func (n *Name) Sym() *types.Sym     { return n.sym }
func (n *Name) SetSym(x *types.Sym) { n.sym = x }
func (n *Name) SubOp() Op           { return n.BuiltinOp }
func (n *Name) SetSubOp(x Op)       { n.BuiltinOp = x }
func (n *Name) SetFunc(x *Func)     { n.Func = x }
func (n *Name) Offset() int64       { panic("Name.Offset") }
func (n *Name) SetOffset(x int64) {
	if x != 0 {
		panic("Name.SetOffset")
	}
}
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
	nameLibfuzzerExtraCounter    // if PEXTERN should be assigned to __libfuzzer_extra_counters section
	nameAlias                    // is type name an alias
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
func (n *Name) LibfuzzerExtraCounter() bool    { return n.flags&nameLibfuzzerExtraCounter != 0 }

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
func (n *Name) SetLibfuzzerExtraCounter(b bool)    { n.flags.set(nameLibfuzzerExtraCounter, b) }

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
	c := NewNameAt(pos, n.Sym())
	c.Curfn = fn
	c.Class = PAUTOHEAP
	c.SetIsClosureVar(true)
	c.Defn = n.Canonical()
	c.Outer = n

	c.SetType(n.Type())
	c.SetTypecheck(n.Typecheck())

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
	fake := NewNameAt(pos, sym)
	fake.Class = PPARAM
	fake.SetType(typ)
	fake.SetByval(true)

	return NewClosureVar(pos, fn, fake)
}

// CaptureName returns a Name suitable for referring to n from within function
// fn or from the package block if fn is nil. If n is a free variable declared
// within a function that encloses fn, then CaptureName returns the closure
// variable that refers to n within fn, creating it if necessary.
// Otherwise, it simply returns n.
func CaptureName(pos src.XPos, fn *Func, n *Name) *Name {
	if n.Op() != ONAME || n.Curfn == nil {
		return n // okay to use directly
	}
	if n.IsClosureVar() {
		base.FatalfAt(pos, "misuse of CaptureName on closure variable: %v", n)
	}

	c := n.Innermost
	if c == nil {
		c = n
	}
	if c.Curfn == fn {
		return c
	}

	if fn == nil {
		base.FatalfAt(pos, "package-block reference to %v, declared in %v", n, n.Curfn)
	}

	// Do not have a closure var for the active closure yet; make one.
	c = NewClosureVar(pos, fn, c)

	// Link into list of active closure variables.
	// Popped from list in FinishCaptureNames.
	n.Innermost = c

	return c
}

// FinishCaptureNames handles any work leftover from calling CaptureName
// earlier. outerfn should be the function that immediately encloses fn.
func FinishCaptureNames(pos src.XPos, outerfn, fn *Func) {
	// closure-specific variables are hanging off the
	// ordinary ones; see CaptureName above.
	// unhook them.
	// make the list of pointers for the closure call.
	for _, cv := range fn.ClosureVars {
		// Unlink from n; see comment above on type Name for these fields.
		n := cv.Defn.(*Name)
		n.Innermost = cv.Outer

		// If the closure usage of n is not dense, we need to make it
		// dense by recapturing n within the enclosing function.
		//
		// That is, suppose we just finished parsing the innermost
		// closure f4 in this code:
		//
		//	func f() {
		//		n := 1
		//		func() { // f2
		//			use(n)
		//			func() { // f3
		//				func() { // f4
		//					use(n)
		//				}()
		//			}()
		//		}()
		//	}
		//
		// At this point cv.Outer is f2's n; there is no n for f3. To
		// construct the closure f4 from within f3, we need to use f3's
		// n and in this case we need to create f3's n with CaptureName.
		//
		// We'll decide later in walk whether to use v directly or &v.
		cv.Outer = CaptureName(pos, outerfn, n)
	}
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
