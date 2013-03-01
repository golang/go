package ssa

// Helpers for emitting SSA instructions.

import (
	"go/token"
	"go/types"
)

// emitNew emits to f a new (heap Alloc) instruction allocating an
// object of type typ.
//
func emitNew(f *Function, typ types.Type) Value {
	return f.emit(&Alloc{
		Type_: pointer(typ),
		Heap:  true,
	})
}

// emitLoad emits to f an instruction to load the address addr into a
// new temporary, and returns the value so defined.
//
func emitLoad(f *Function, addr Value) Value {
	v := &UnOp{Op: token.MUL, X: addr}
	v.setType(indirectType(addr.Type()))
	return f.emit(v)
}

// emitArith emits to f code to compute the binary operation op(x, y)
// where op is an eager shift, logical or arithmetic operation.
// (Use emitCompare() for comparisons and Builder.logicalBinop() for
// non-eager operations.)
//
func emitArith(f *Function, op token.Token, x, y Value, t types.Type) Value {
	switch op {
	case token.SHL, token.SHR:
		x = emitConv(f, x, t)
		y = emitConv(f, y, types.Typ[types.Uint64])

	case token.ADD, token.SUB, token.MUL, token.QUO, token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
		x = emitConv(f, x, t)
		y = emitConv(f, y, t)

	default:
		panic("illegal op in emitArith: " + op.String())

	}
	v := &BinOp{
		Op: op,
		X:  x,
		Y:  y,
	}
	v.setType(t)
	return f.emit(v)
}

// emitCompare emits to f code compute the boolean result of
// comparison comparison 'x op y'.
//
func emitCompare(f *Function, op token.Token, x, y Value) Value {
	xt := underlyingType(x.Type())
	yt := underlyingType(y.Type())

	// Special case to optimise a tagless SwitchStmt so that
	// these are equivalent
	//   switch { case e: ...}
	//   switch true { case e: ... }
	//   if e==true { ... }
	// even in the case when e's type is an interface.
	// TODO(adonovan): opt: generalise to x==true, false!=y, etc.
	if x == vTrue && op == token.EQL {
		if yt, ok := yt.(*types.Basic); ok && yt.Info&types.IsBoolean != 0 {
			return y
		}
	}

	if types.IsIdentical(xt, yt) {
		// no conversion necessary
	} else if _, ok := xt.(*types.Interface); ok {
		y = emitConv(f, y, x.Type())
	} else if _, ok := yt.(*types.Interface); ok {
		x = emitConv(f, x, y.Type())
	} else if _, ok := x.(*Literal); ok {
		x = emitConv(f, x, y.Type())
	} else if _, ok := y.(*Literal); ok {
		y = emitConv(f, y, x.Type())
	} else {
		// other cases, e.g. channels.  No-op.
	}

	v := &BinOp{
		Op: op,
		X:  x,
		Y:  y,
	}
	v.setType(tBool)
	return f.emit(v)
}

// emitConv emits to f code to convert Value val to exactly type typ,
// and returns the converted value.  Implicit conversions are implied
// by language assignability rules in the following operations:
//
// - from rvalue type to lvalue type in assignments.
// - from actual- to formal-parameter types in function calls.
// - from return value type to result type in return statements.
// - population of struct fields, array and slice elements, and map
//   keys and values within compoisite literals
// - from index value to index type in indexing expressions.
// - for both arguments of comparisons.
// - from value type to channel type in send expressions.
//
func emitConv(f *Function, val Value, typ types.Type) Value {
	// fmt.Printf("emitConv %s -> %s, %T", val.Type(), typ, val) // debugging

	// Identical types?  Conversion is a no-op.
	if types.IsIdentical(val.Type(), typ) {
		return val
	}

	ut_dst := underlyingType(typ)
	ut_src := underlyingType(val.Type())

	// Identical underlying types?  Conversion is a name change.
	if types.IsIdentical(ut_dst, ut_src) {
		// TODO(adonovan): make this use a distinct
		// instruction, ChangeType.  This instruction must
		// also cover the cases of channel type restrictions and
		// conversions between pointers to identical base
		// types.
		c := &Conv{X: val}
		c.setType(typ)
		return f.emit(c)
	}

	// Conversion to, or construction of a value of, an interface type?
	if _, ok := ut_dst.(*types.Interface); ok {

		// Assignment from one interface type to a different one?
		if _, ok := ut_src.(*types.Interface); ok {
			c := &ChangeInterface{X: val}
			c.setType(typ)
			return f.emit(c)
		}

		// Untyped nil literal?  Return interface-typed nil literal.
		if ut_src == tUntypedNil {
			return nilLiteral(typ)
		}

		// Convert (non-nil) "untyped" literals to their default type.
		// TODO(gri): expose types.isUntyped().
		if t, ok := ut_src.(*types.Basic); ok && t.Info&types.IsUntyped != 0 {
			val = emitConv(f, val, DefaultType(ut_src))
		}

		mi := &MakeInterface{
			X:       val,
			Methods: f.Prog.MethodSet(val.Type()),
		}
		mi.setType(typ)
		return f.emit(mi)
	}

	// Conversion of a literal to a non-interface type results in
	// a new literal of the destination type and (initially) the
	// same abstract value.  We don't compute the representation
	// change yet; this defers the point at which the number of
	// possible representations explodes.
	if l, ok := val.(*Literal); ok {
		return newLiteral(l.Value, typ)
	}

	// A representation-changing conversion.
	c := &Conv{X: val}
	c.setType(typ)
	return f.emit(c)
}

// emitStore emits to f an instruction to store value val at location
// addr, applying implicit conversions as required by assignabilty rules.
//
func emitStore(f *Function, addr, val Value) {
	f.emit(&Store{
		Addr: addr,
		Val:  emitConv(f, val, indirectType(addr.Type())),
	})
}

// emitJump emits to f a jump to target, and updates the control-flow graph.
// Postcondition: f.currentBlock is nil.
//
func emitJump(f *Function, target *BasicBlock) {
	b := f.currentBlock
	b.emit(new(Jump))
	addEdge(b, target)
	f.currentBlock = nil
}

// emitIf emits to f a conditional jump to tblock or fblock based on
// cond, and updates the control-flow graph.
// Postcondition: f.currentBlock is nil.
//
func emitIf(f *Function, cond Value, tblock, fblock *BasicBlock) {
	b := f.currentBlock
	b.emit(&If{Cond: cond})
	addEdge(b, tblock)
	addEdge(b, fblock)
	f.currentBlock = nil
}

// emitExtract emits to f an instruction to extract the index'th
// component of tuple, ascribing it type typ.  It returns the
// extracted value.
//
func emitExtract(f *Function, tuple Value, index int, typ types.Type) Value {
	e := &Extract{Tuple: tuple, Index: index}
	// In all cases but one (tSelect's recv), typ is redundant w.r.t.
	// tuple.Type().(*types.Result).Values[index].Type.
	e.setType(typ)
	return f.emit(e)
}

// emitTailCall emits to f a function call in tail position,
// passing on all but the first formal parameter to f as actual
// values in the call.  Intended for delegating bridge methods.
// Precondition: f does/will not use deferred procedure calls.
// Postcondition: f.currentBlock is nil.
//
func emitTailCall(f *Function, call *Call) {
	for _, arg := range f.Params[1:] {
		call.Args = append(call.Args, arg)
	}
	call.Type_ = &types.Result{Values: f.Signature.Results}
	tuple := f.emit(call)
	var ret Ret
	switch {
	case len(f.Signature.Results) > 1:
		for i, o := range call.Type().(*types.Result).Values {
			v := emitExtract(f, tuple, i, o.Type)
			// TODO(adonovan): in principle, this is required:
			//   v = emitConv(f, o.Type, f.Signature.Results[i].Type)
			// but in practice emitTailCall is only used when
			// the types exactly match.
			ret.Results = append(ret.Results, v)
		}
	case len(f.Signature.Results) == 1:
		ret.Results = []Value{tuple}
	default:
		// no-op
	}
	f.emit(&ret)
	f.currentBlock = nil
}
