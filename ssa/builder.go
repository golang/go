package ssa

// This file implements the BUILD phase of SSA construction.
//
// SSA construction has two phases, CREATE and BUILD.  In the CREATE phase
// (create.go), all packages are constructed and type-checked and
// definitions of all package members are created, method-sets are
// computed, and wrapper methods are synthesized.  The create phase
// proceeds in topological order over the import dependency graph,
// initiated by client calls to CreatePackages.
//
// In the BUILD phase (builder.go), the builder traverses the AST of
// each Go source function and generates SSA instructions for the
// function body.
// Within each package, building proceeds in a topological order over
// the intra-package symbol reference graph, whose roots are the set
// of package-level declarations in lexical order.  The BUILD phases
// for distinct packages are independent and are executed in parallel.
//
// The builder's and Program's indices (maps) are populated and
// mutated during the CREATE phase, but during the BUILD phase they
// remain constant.  The sole exception is Prog.methodSets, which is
// protected by a dedicated mutex.

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"sync"
	"sync/atomic"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

type opaqueType struct {
	types.Type
	name string
}

func (t *opaqueType) String() string { return t.name }

var (
	varOk = types.NewVar(token.NoPos, nil, "ok", tBool)

	// Type constants.
	tBool       = types.Typ[types.Bool]
	tByte       = types.Typ[types.Byte]
	tInt        = types.Typ[types.Int]
	tInvalid    = types.Typ[types.Invalid]
	tUntypedNil = types.Typ[types.UntypedNil]
	tRangeIter  = &opaqueType{nil, "iter"} // the type of all "range" iterators
	tEface      = new(types.Interface)

	// The result type of a "select".
	tSelect = types.NewTuple(
		types.NewVar(token.NoPos, nil, "index", tInt),
		types.NewVar(token.NoPos, nil, "recv", tEface),
		varOk)

	// SSA Value constants.
	vZero  = intLiteral(0)
	vOne   = intLiteral(1)
	vTrue  = NewLiteral(exact.MakeBool(true), tBool)
	vFalse = NewLiteral(exact.MakeBool(false), tBool)
)

// builder holds state associated with the package currently being built.
// Its methods contain all the logic for AST-to-SSA conversion.
type builder struct {
	nTo1Vars map[*ast.ValueSpec]bool // set of n:1 ValueSpecs already built
}

// lookup returns the package-level *Function or *Global for the named
// object obj, building it if necessary.
//
// Intra-package references are edges in the initialization dependency
// graph.  If the result v is a Function or Global belonging to
// 'from', the package on whose behalf this lookup occurs, then lookup
// emits initialization code into from.Init if not already done.
//
func (b *builder) lookup(from *Package, obj types.Object) Value {
	v := from.Prog.Value(obj)
	switch v := v.(type) {
	case *Function:
		if from == v.Pkg {
			b.buildFunction(v)
		}
	case *Global:
		if from == v.Pkg {
			b.buildGlobal(v, obj)
		}
	}
	return v
}

// cond emits to fn code to evaluate boolean condition e and jump
// to t or f depending on its value, performing various simplifications.
//
// Postcondition: fn.currentBlock is nil.
//
func (b *builder) cond(fn *Function, e ast.Expr, t, f *BasicBlock) {
	switch e := e.(type) {
	case *ast.ParenExpr:
		b.cond(fn, e.X, t, f)
		return

	case *ast.BinaryExpr:
		switch e.Op {
		case token.LAND:
			ltrue := fn.newBasicBlock("cond.true")
			b.cond(fn, e.X, ltrue, f)
			fn.currentBlock = ltrue
			b.cond(fn, e.Y, t, f)
			return

		case token.LOR:
			lfalse := fn.newBasicBlock("cond.false")
			b.cond(fn, e.X, t, lfalse)
			fn.currentBlock = lfalse
			b.cond(fn, e.Y, t, f)
			return
		}

	case *ast.UnaryExpr:
		if e.Op == token.NOT {
			b.cond(fn, e.X, f, t)
			return
		}
	}

	switch cond := b.expr(fn, e).(type) {
	case *Literal:
		// Dispatch constant conditions statically.
		if exact.BoolVal(cond.Value) {
			emitJump(fn, t)
		} else {
			emitJump(fn, f)
		}
	default:
		emitIf(fn, cond, t, f)
	}
}

// logicalBinop emits code to fn to evaluate e, a &&- or
// ||-expression whose reified boolean value is wanted.
// The value is returned.
//
func (b *builder) logicalBinop(fn *Function, e *ast.BinaryExpr) Value {
	rhs := fn.newBasicBlock("binop.rhs")
	done := fn.newBasicBlock("binop.done")

	var short Value // value of the short-circuit path
	switch e.Op {
	case token.LAND:
		b.cond(fn, e.X, rhs, done)
		short = vFalse
	case token.LOR:
		b.cond(fn, e.X, done, rhs)
		short = vTrue
	}

	// Is rhs unreachable?
	if rhs.Preds == nil {
		// Simplify false&&y to false, true||y to true.
		fn.currentBlock = done
		return short
	}

	// Is done unreachable?
	if done.Preds == nil {
		// Simplify true&&y (or false||y) to y.
		fn.currentBlock = rhs
		return b.expr(fn, e.Y)
	}

	// All edges from e.X to done carry the short-circuit value.
	var edges []Value
	for _ = range done.Preds {
		edges = append(edges, short)
	}

	// The edge from e.Y to done carries the value of e.Y.
	fn.currentBlock = rhs
	edges = append(edges, b.expr(fn, e.Y))
	emitJump(fn, done)
	fn.currentBlock = done

	phi := &Phi{Edges: edges, Comment: e.Op.String()}
	phi.pos = e.OpPos
	phi.typ = phi.Edges[0].Type()
	return done.emit(phi)
}

// exprN lowers a multi-result expression e to SSA form, emitting code
// to fn and returning a single Value whose type is a *types.Tuple.
// The caller must access the components via Extract.
//
// Multi-result expressions include CallExprs in a multi-value
// assignment or return statement, and "value,ok" uses of
// TypeAssertExpr, IndexExpr (when X is a map), and UnaryExpr (when Op
// is token.ARROW).
//
func (b *builder) exprN(fn *Function, e ast.Expr) Value {
	var typ types.Type
	var tuple Value
	switch e := e.(type) {
	case *ast.ParenExpr:
		return b.exprN(fn, e.X)

	case *ast.CallExpr:
		// Currently, no built-in function nor type conversion
		// has multiple results, so we can avoid some of the
		// cases for single-valued CallExpr.
		var c Call
		b.setCall(fn, e, &c.Call)
		c.typ = fn.Pkg.typeOf(e)
		return fn.emit(&c)

	case *ast.IndexExpr:
		mapt := fn.Pkg.typeOf(e.X).Underlying().(*types.Map)
		typ = mapt.Elem()
		lookup := &Lookup{
			X:       b.expr(fn, e.X),
			Index:   emitConv(fn, b.expr(fn, e.Index), mapt.Key()),
			CommaOk: true,
		}
		lookup.setPos(e.Lbrack)
		tuple = fn.emit(lookup)

	case *ast.TypeAssertExpr:
		return emitTypeTest(fn, b.expr(fn, e.X), fn.Pkg.typeOf(e), e.Lparen)

	case *ast.UnaryExpr: // must be receive <-
		typ = fn.Pkg.typeOf(e.X).Underlying().(*types.Chan).Elem()
		unop := &UnOp{
			Op:      token.ARROW,
			X:       b.expr(fn, e.X),
			CommaOk: true,
		}
		unop.setPos(e.OpPos)
		tuple = fn.emit(unop)

	default:
		panic(fmt.Sprintf("unexpected exprN: %T", e))
	}

	// The typechecker sets the type of the expression to just the
	// asserted type in the "value, ok" form, not to *types.Tuple
	// (though it includes the valueOk operand in its error messages).

	tuple.(interface {
		setType(types.Type)
	}).setType(types.NewTuple(
		types.NewVar(token.NoPos, nil, "value", typ),
		varOk,
	))
	return tuple
}

// builtin emits to fn SSA instructions to implement a call to the
// built-in function called name with the specified arguments
// and return type.  It returns the value defined by the result.
//
// The result is nil if no special handling was required; in this case
// the caller should treat this like an ordinary library function
// call.
//
func (b *builder) builtin(fn *Function, name string, args []ast.Expr, typ types.Type, pos token.Pos) Value {
	switch name {
	case "make":
		switch typ.Underlying().(type) {
		case *types.Slice:
			n := b.expr(fn, args[1])
			m := n
			if len(args) == 3 {
				m = b.expr(fn, args[2])
			}
			v := &MakeSlice{
				Len: n,
				Cap: m,
			}
			v.setPos(pos)
			v.setType(typ)
			return fn.emit(v)

		case *types.Map:
			var res Value
			if len(args) == 2 {
				res = b.expr(fn, args[1])
			}
			v := &MakeMap{Reserve: res}
			v.setPos(pos)
			v.setType(typ)
			return fn.emit(v)

		case *types.Chan:
			var sz Value = vZero
			if len(args) == 2 {
				sz = b.expr(fn, args[1])
			}
			v := &MakeChan{Size: sz}
			v.setPos(pos)
			v.setType(typ)
			return fn.emit(v)
		}

	case "new":
		return emitNew(fn, typ.Underlying().Deref(), pos)

	case "len", "cap":
		// Special case: len or cap of an array or *array is
		// based on the type, not the value which may be nil.
		// We must still evaluate the value, though.  (If it
		// was side-effect free, the whole call would have
		// been constant-folded.)
		t := fn.Pkg.typeOf(args[0]).Deref().Underlying()
		if at, ok := t.(*types.Array); ok {
			b.expr(fn, args[0]) // for effects only
			return intLiteral(at.Len())
		}
		// Otherwise treat as normal.

	case "panic":
		fn.emit(&Panic{
			X:   emitConv(fn, b.expr(fn, args[0]), tEface),
			pos: pos,
		})
		fn.currentBlock = fn.newBasicBlock("unreachable")
		return vFalse // any non-nil Value will do
	}
	return nil // treat all others as a regular function call
}

// selector evaluates the selector expression e and returns its value,
// or if wantAddr is true, its address, in which case escaping
// indicates whether the caller intends to use the resulting pointer
// in a potentially escaping way.
//
func (b *builder) selector(fn *Function, e *ast.SelectorExpr, wantAddr, escaping bool) Value {
	id := MakeId(e.Sel.Name, fn.Pkg.Types)

	// Bound method closure?  (e.m where m is a method)
	if !wantAddr {
		if m, recv := b.findMethod(fn, e.X, id); m != nil {
			c := &MakeClosure{
				Fn:       boundMethodWrapper(m),
				Bindings: []Value{recv},
			}
			c.setPos(e.Sel.Pos())
			c.setType(fn.Pkg.typeOf(e))
			return fn.emit(c)
		}
	}

	st := fn.Pkg.typeOf(e.X).Deref().Underlying().(*types.Struct)
	index := -1
	for i, n := 0, st.NumFields(); i < n; i++ {
		f := st.Field(i)
		if MakeId(f.Name(), f.Pkg()) == id {
			index = i
			break
		}
	}
	var path *anonFieldPath
	if index == -1 {
		// Not a named field.  Use breadth-first algorithm.
		path, index = findPromotedField(st, id)
		if path == nil {
			panic("field not found, even with promotion: " + e.Sel.Name)
		}
	}
	fieldType := fn.Pkg.typeOf(e)
	pos := e.Sel.Pos()
	if wantAddr {
		return b.fieldAddr(fn, e.X, path, index, fieldType, pos, escaping)
	}
	return b.fieldExpr(fn, e.X, path, index, fieldType, pos)
}

// fieldAddr evaluates the base expression (a struct or *struct),
// applies to it any implicit field selections from path, and then
// selects the field #index of type fieldType.
// Its address is returned.
//
// (fieldType can be derived from base+index.)
//
func (b *builder) fieldAddr(fn *Function, base ast.Expr, path *anonFieldPath, index int, fieldType types.Type, pos token.Pos, escaping bool) Value {
	var x Value
	if path != nil {
		switch path.field.Type().Underlying().(type) {
		case *types.Struct:
			x = b.fieldAddr(fn, base, path.tail, path.index, path.field.Type(), token.NoPos, escaping)
		case *types.Pointer:
			x = b.fieldExpr(fn, base, path.tail, path.index, path.field.Type(), token.NoPos)
		}
	} else {
		switch fn.Pkg.typeOf(base).Underlying().(type) {
		case *types.Struct:
			x = b.addr(fn, base, escaping).(address).addr
		case *types.Pointer:
			x = b.expr(fn, base)
		}
	}
	v := &FieldAddr{
		X:     x,
		Field: index,
	}
	v.setPos(pos)
	v.setType(pointer(fieldType))
	return fn.emit(v)
}

// fieldExpr evaluates the base expression (a struct or *struct),
// applies to it any implicit field selections from path, and then
// selects the field #index of type fieldType.
// Its value is returned.
//
// (fieldType can be derived from base+index.)
//
func (b *builder) fieldExpr(fn *Function, base ast.Expr, path *anonFieldPath, index int, fieldType types.Type, pos token.Pos) Value {
	var x Value
	if path != nil {
		x = b.fieldExpr(fn, base, path.tail, path.index, path.field.Type(), token.NoPos)
	} else {
		x = b.expr(fn, base)
	}
	switch x.Type().Underlying().(type) {
	case *types.Struct:
		v := &Field{
			X:     x,
			Field: index,
		}
		v.setPos(pos)
		v.setType(fieldType)
		return fn.emit(v)

	case *types.Pointer: // *struct
		v := &FieldAddr{
			X:     x,
			Field: index,
		}
		v.setPos(pos)
		v.setType(pointer(fieldType))
		return emitLoad(fn, fn.emit(v))
	}
	panic("unreachable")
}

// addr lowers a single-result addressable expression e to SSA form,
// emitting code to fn and returning the location (an lvalue) defined
// by the expression.
//
// If escaping is true, addr marks the base variable of the
// addressable expression e as being a potentially escaping pointer
// value.  For example, in this code:
//
//   a := A{
//     b: [1]B{B{c: 1}}
//   }
//   return &a.b[0].c
//
// the application of & causes a.b[0].c to have its address taken,
// which means that ultimately the local variable a must be
// heap-allocated.  This is a simple but very conservative escape
// analysis.
//
// Operations forming potentially escaping pointers include:
// - &x, including when implicit in method call or composite literals.
// - a[:] iff a is an array (not *array)
// - references to variables in lexically enclosing functions.
//
func (b *builder) addr(fn *Function, e ast.Expr, escaping bool) lvalue {
	switch e := e.(type) {
	case *ast.Ident:
		obj := fn.Pkg.objectOf(e)
		v := b.lookup(fn.Pkg, obj) // var (address)
		if v == nil {
			v = fn.lookup(obj, escaping)
		}
		return address{addr: v}

	case *ast.CompositeLit:
		t := fn.Pkg.typeOf(e).Deref()
		var v Value
		if escaping {
			v = emitNew(fn, t, e.Lbrace)
		} else {
			v = fn.addLocal(t, e.Lbrace)
		}
		b.compLit(fn, v, e, t) // initialize in place
		return address{addr: v}

	case *ast.ParenExpr:
		return b.addr(fn, e.X, escaping)

	case *ast.SelectorExpr:
		// p.M where p is a package.
		if obj := fn.Pkg.info.IsPackageRef(e); obj != nil {
			if v := b.lookup(fn.Pkg, obj); v != nil {
				return address{addr: v}
			}
			panic("undefined package-qualified name: " + obj.Name())
		}

		// e.f where e is an expression.
		return address{addr: b.selector(fn, e, true, escaping)}

	case *ast.IndexExpr:
		var x Value
		var et types.Type
		switch t := fn.Pkg.typeOf(e.X).Underlying().(type) {
		case *types.Array:
			x = b.addr(fn, e.X, escaping).(address).addr
			et = pointer(t.Elem())
		case *types.Pointer: // *array
			x = b.expr(fn, e.X)
			et = pointer(t.Elem().Underlying().(*types.Array).Elem())
		case *types.Slice:
			x = b.expr(fn, e.X)
			et = pointer(t.Elem())
		case *types.Map:
			return &element{
				m: b.expr(fn, e.X),
				k: emitConv(fn, b.expr(fn, e.Index), t.Key()),
				t: t.Elem(),
			}
		default:
			panic("unexpected container type in IndexExpr: " + t.String())
		}
		v := &IndexAddr{
			X:     x,
			Index: emitConv(fn, b.expr(fn, e.Index), tInt),
		}
		v.setType(et)
		return address{addr: fn.emit(v)}

	case *ast.StarExpr:
		return address{addr: b.expr(fn, e.X), star: e.Star}
	}

	panic(fmt.Sprintf("unexpected address expression: %T", e))
}

// exprInPlace emits to fn code to initialize the lvalue loc with the
// value of expression e.
//
// This is equivalent to loc.store(fn, b.expr(fn, e)) but may
// generate better code in some cases, e.g. for composite literals
// in an addressable location.
//
func (b *builder) exprInPlace(fn *Function, loc lvalue, e ast.Expr) {
	if addr, ok := loc.(address); ok {
		if e, ok := e.(*ast.CompositeLit); ok {
			typ := addr.typ()
			switch typ.Underlying().(type) {
			case *types.Pointer: // implicit & -- possibly escaping
				ptr := b.addr(fn, e, true).(address).addr
				addr.store(fn, ptr) // copy address
				return

			case *types.Interface:
				// e.g. var x interface{} = T{...}
				// Can't in-place initialize an interface value.
				// Fall back to copying.

			default:
				b.compLit(fn, addr.addr, e, typ) // in place
				return
			}
		}
	}
	loc.store(fn, b.expr(fn, e)) // copy value
}

// expr lowers a single-result expression e to SSA form, emitting code
// to fn and returning the Value defined by the expression.
//
func (b *builder) expr(fn *Function, e ast.Expr) Value {
	if v := fn.Pkg.info.ValueOf(e); v != nil {
		return NewLiteral(v, fn.Pkg.typeOf(e))
	}

	switch e := e.(type) {
	case *ast.BasicLit:
		panic("non-constant BasicLit") // unreachable

	case *ast.FuncLit:
		posn := fn.Prog.Files.Position(e.Type.Func)
		fn2 := &Function{
			name:      fmt.Sprintf("func@%d.%d", posn.Line, posn.Column),
			Signature: fn.Pkg.typeOf(e.Type).Underlying().(*types.Signature),
			pos:       e.Type.Func,
			Enclosing: fn,
			Pkg:       fn.Pkg,
			Prog:      fn.Prog,
			syntax: &funcSyntax{
				paramFields:  e.Type.Params,
				resultFields: e.Type.Results,
				body:         e.Body,
			},
		}
		fn.AnonFuncs = append(fn.AnonFuncs, fn2)
		b.buildFunction(fn2)
		if fn2.FreeVars == nil {
			return fn2
		}
		v := &MakeClosure{Fn: fn2}
		v.setType(fn.Pkg.typeOf(e))
		for _, fv := range fn2.FreeVars {
			v.Bindings = append(v.Bindings, fv.outer)
			fv.outer = nil
		}
		return fn.emit(v)

	case *ast.ParenExpr:
		return b.expr(fn, e.X)

	case *ast.TypeAssertExpr: // single-result form only
		return emitTypeAssert(fn, b.expr(fn, e.X), fn.Pkg.typeOf(e), e.Lparen)

	case *ast.CallExpr:
		typ := fn.Pkg.typeOf(e)
		if fn.Pkg.info.IsType(e.Fun) {
			// Explicit type conversion, e.g. string(x) or big.Int(x)
			x := b.expr(fn, e.Args[0])
			y := emitConv(fn, x, typ)
			if y != x {
				switch y := y.(type) {
				case *Convert:
					y.pos = e.Lparen
				case *ChangeType:
					y.pos = e.Lparen
				case *MakeInterface:
					y.pos = e.Lparen
				}
			}
			return y
		}
		// Call to "intrinsic" built-ins, e.g. new, make, panic.
		if id, ok := e.Fun.(*ast.Ident); ok {
			obj := fn.Pkg.objectOf(id)
			if _, ok := fn.Prog.Builtins[obj]; ok {
				if v := b.builtin(fn, id.Name, e.Args, typ, e.Lparen); v != nil {
					return v
				}
			}
		}
		// Regular function call.
		var v Call
		b.setCall(fn, e, &v.Call)
		v.setType(typ)
		return fn.emit(&v)

	case *ast.UnaryExpr:
		switch e.Op {
		case token.AND: // &X --- potentially escaping.
			return b.addr(fn, e.X, true).(address).addr
		case token.ADD:
			return b.expr(fn, e.X)
		case token.NOT, token.ARROW, token.SUB, token.XOR: // ! <- - ^
			v := &UnOp{
				Op: e.Op,
				X:  b.expr(fn, e.X),
			}
			v.setPos(e.OpPos)
			v.setType(fn.Pkg.typeOf(e))
			return fn.emit(v)
		default:
			panic(e.Op)
		}

	case *ast.BinaryExpr:
		switch e.Op {
		case token.LAND, token.LOR:
			return b.logicalBinop(fn, e)
		case token.SHL, token.SHR:
			fallthrough
		case token.ADD, token.SUB, token.MUL, token.QUO, token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
			return emitArith(fn, e.Op, b.expr(fn, e.X), b.expr(fn, e.Y), fn.Pkg.typeOf(e), e.OpPos)

		case token.EQL, token.NEQ, token.GTR, token.LSS, token.LEQ, token.GEQ:
			return emitCompare(fn, e.Op, b.expr(fn, e.X), b.expr(fn, e.Y), e.OpPos)
		default:
			panic("illegal op in BinaryExpr: " + e.Op.String())
		}

	case *ast.SliceExpr:
		var low, high Value
		var x Value
		switch fn.Pkg.typeOf(e.X).Underlying().(type) {
		case *types.Array:
			// Potentially escaping.
			x = b.addr(fn, e.X, true).(address).addr
		case *types.Basic, *types.Slice, *types.Pointer: // *array
			x = b.expr(fn, e.X)
		default:
			unreachable()
		}
		if e.High != nil {
			high = b.expr(fn, e.High)
		}
		if e.Low != nil {
			low = b.expr(fn, e.Low)
		}
		v := &Slice{
			X:    x,
			Low:  low,
			High: high,
		}
		v.setPos(e.Lbrack)
		v.setType(fn.Pkg.typeOf(e))
		return fn.emit(v)

	case *ast.Ident:
		obj := fn.Pkg.objectOf(e)
		// Universal built-in?
		if obj.Pkg() == nil {
			return fn.Prog.Builtins[obj]
		}
		// Package-level func or var?
		if v := b.lookup(fn.Pkg, obj); v != nil {
			if _, ok := obj.(*types.Var); ok {
				return emitLoad(fn, v) // var (address)
			}
			return v // (func)
		}
		// Local?
		return emitLoad(fn, fn.lookup(obj, false)) // var (address)

	case *ast.SelectorExpr:
		// p.M where p is a package.
		if obj := fn.Pkg.info.IsPackageRef(e); obj != nil {
			return b.expr(fn, e.Sel)
		}

		// (*T).f or T.f, the method f from the method-set of type T.
		if fn.Pkg.info.IsType(e.X) {
			id := MakeId(e.Sel.Name, fn.Pkg.Types)
			typ := fn.Pkg.typeOf(e.X)
			if m := fn.Prog.MethodSet(typ)[id]; m != nil {
				return m
			}

			// T must be an interface; return wrapper.
			return interfaceMethodWrapper(fn.Prog, typ, id)
		}

		// e.f where e is an expression.  f may be a method.
		return b.selector(fn, e, false, false)

	case *ast.IndexExpr:
		switch t := fn.Pkg.typeOf(e.X).Underlying().(type) {
		case *types.Array:
			// Non-addressable array (in a register).
			v := &Index{
				X:     b.expr(fn, e.X),
				Index: emitConv(fn, b.expr(fn, e.Index), tInt),
			}
			v.setType(t.Elem())
			return fn.emit(v)

		case *types.Map:
			// Maps are not addressable.
			mapt := fn.Pkg.typeOf(e.X).Underlying().(*types.Map)
			v := &Lookup{
				X:     b.expr(fn, e.X),
				Index: emitConv(fn, b.expr(fn, e.Index), mapt.Key()),
			}
			v.setPos(e.Lbrack)
			v.setType(mapt.Elem())
			return fn.emit(v)

		case *types.Basic: // => string
			// Strings are not addressable.
			v := &Lookup{
				X:     b.expr(fn, e.X),
				Index: b.expr(fn, e.Index),
			}
			v.setPos(e.Lbrack)
			v.setType(tByte)
			return fn.emit(v)

		case *types.Slice, *types.Pointer: // *array
			// Addressable slice/array; use IndexAddr and Load.
			return b.addr(fn, e, false).load(fn)

		default:
			panic("unexpected container type in IndexExpr: " + t.String())
		}

	case *ast.CompositeLit, *ast.StarExpr:
		// Addressable types (lvalues)
		return b.addr(fn, e, false).load(fn)
	}

	panic(fmt.Sprintf("unexpected expr: %T", e))
}

// stmtList emits to fn code for all statements in list.
func (b *builder) stmtList(fn *Function, list []ast.Stmt) {
	for _, s := range list {
		b.stmt(fn, s)
	}
}

// findMethod returns the method and receiver for a call base.id().
// It locates the method using the method-set for base's type,
// and emits code for the receiver, handling the cases where
// the formal and actual parameter's pointerness are unequal.
//
// findMethod returns (nil, nil) if no such method was found.
//
func (b *builder) findMethod(fn *Function, base ast.Expr, id Id) (*Function, Value) {
	typ := fn.Pkg.typeOf(base)

	// Consult method-set of X.
	if m := fn.Prog.MethodSet(typ)[id]; m != nil {
		aptr := isPointer(typ)
		fptr := isPointer(m.Signature.Recv().Type())
		if aptr == fptr {
			// Actual's and formal's "pointerness" match.
			return m, b.expr(fn, base)
		}
		// Actual is a pointer, formal is not.
		// Load a copy.
		return m, emitLoad(fn, b.expr(fn, base))
	}
	if !isPointer(typ) {
		// Consult method-set of *X.
		if m := fn.Prog.MethodSet(pointer(typ))[id]; m != nil {
			// A method found only in MS(*X) must have a
			// pointer formal receiver; but the actual
			// value is not a pointer.
			// Implicit & -- possibly escaping.
			return m, b.addr(fn, base, true).(address).addr
		}
	}
	return nil, nil
}

// setCallFunc populates the function parts of a CallCommon structure
// (Func, Method, Recv, Args[0]) based on the kind of invocation
// occurring in e.
//
func (b *builder) setCallFunc(fn *Function, e *ast.CallExpr, c *CallCommon) {
	c.pos = e.Lparen
	c.HasEllipsis = e.Ellipsis != 0

	// Is the call of the form x.f()?
	sel, ok := unparen(e.Fun).(*ast.SelectorExpr)

	// Case 0: e.Fun evaluates normally to a function.
	if !ok {
		c.Func = b.expr(fn, e.Fun)
		return
	}

	// Case 1: call of form x.F() where x is a package name.
	if obj := fn.Pkg.info.IsPackageRef(sel); obj != nil {
		// This is a specialization of expr(ast.Ident(obj)).
		if v := b.lookup(fn.Pkg, obj); v != nil {
			if _, ok := v.(*Function); !ok {
				v = emitLoad(fn, v) // var (address)
			}
			c.Func = v
			return
		}
		panic("undefined package-qualified name: " + obj.Name())
	}

	// Case 2a: X.f() or (*X).f(): a statically dipatched call to
	// the method f in the method-set of X or *X.  X may be
	// an interface.  Treat like case 0.
	// TODO(adonovan): opt: inline expr() here, to make the call static
	// and to avoid generation of a stub for an interface method.
	if fn.Pkg.info.IsType(sel.X) {
		c.Func = b.expr(fn, e.Fun)
		return
	}

	id := MakeId(sel.Sel.Name, fn.Pkg.Types)

	// Let X be the type of x.

	// Case 2: x.f(): a statically dispatched call to a method
	// from the method-set of X or perhaps *X (if x is addressable
	// but not a pointer).
	if m, recv := b.findMethod(fn, sel.X, id); m != nil {
		c.Func = m
		c.Args = append(c.Args, recv)
		return
	}

	switch t := fn.Pkg.typeOf(sel.X).Underlying().(type) {
	case *types.Struct, *types.Pointer:
		// Case 3: x.f() where x.f is a function value in a
		// struct field f; not a method call.  f is a 'var'
		// (of function type) in the Fields of types.Struct X.
		// Treat like case 0.
		c.Func = b.expr(fn, e.Fun)

	case *types.Interface:
		// Case 4: x.f() where a dynamically dispatched call
		// to an interface method f.  f is a 'func' object in
		// the Methods of types.Interface X
		c.Method, _ = interfaceMethodIndex(t, id)
		c.Recv = b.expr(fn, sel.X)

	default:
		panic(fmt.Sprintf("illegal (%s).%s() call; X:%T", t, sel.Sel.Name, sel.X))
	}
}

// emitCallArgs emits to f code for the actual parameters of call e to
// a (possibly built-in) function of effective type sig.
// The argument values are appended to args, which is then returned.
//
func (b *builder) emitCallArgs(fn *Function, sig *types.Signature, e *ast.CallExpr, args []Value) []Value {
	// f(x, y, z...): pass slice z straight through.
	if e.Ellipsis != 0 {
		for i, arg := range e.Args {
			// TODO(gri): annoyingly Signature.Params doesn't
			// reflect the slice type for a final ...T param.
			t := sig.Params().At(i).Type()
			if sig.IsVariadic() && i == len(e.Args)-1 {
				t = types.NewSlice(t)
			}
			args = append(args, emitConv(fn, b.expr(fn, arg), t))
		}
		return args
	}

	offset := len(args) // 1 if call has receiver, 0 otherwise

	// Evaluate actual parameter expressions.
	//
	// If this is a chained call of the form f(g()) where g has
	// multiple return values (MRV), they are flattened out into
	// args; a suffix of them may end up in a varargs slice.
	for _, arg := range e.Args {
		v := b.expr(fn, arg)
		if ttuple, ok := v.Type().(*types.Tuple); ok { // MRV chain
			for i, n := 0, ttuple.Len(); i < n; i++ {
				args = append(args, emitExtract(fn, v, i, ttuple.At(i).Type()))
			}
		} else {
			args = append(args, v)
		}
	}

	// Actual->formal assignability conversions for normal parameters.
	np := sig.Params().Len() // number of normal parameters
	if sig.IsVariadic() {
		np--
	}
	for i := 0; i < np; i++ {
		args[offset+i] = emitConv(fn, args[offset+i], sig.Params().At(i).Type())
	}

	// Actual->formal assignability conversions for variadic parameter,
	// and construction of slice.
	if sig.IsVariadic() {
		varargs := args[offset+np:]
		vt := sig.Params().At(np).Type()
		st := types.NewSlice(vt)
		if len(varargs) == 0 {
			args = append(args, nilLiteral(st))
		} else {
			// Replace a suffix of args with a slice containing it.
			at := types.NewArray(vt, int64(len(varargs)))
			// Don't set pos (e.g. to e.Lparen) for implicit Allocs.
			a := emitNew(fn, at, token.NoPos)
			for i, arg := range varargs {
				iaddr := &IndexAddr{
					X:     a,
					Index: intLiteral(int64(i)),
				}
				iaddr.setType(pointer(vt))
				fn.emit(iaddr)
				emitStore(fn, iaddr, arg)
			}
			s := &Slice{X: a}
			s.setType(st)
			args[offset+np] = fn.emit(s)
			args = args[:offset+np+1]
		}
	}
	return args
}

// setCall emits to fn code to evaluate all the parameters of a function
// call e, and populates *c with those values.
//
func (b *builder) setCall(fn *Function, e *ast.CallExpr, c *CallCommon) {
	// First deal with the f(...) part and optional receiver.
	b.setCallFunc(fn, e, c)

	// Then append the other actual parameters.
	sig, _ := fn.Pkg.typeOf(e.Fun).Underlying().(*types.Signature)
	if sig == nil {
		sig = fn.Pkg.info.BuiltinCallSignature(e)
	}
	c.Args = b.emitCallArgs(fn, sig, e, c.Args)
}

// assignOp emits to fn code to perform loc += incr or loc -= incr.
func (b *builder) assignOp(fn *Function, loc lvalue, incr Value, op token.Token) {
	oldv := loc.load(fn)
	loc.store(fn, emitArith(fn, op, oldv, emitConv(fn, incr, oldv.Type()), loc.typ(), token.NoPos))
}

// buildGlobal emits code to the g.Pkg.Init function for the variable
// definition(s) of g.  Effects occur out of lexical order; see
// explanation at globalValueSpec.
// Precondition: g == g.Prog.Value(obj)
//
func (b *builder) buildGlobal(g *Global, obj types.Object) {
	spec := g.spec
	if spec == nil {
		return // already built (or in progress)
	}
	b.globalValueSpec(g.Pkg.Init, spec, g, obj)
}

// globalValueSpec emits to init code to define one or all of the vars
// in the package-level ValueSpec spec.
//
// It implements the build phase for a ValueSpec, ensuring that all
// vars are initialized if not already visited by buildGlobal during
// the reference graph traversal.
//
// This function may be called in two modes:
// A) with g and obj non-nil, to initialize just a single global.
//    This occurs during the reference graph traversal.
// B) with g and obj nil, to initialize all globals in the same ValueSpec.
//    This occurs during the left-to-right traversal over the ast.File.
//
// Precondition: g == g.Prog.Value(obj)
//
// Package-level var initialization order is quite subtle.
// The side effects of:
//   var a, b = f(), g()
// are not observed left-to-right if b is referenced before a in the
// reference graph traversal.  So, we track which Globals have been
// initialized by setting Global.spec=nil.
//
// Blank identifiers make things more complex since they don't have
// associated types.Objects or ssa.Globals yet we must still ensure
// that their corresponding side effects are observed at the right
// moment.  Consider:
//   var a, _, b = f(), g(), h()
// Here, the relative ordering of the call to g() is unspecified but
// it must occur exactly once, during mode B.  So globalValueSpec for
// blanks must special-case n:n assigments and just evaluate the RHS
// g() for effect.
//
// In a n:1 assignment:
//   var a, _, b = f()
// a reference to either a or b causes both globals to be initialized
// at the same time.  Furthermore, no further work is required to
// ensure that the effects of the blank assignment occur.  We must
// keep track of which n:1 specs have been evaluated, independent of
// which Globals are on the LHS (possibly none, if all are blank).
//
// See also localValueSpec.
//
func (b *builder) globalValueSpec(init *Function, spec *ast.ValueSpec, g *Global, obj types.Object) {
	switch {
	case len(spec.Values) == len(spec.Names):
		// e.g. var x, y = 0, 1
		// 1:1 assignment.
		// Only the first time for a given GLOBAL has any effect.
		for i, id := range spec.Names {
			var lval lvalue = blank{}
			if g != nil {
				// Mode A: initialized only a single global, g
				if isBlankIdent(id) || init.Pkg.objectOf(id) != obj {
					continue
				}
				g.spec = nil
				lval = address{addr: g}
			} else {
				// Mode B: initialize all globals.
				if !isBlankIdent(id) {
					g2 := init.Prog.Value(init.Pkg.objectOf(id)).(*Global)
					if g2.spec == nil {
						continue // already done
					}
					g2.spec = nil
					lval = address{addr: g2}
				}
			}
			if init.Prog.mode&LogSource != 0 {
				fmt.Fprintln(os.Stderr, "build global", id.Name)
			}
			b.exprInPlace(init, lval, spec.Values[i])
			if g != nil {
				break
			}
		}

	case len(spec.Values) == 0:
		// e.g. var x, y int
		// Globals are implicitly zero-initialized.

	default:
		// e.g. var x, _, y = f()
		// n:1 assignment.
		// Only the first time for a given SPEC has any effect.
		if !b.nTo1Vars[spec] {
			b.nTo1Vars[spec] = true
			if init.Prog.mode&LogSource != 0 {
				defer logStack("build globals %s", spec.Names)()
			}
			tuple := b.exprN(init, spec.Values[0])
			result := tuple.Type().(*types.Tuple)
			for i, id := range spec.Names {
				if !isBlankIdent(id) {
					g := init.Prog.Value(init.Pkg.objectOf(id)).(*Global)
					g.spec = nil // just an optimization
					emitStore(init, g, emitExtract(init, tuple, i, result.At(i).Type()))
				}
			}
		}
	}
}

// localValueSpec emits to fn code to define all of the vars in the
// function-local ValueSpec, spec.
//
// See also globalValueSpec: the two routines are similar but local
// ValueSpecs are much simpler since they are encountered once only,
// in their entirety, in lexical order.
//
func (b *builder) localValueSpec(fn *Function, spec *ast.ValueSpec) {
	switch {
	case len(spec.Values) == len(spec.Names):
		// e.g. var x, y = 0, 1
		// 1:1 assignment
		for i, id := range spec.Names {
			var lval lvalue = blank{}
			if !isBlankIdent(id) {
				lval = address{addr: fn.addNamedLocal(fn.Pkg.objectOf(id))}
			}
			b.exprInPlace(fn, lval, spec.Values[i])
		}

	case len(spec.Values) == 0:
		// e.g. var x, y int
		// Locals are implicitly zero-initialized.
		for _, id := range spec.Names {
			if !isBlankIdent(id) {
				fn.addNamedLocal(fn.Pkg.objectOf(id))
			}
		}

	default:
		// e.g. var x, y = pos()
		tuple := b.exprN(fn, spec.Values[0])
		result := tuple.Type().(*types.Tuple)
		for i, id := range spec.Names {
			if !isBlankIdent(id) {
				lhs := fn.addNamedLocal(fn.Pkg.objectOf(id))
				emitStore(fn, lhs, emitExtract(fn, tuple, i, result.At(i).Type()))
			}
		}
	}
}

// assignStmt emits code to fn for a parallel assignment of rhss to lhss.
// isDef is true if this is a short variable declaration (:=).
//
// Note the similarity with localValueSpec.
//
func (b *builder) assignStmt(fn *Function, lhss, rhss []ast.Expr, isDef bool) {
	// Side effects of all LHSs and RHSs must occur in left-to-right order.
	var lvals []lvalue
	for _, lhs := range lhss {
		var lval lvalue = blank{}
		if !isBlankIdent(lhs) {
			if isDef {
				// Local may be "redeclared" in the same
				// scope, so don't blindly create anew.
				obj := fn.Pkg.objectOf(lhs.(*ast.Ident))
				if _, ok := fn.objects[obj]; !ok {
					fn.addNamedLocal(obj)
				}
			}
			lval = b.addr(fn, lhs, false) // non-escaping
		}
		lvals = append(lvals, lval)
	}
	if len(lhss) == len(rhss) {
		// e.g. x, y = f(), g()
		if len(lhss) == 1 {
			// x = type{...}
			// Optimization: in-place construction
			// of composite literals.
			b.exprInPlace(fn, lvals[0], rhss[0])
		} else {
			// Parallel assignment.  All reads must occur
			// before all updates, precluding exprInPlace.
			// TODO(adonovan): opt: is it sound to
			// perform exprInPlace if !isDef?
			var rvals []Value
			for _, rval := range rhss {
				rvals = append(rvals, b.expr(fn, rval))
			}
			for i, lval := range lvals {
				lval.store(fn, rvals[i])
			}
		}
	} else {
		// e.g. x, y = pos()
		tuple := b.exprN(fn, rhss[0])
		result := tuple.Type().(*types.Tuple)
		for i, lval := range lvals {
			lval.store(fn, emitExtract(fn, tuple, i, result.At(i).Type()))
		}
	}
}

// arrayLen returns the length of the array whose composite literal elements are elts.
func (b *builder) arrayLen(fn *Function, elts []ast.Expr) int64 {
	var max int64 = -1
	var i int64 = -1
	for _, e := range elts {
		if kv, ok := e.(*ast.KeyValueExpr); ok {
			i = b.expr(fn, kv.Key).(*Literal).Int64()
		} else {
			i++
		}
		if i > max {
			max = i
		}
	}
	return max + 1
}

// compLit emits to fn code to initialize a composite literal e at
// address addr with type typ, typically allocated by Alloc.
// Nested composite literals are recursively initialized in place
// where possible.
//
func (b *builder) compLit(fn *Function, addr Value, e *ast.CompositeLit, typ types.Type) {
	// TODO(adonovan): document how and why typ ever differs from
	// fn.Pkg.typeOf(e).

	switch t := typ.Underlying().(type) {
	case *types.Struct:
		for i, e := range e.Elts {
			fieldIndex := i
			if kv, ok := e.(*ast.KeyValueExpr); ok {
				fname := kv.Key.(*ast.Ident).Name
				for i, n := 0, t.NumFields(); i < n; i++ {
					sf := t.Field(i)
					if sf.Name() == fname {
						fieldIndex = i
						e = kv.Value
						break
					}
				}
			}
			sf := t.Field(fieldIndex)
			faddr := &FieldAddr{
				X:     addr,
				Field: fieldIndex,
			}
			faddr.setType(pointer(sf.Type()))
			fn.emit(faddr)
			b.exprInPlace(fn, address{addr: faddr}, e)
		}

	case *types.Array, *types.Slice:
		var at *types.Array
		var array Value
		switch t := t.(type) {
		case *types.Slice:
			at = types.NewArray(t.Elem(), b.arrayLen(fn, e.Elts))
			array = emitNew(fn, at, e.Lbrace)
		case *types.Array:
			at = t
			array = addr
		}

		var idx *Literal
		for _, e := range e.Elts {
			if kv, ok := e.(*ast.KeyValueExpr); ok {
				idx = b.expr(fn, kv.Key).(*Literal)
				e = kv.Value
			} else {
				var idxval int64
				if idx != nil {
					idxval = idx.Int64() + 1
				}
				idx = intLiteral(idxval)
			}
			iaddr := &IndexAddr{
				X:     array,
				Index: idx,
			}
			iaddr.setType(pointer(at.Elem()))
			fn.emit(iaddr)
			b.exprInPlace(fn, address{addr: iaddr}, e)
		}
		if t != at { // slice
			s := &Slice{X: array}
			s.setPos(e.Lbrace)
			s.setType(t)
			emitStore(fn, addr, fn.emit(s))
		}

	case *types.Map:
		m := &MakeMap{Reserve: intLiteral(int64(len(e.Elts)))}
		m.setPos(e.Lbrace)
		m.setType(typ)
		emitStore(fn, addr, fn.emit(m))
		for _, e := range e.Elts {
			e := e.(*ast.KeyValueExpr)
			up := &MapUpdate{
				Map:   m,
				Key:   emitConv(fn, b.expr(fn, e.Key), t.Key()),
				Value: emitConv(fn, b.expr(fn, e.Value), t.Elem()),
				pos:   e.Colon,
			}
			fn.emit(up)
		}

	case *types.Pointer:
		// Pointers can only occur in the recursive case; we
		// strip them off in addr() before calling compLit
		// again, so that we allocate space for a T not a *T.
		panic("compLit(fn, addr, e, *types.Pointer")

	default:
		panic("unexpected CompositeLit type: " + t.String())
	}
}

// switchStmt emits to fn code for the switch statement s, optionally
// labelled by label.
//
func (b *builder) switchStmt(fn *Function, s *ast.SwitchStmt, label *lblock) {
	// We treat SwitchStmt like a sequential if-else chain.
	// More efficient strategies (e.g. multiway dispatch)
	// are possible if all cases are free of side effects.
	if s.Init != nil {
		b.stmt(fn, s.Init)
	}
	var tag Value = vTrue
	if s.Tag != nil {
		tag = b.expr(fn, s.Tag)
	}
	done := fn.newBasicBlock("switch.done")
	if label != nil {
		label._break = done
	}
	// We pull the default case (if present) down to the end.
	// But each fallthrough label must point to the next
	// body block in source order, so we preallocate a
	// body block (fallthru) for the next case.
	// Unfortunately this makes for a confusing block order.
	var dfltBody *[]ast.Stmt
	var dfltFallthrough *BasicBlock
	var fallthru, dfltBlock *BasicBlock
	ncases := len(s.Body.List)
	for i, clause := range s.Body.List {
		body := fallthru
		if body == nil {
			body = fn.newBasicBlock("switch.body") // first case only
		}

		// Preallocate body block for the next case.
		fallthru = done
		if i+1 < ncases {
			fallthru = fn.newBasicBlock("switch.body")
		}

		cc := clause.(*ast.CaseClause)
		if cc.List == nil {
			// Default case.
			dfltBody = &cc.Body
			dfltFallthrough = fallthru
			dfltBlock = body
			continue
		}

		var nextCond *BasicBlock
		for _, cond := range cc.List {
			nextCond = fn.newBasicBlock("switch.next")
			// TODO(adonovan): opt: when tag==vTrue, we'd
			// get better much code if we use b.cond(cond)
			// instead of BinOp(EQL, tag, b.expr(cond))
			// followed by If.  Don't forget conversions
			// though.
			cond := emitCompare(fn, token.EQL, tag, b.expr(fn, cond), token.NoPos)
			emitIf(fn, cond, body, nextCond)
			fn.currentBlock = nextCond
		}
		fn.currentBlock = body
		fn.targets = &targets{
			tail:         fn.targets,
			_break:       done,
			_fallthrough: fallthru,
		}
		b.stmtList(fn, cc.Body)
		fn.targets = fn.targets.tail
		emitJump(fn, done)
		fn.currentBlock = nextCond
	}
	if dfltBlock != nil {
		emitJump(fn, dfltBlock)
		fn.currentBlock = dfltBlock
		fn.targets = &targets{
			tail:         fn.targets,
			_break:       done,
			_fallthrough: dfltFallthrough,
		}
		b.stmtList(fn, *dfltBody)
		fn.targets = fn.targets.tail
	}
	emitJump(fn, done)
	fn.currentBlock = done
}

// typeSwitchStmt emits to fn code for the type switch statement s, optionally
// labelled by label.
//
func (b *builder) typeSwitchStmt(fn *Function, s *ast.TypeSwitchStmt, label *lblock) {
	// We treat TypeSwitchStmt like a sequential if-else
	// chain.  More efficient strategies (e.g. multiway
	// dispatch) are possible.

	// Typeswitch lowering:
	//
	// var x X
	// switch y := x.(type) {
	// case T1, T2: S1                  // >1 	(y := x)
	// default:     SD                  // 0 types 	(y := x)
	// case T3:     S3                  // 1 type 	(y := x.(T3))
	// }
	//
	//      ...s.Init...
	// 	x := eval x
	//      y := x
	// .caseT1:
	// 	t1, ok1 := typeswitch,ok x <T1>
	// 	if ok1 then goto S1 else goto .caseT2
	// .caseT2:
	// 	t2, ok2 := typeswitch,ok x <T2>
	// 	if ok2 then goto S1 else goto .caseT3
	// .S1:
	// 	...S1...
	// 	goto done
	// .caseT3:
	// 	t3, ok3 := typeswitch,ok x <T3>
	// 	if ok3 then goto S3 else goto default
	// .S3:
	// 	y' := t3  // Kludge: within scope of S3, y resolves here
	// 	...S3...
	// 	goto done
	// .default:
	// 	goto done
	// .done:

	if s.Init != nil {
		b.stmt(fn, s.Init)
	}

	var x, y Value
	var id *ast.Ident
	switch ass := s.Assign.(type) {
	case *ast.ExprStmt: // x.(type)
		x = b.expr(fn, unparen(ass.X).(*ast.TypeAssertExpr).X)
	case *ast.AssignStmt: // y := x.(type)
		x = b.expr(fn, unparen(ass.Rhs[0]).(*ast.TypeAssertExpr).X)
		id = ass.Lhs[0].(*ast.Ident)
		y = fn.addNamedLocal(fn.Pkg.objectOf(id))
		emitStore(fn, y, x)
	}

	done := fn.newBasicBlock("typeswitch.done")
	if label != nil {
		label._break = done
	}
	var dfltBody []ast.Stmt
	for _, clause := range s.Body.List {
		cc := clause.(*ast.CaseClause)
		if cc.List == nil {
			dfltBody = cc.Body
			continue
		}
		body := fn.newBasicBlock("typeswitch.body")
		var next *BasicBlock
		var casetype types.Type
		var ti Value // t_i, ok := typeassert,ok x <T_i>
		for _, cond := range cc.List {
			next = fn.newBasicBlock("typeswitch.next")
			casetype = fn.Pkg.typeOf(cond)
			var condv Value
			if casetype == tUntypedNil {
				condv = emitCompare(fn, token.EQL, x, nilLiteral(x.Type()), token.NoPos)
			} else {
				yok := emitTypeTest(fn, x, casetype, token.NoPos)
				ti = emitExtract(fn, yok, 0, casetype)
				condv = emitExtract(fn, yok, 1, tBool)
			}
			emitIf(fn, condv, body, next)
			fn.currentBlock = next
		}
		fn.currentBlock = body
		if id != nil && len(cc.List) == 1 && casetype != tUntypedNil {
			// Declare a new shadow local variable of the
			// same name but a more specific type.
			y2 := fn.addNamedLocal(fn.Pkg.info.TypeCaseVar(cc))
			y2.name += "'" // debugging aid
			y2.typ = pointer(casetype)
			emitStore(fn, y2, ti)
		}
		fn.targets = &targets{
			tail:   fn.targets,
			_break: done,
		}
		b.stmtList(fn, cc.Body)
		fn.targets = fn.targets.tail
		if id != nil {
			fn.objects[fn.Pkg.objectOf(id)] = y // restore previous y binding
		}
		emitJump(fn, done)
		fn.currentBlock = next
	}
	fn.targets = &targets{
		tail:   fn.targets,
		_break: done,
	}
	b.stmtList(fn, dfltBody)
	fn.targets = fn.targets.tail
	emitJump(fn, done)
	fn.currentBlock = done
}

// selectStmt emits to fn code for the select statement s, optionally
// labelled by label.
//
func (b *builder) selectStmt(fn *Function, s *ast.SelectStmt, label *lblock) {
	// A blocking select of a single case degenerates to a
	// simple send or receive.
	// TODO(adonovan): opt: is this optimization worth its weight?
	if len(s.Body.List) == 1 {
		clause := s.Body.List[0].(*ast.CommClause)
		if clause.Comm != nil {
			b.stmt(fn, clause.Comm)
			done := fn.newBasicBlock("select.done")
			if label != nil {
				label._break = done
			}
			fn.targets = &targets{
				tail:   fn.targets,
				_break: done,
			}
			b.stmtList(fn, clause.Body)
			fn.targets = fn.targets.tail
			emitJump(fn, done)
			fn.currentBlock = done
			return
		}
	}

	// First evaluate all channels in all cases, and find
	// the directions of each state.
	var states []SelectState
	blocking := true
	for _, clause := range s.Body.List {
		switch comm := clause.(*ast.CommClause).Comm.(type) {
		case nil: // default case
			blocking = false

		case *ast.SendStmt: // ch<- i
			ch := b.expr(fn, comm.Chan)
			states = append(states, SelectState{
				Dir:  ast.SEND,
				Chan: ch,
				Send: emitConv(fn, b.expr(fn, comm.Value),
					ch.Type().Underlying().(*types.Chan).Elem()),
			})

		case *ast.AssignStmt: // x := <-ch
			states = append(states, SelectState{
				Dir:  ast.RECV,
				Chan: b.expr(fn, unparen(comm.Rhs[0]).(*ast.UnaryExpr).X),
			})

		case *ast.ExprStmt: // <-ch
			states = append(states, SelectState{
				Dir:  ast.RECV,
				Chan: b.expr(fn, unparen(comm.X).(*ast.UnaryExpr).X),
			})
		}
	}

	// We dispatch on the (fair) result of Select using a
	// sequential if-else chain, in effect:
	//
	// idx, recv, recvOk := select(...)
	// if idx == 0 {  // receive on channel 0
	//     x, ok := recv.(T0), recvOk
	//     ...state0...
	// } else if v == 1 {   // send on channel 1
	//     ...state1...
	// } else {
	//     ...default...
	// }
	triple := &Select{
		States:   states,
		Blocking: blocking,
	}
	triple.setPos(s.Select)
	triple.setType(tSelect)
	fn.emit(triple)
	idx := emitExtract(fn, triple, 0, tInt)

	done := fn.newBasicBlock("select.done")
	if label != nil {
		label._break = done
	}

	var dfltBody *[]ast.Stmt
	state := 0
	for _, cc := range s.Body.List {
		clause := cc.(*ast.CommClause)
		if clause.Comm == nil {
			dfltBody = &clause.Body
			continue
		}
		body := fn.newBasicBlock("select.body")
		next := fn.newBasicBlock("select.next")
		emitIf(fn, emitCompare(fn, token.EQL, idx, intLiteral(int64(state)), token.NoPos), body, next)
		fn.currentBlock = body
		fn.targets = &targets{
			tail:   fn.targets,
			_break: done,
		}
		switch comm := clause.Comm.(type) {
		case *ast.AssignStmt: // x := <-states[state].Chan
			xdecl := fn.addNamedLocal(fn.Pkg.objectOf(comm.Lhs[0].(*ast.Ident)))
			recv := emitTypeAssert(fn, emitExtract(fn, triple, 1, tEface), xdecl.Type().Deref(), token.NoPos)
			emitStore(fn, xdecl, recv)

			if len(comm.Lhs) == 2 { // x, ok := ...
				okdecl := fn.addNamedLocal(fn.Pkg.objectOf(comm.Lhs[1].(*ast.Ident)))
				emitStore(fn, okdecl, emitExtract(fn, triple, 2, okdecl.Type().Deref()))
			}
		}
		b.stmtList(fn, clause.Body)
		fn.targets = fn.targets.tail
		emitJump(fn, done)
		fn.currentBlock = next
		state++
	}
	if dfltBody != nil {
		fn.targets = &targets{
			tail:   fn.targets,
			_break: done,
		}
		b.stmtList(fn, *dfltBody)
		fn.targets = fn.targets.tail
	}
	emitJump(fn, done)
	fn.currentBlock = done
}

// forStmt emits to fn code for the for statement s, optionally
// labelled by label.
//
func (b *builder) forStmt(fn *Function, s *ast.ForStmt, label *lblock) {
	//	...init...
	//      jump loop
	// loop:
	//      if cond goto body else done
	// body:
	//      ...body...
	//      jump post
	// post:				 (target of continue)
	//      ...post...
	//      jump loop
	// done:                                 (target of break)
	if s.Init != nil {
		b.stmt(fn, s.Init)
	}
	body := fn.newBasicBlock("for.body")
	done := fn.newBasicBlock("for.done") // target of 'break'
	loop := body                         // target of back-edge
	if s.Cond != nil {
		loop = fn.newBasicBlock("for.loop")
	}
	cont := loop // target of 'continue'
	if s.Post != nil {
		cont = fn.newBasicBlock("for.post")
	}
	if label != nil {
		label._break = done
		label._continue = cont
	}
	emitJump(fn, loop)
	fn.currentBlock = loop
	if loop != body {
		b.cond(fn, s.Cond, body, done)
		fn.currentBlock = body
	}
	fn.targets = &targets{
		tail:      fn.targets,
		_break:    done,
		_continue: cont,
	}
	b.stmt(fn, s.Body)
	fn.targets = fn.targets.tail
	emitJump(fn, cont)

	if s.Post != nil {
		fn.currentBlock = cont
		b.stmt(fn, s.Post)
		emitJump(fn, loop) // back-edge
	}
	fn.currentBlock = done
}

// rangeIndexed emits to fn the header for an integer indexed loop
// over array, *array or slice value x.
// The v result is defined only if tv is non-nil.
//
func (b *builder) rangeIndexed(fn *Function, x Value, tv types.Type) (k, v Value, loop, done *BasicBlock) {
	//
	//      length = len(x)
	//      index = -1
	// loop:                                   (target of continue)
	//      index++
	// 	if index < length goto body else done
	// body:
	//      k = index
	//      v = x[index]
	//      ...body...
	// 	jump loop
	// done:                                   (target of break)

	// Determine number of iterations.
	var length Value
	if arr, ok := x.Type().Deref().(*types.Array); ok {
		// For array or *array, the number of iterations is
		// known statically thanks to the type.  We avoid a
		// data dependence upon x, permitting later dead-code
		// elimination if x is pure, static unrolling, etc.
		// Ranging over a nil *array may have >0 iterations.
		length = intLiteral(arr.Len())
	} else {
		// length = len(x).
		var c Call
		c.Call.Func = fn.Prog.Builtins[types.Universe.Lookup(nil, "len")]
		c.Call.Args = []Value{x}
		c.setType(tInt)
		length = fn.emit(&c)
	}

	index := fn.addLocal(tInt, token.NoPos)
	emitStore(fn, index, intLiteral(-1))

	loop = fn.newBasicBlock("rangeindex.loop")
	emitJump(fn, loop)
	fn.currentBlock = loop

	incr := &BinOp{
		Op: token.ADD,
		X:  emitLoad(fn, index),
		Y:  vOne,
	}
	incr.setType(tInt)
	emitStore(fn, index, fn.emit(incr))

	body := fn.newBasicBlock("rangeindex.body")
	done = fn.newBasicBlock("rangeindex.done")
	emitIf(fn, emitCompare(fn, token.LSS, incr, length, token.NoPos), body, done)
	fn.currentBlock = body

	k = emitLoad(fn, index)
	if tv != nil {
		switch t := x.Type().Underlying().(type) {
		case *types.Array:
			instr := &Index{
				X:     x,
				Index: k,
			}
			instr.setType(t.Elem())
			v = fn.emit(instr)

		case *types.Pointer: // *array
			instr := &IndexAddr{
				X:     x,
				Index: k,
			}
			instr.setType(pointer(t.Elem().(*types.Array).Elem()))
			v = emitLoad(fn, fn.emit(instr))

		case *types.Slice:
			instr := &IndexAddr{
				X:     x,
				Index: k,
			}
			instr.setType(pointer(t.Elem()))
			v = emitLoad(fn, fn.emit(instr))

		default:
			panic("rangeIndexed x:" + t.String())
		}
	}
	return
}

// rangeIter emits to fn the header for a loop using
// Range/Next/Extract to iterate over map or string value x.
// tk and tv are the types of the key/value results k and v, or nil
// if the respective component is not wanted.
//
func (b *builder) rangeIter(fn *Function, x Value, tk, tv types.Type, pos token.Pos) (k, v Value, loop, done *BasicBlock) {
	//
	//	it = range x
	// loop:                                   (target of continue)
	//	okv = next it                      (ok, key, value)
	//  	ok = extract okv #0
	// 	if ok goto body else done
	// body:
	// 	k = extract okv #1
	// 	v = extract okv #2
	//      ...body...
	// 	jump loop
	// done:                                   (target of break)
	//

	if tk == nil {
		tk = tInvalid
	}
	if tv == nil {
		tv = tInvalid
	}

	rng := &Range{X: x}
	rng.setPos(pos)
	rng.setType(tRangeIter)
	it := fn.emit(rng)

	loop = fn.newBasicBlock("rangeiter.loop")
	emitJump(fn, loop)
	fn.currentBlock = loop

	_, isString := x.Type().Underlying().(*types.Basic)

	okv := &Next{
		Iter:     it,
		IsString: isString,
	}
	okv.setType(types.NewTuple(
		varOk,
		types.NewVar(token.NoPos, nil, "k", tk),
		types.NewVar(token.NoPos, nil, "v", tv),
	))
	fn.emit(okv)

	body := fn.newBasicBlock("rangeiter.body")
	done = fn.newBasicBlock("rangeiter.done")
	emitIf(fn, emitExtract(fn, okv, 0, tBool), body, done)
	fn.currentBlock = body

	if tk != tInvalid {
		k = emitExtract(fn, okv, 1, tk)
	}
	if tv != tInvalid {
		v = emitExtract(fn, okv, 2, tv)
	}
	return
}

// rangeChan emits to fn the header for a loop that receives from
// channel x until it fails.
// tk is the channel's element type, or nil if the k result is
// not wanted
//
func (b *builder) rangeChan(fn *Function, x Value, tk types.Type) (k Value, loop, done *BasicBlock) {
	//
	// loop:                                   (target of continue)
	//      ko = <-x                           (key, ok)
	//      ok = extract ko #1
	//      if ok goto body else done
	// body:
	//      k = extract ko #0
	//      ...
	//      goto loop
	// done:                                   (target of break)

	loop = fn.newBasicBlock("rangechan.loop")
	emitJump(fn, loop)
	fn.currentBlock = loop
	recv := &UnOp{
		Op:      token.ARROW,
		X:       x,
		CommaOk: true,
	}
	recv.setType(types.NewTuple(
		types.NewVar(token.NoPos, nil, "k", tk),
		varOk,
	))
	ko := fn.emit(recv)
	body := fn.newBasicBlock("rangechan.body")
	done = fn.newBasicBlock("rangechan.done")
	emitIf(fn, emitExtract(fn, ko, 1, tBool), body, done)
	fn.currentBlock = body
	if tk != nil {
		k = emitExtract(fn, ko, 0, tk)
	}
	return
}

// rangeStmt emits to fn code for the range statement s, optionally
// labelled by label.
//
func (b *builder) rangeStmt(fn *Function, s *ast.RangeStmt, label *lblock) {
	var tk, tv types.Type
	if !isBlankIdent(s.Key) {
		tk = fn.Pkg.typeOf(s.Key)
	}
	if s.Value != nil && !isBlankIdent(s.Value) {
		tv = fn.Pkg.typeOf(s.Value)
	}

	// If iteration variables are defined (:=), this
	// occurs once outside the loop.
	//
	// Unlike a short variable declaration, a RangeStmt
	// using := never redeclares an existing variable; it
	// always creates a new one.
	if s.Tok == token.DEFINE {
		if tk != nil {
			fn.addNamedLocal(fn.Pkg.objectOf(s.Key.(*ast.Ident)))
		}
		if tv != nil {
			fn.addNamedLocal(fn.Pkg.objectOf(s.Value.(*ast.Ident)))
		}
	}

	x := b.expr(fn, s.X)

	var k, v Value
	var loop, done *BasicBlock
	switch rt := x.Type().Underlying().(type) {
	case *types.Slice, *types.Array, *types.Pointer: // *array
		k, v, loop, done = b.rangeIndexed(fn, x, tv)

	case *types.Chan:
		k, loop, done = b.rangeChan(fn, x, tk)

	case *types.Map, *types.Basic: // string
		k, v, loop, done = b.rangeIter(fn, x, tk, tv, s.For)

	default:
		panic("Cannot range over: " + rt.String())
	}

	// Evaluate both LHS expressions before we update either.
	var kl, vl lvalue
	if tk != nil {
		kl = b.addr(fn, s.Key, false) // non-escaping
	}
	if tv != nil {
		vl = b.addr(fn, s.Value, false) // non-escaping
	}
	if tk != nil {
		kl.store(fn, k)
	}
	if tv != nil {
		vl.store(fn, v)
	}

	if label != nil {
		label._break = done
		label._continue = loop
	}

	fn.targets = &targets{
		tail:      fn.targets,
		_break:    done,
		_continue: loop,
	}
	b.stmt(fn, s.Body)
	fn.targets = fn.targets.tail
	emitJump(fn, loop) // back-edge
	fn.currentBlock = done
}

// stmt lowers statement s to SSA form, emitting code to fn.
func (b *builder) stmt(fn *Function, _s ast.Stmt) {
	// The label of the current statement.  If non-nil, its _goto
	// target is always set; its _break and _continue are set only
	// within the body of switch/typeswitch/select/for/range.
	// It is effectively an additional default-nil parameter of stmt().
	var label *lblock
start:
	switch s := _s.(type) {
	case *ast.EmptyStmt:
		// ignore.  (Usually removed by gofmt.)

	case *ast.DeclStmt: // Con, Var or Typ
		d := s.Decl.(*ast.GenDecl)
		for _, spec := range d.Specs {
			if vs, ok := spec.(*ast.ValueSpec); ok {
				b.localValueSpec(fn, vs)
			}
		}

	case *ast.LabeledStmt:
		label = fn.labelledBlock(s.Label)
		emitJump(fn, label._goto)
		fn.currentBlock = label._goto
		_s = s.Stmt
		goto start // effectively: tailcall stmt(fn, s.Stmt, label)

	case *ast.ExprStmt:
		b.expr(fn, s.X)

	case *ast.SendStmt:
		fn.emit(&Send{
			Chan: b.expr(fn, s.Chan),
			X: emitConv(fn, b.expr(fn, s.Value),
				fn.Pkg.typeOf(s.Chan).Underlying().(*types.Chan).Elem()),
			pos: s.Arrow,
		})

	case *ast.IncDecStmt:
		op := token.ADD
		if s.Tok == token.DEC {
			op = token.SUB
		}
		b.assignOp(fn, b.addr(fn, s.X, false), vOne, op)

	case *ast.AssignStmt:
		switch s.Tok {
		case token.ASSIGN, token.DEFINE:
			b.assignStmt(fn, s.Lhs, s.Rhs, s.Tok == token.DEFINE)

		default: // +=, etc.
			op := s.Tok + token.ADD - token.ADD_ASSIGN
			b.assignOp(fn, b.addr(fn, s.Lhs[0], false), b.expr(fn, s.Rhs[0]), op)
		}

	case *ast.GoStmt:
		// The "intrinsics" new/make/len/cap are forbidden here.
		// panic is treated like an ordinary function call.
		var v Go
		b.setCall(fn, s.Call, &v.Call)
		fn.emit(&v)

	case *ast.DeferStmt:
		// The "intrinsics" new/make/len/cap are forbidden here.
		// panic is treated like an ordinary function call.
		var v Defer
		b.setCall(fn, s.Call, &v.Call)
		fn.emit(&v)

	case *ast.ReturnStmt:
		if fn == fn.Pkg.Init {
			// A "return" within an init block is treated
			// like a "goto" to the next init block.  We
			// use the outermost BREAK target for this purpose.
			var block *BasicBlock
			for t := fn.targets; t != nil; t = t.tail {
				if t._break != nil {
					block = t._break
				}
			}
			// Run function calls deferred in this init
			// block when explicitly returning from it.
			fn.emit(new(RunDefers))
			emitJump(fn, block)
			fn.currentBlock = fn.newBasicBlock("unreachable")
			return
		}

		var results []Value
		if len(s.Results) == 1 && fn.Signature.Results().Len() > 1 {
			// Return of one expression in a multi-valued function.
			tuple := b.exprN(fn, s.Results[0])
			ttuple := tuple.Type().(*types.Tuple)
			for i, n := 0, ttuple.Len(); i < n; i++ {
				results = append(results,
					emitConv(fn, emitExtract(fn, tuple, i, ttuple.At(i).Type()),
						fn.Signature.Results().At(i).Type()))
			}
		} else {
			// 1:1 return, or no-arg return in non-void function.
			for i, r := range s.Results {
				v := emitConv(fn, b.expr(fn, r), fn.Signature.Results().At(i).Type())
				results = append(results, v)
			}
		}
		if fn.namedResults != nil {
			// Function has named result parameters (NRPs).
			// Perform parallel assignment of return operands to NRPs.
			for i, r := range results {
				emitStore(fn, fn.namedResults[i], r)
			}
		}
		// Run function calls deferred in this
		// function when explicitly returning from it.
		fn.emit(new(RunDefers))
		if fn.namedResults != nil {
			// Reload NRPs to form the result tuple.
			results = results[:0]
			for _, r := range fn.namedResults {
				results = append(results, emitLoad(fn, r))
			}
		}
		fn.emit(&Ret{Results: results, pos: s.Return})
		fn.currentBlock = fn.newBasicBlock("unreachable")

	case *ast.BranchStmt:
		var block *BasicBlock
		switch s.Tok {
		case token.BREAK:
			if s.Label != nil {
				block = fn.labelledBlock(s.Label)._break
			} else {
				for t := fn.targets; t != nil && block == nil; t = t.tail {
					block = t._break
				}
			}

		case token.CONTINUE:
			if s.Label != nil {
				block = fn.labelledBlock(s.Label)._continue
			} else {
				for t := fn.targets; t != nil && block == nil; t = t.tail {
					block = t._continue
				}
			}

		case token.FALLTHROUGH:
			for t := fn.targets; t != nil && block == nil; t = t.tail {
				block = t._fallthrough
			}

		case token.GOTO:
			block = fn.labelledBlock(s.Label)._goto
		}
		if block == nil {
			// TODO(gri): fix: catch these in the typechecker.
			fmt.Printf("ignoring illegal branch: %s %s\n", s.Tok, s.Label)
		} else {
			emitJump(fn, block)
			fn.currentBlock = fn.newBasicBlock("unreachable")
		}

	case *ast.BlockStmt:
		b.stmtList(fn, s.List)

	case *ast.IfStmt:
		if s.Init != nil {
			b.stmt(fn, s.Init)
		}
		then := fn.newBasicBlock("if.then")
		done := fn.newBasicBlock("if.done")
		els := done
		if s.Else != nil {
			els = fn.newBasicBlock("if.else")
		}
		b.cond(fn, s.Cond, then, els)
		fn.currentBlock = then
		b.stmt(fn, s.Body)
		emitJump(fn, done)

		if s.Else != nil {
			fn.currentBlock = els
			b.stmt(fn, s.Else)
			emitJump(fn, done)
		}

		fn.currentBlock = done

	case *ast.SwitchStmt:
		b.switchStmt(fn, s, label)

	case *ast.TypeSwitchStmt:
		b.typeSwitchStmt(fn, s, label)

	case *ast.SelectStmt:
		b.selectStmt(fn, s, label)

	case *ast.ForStmt:
		b.forStmt(fn, s, label)

	case *ast.RangeStmt:
		b.rangeStmt(fn, s, label)

	default:
		panic(fmt.Sprintf("unexpected statement kind: %T", s))
	}
}

// buildFunction builds SSA code for the body of function fn.  Idempotent.
func (b *builder) buildFunction(fn *Function) {
	if fn.Blocks != nil {
		return // building already started
	}
	if fn.syntax == nil {
		return // not a Go source function.  (Synthetic, or from object file.)
	}
	if fn.syntax.body == nil {
		// External function.
		if fn.Params == nil {
			// This condition ensures we add a non-empty
			// params list once only, but we may attempt
			// the degenerate empty case repeatedly.
			// TODO(adonovan): opt: don't do that.

			// We set Function.Params even though there is no body
			// code to reference them.  This simplifies clients.
			if recv := fn.Signature.Recv(); recv != nil {
				fn.addParamObj(recv)
			}
			params := fn.Signature.Params()
			for i, n := 0, params.Len(); i < n; i++ {
				fn.addParamObj(params.At(i))
			}
		}
		return
	}
	if fn.Prog.mode&LogSource != 0 {
		defer logStack("build function %s @ %s",
			fn.FullName(), fn.Prog.Files.Position(fn.pos))()
	}
	fn.startBody()
	fn.createSyntacticParams()
	b.stmt(fn, fn.syntax.body)
	if cb := fn.currentBlock; cb != nil && (cb == fn.Blocks[0] || cb.Preds != nil) {
		// Run function calls deferred in this function when
		// falling off the end of the body block.
		fn.emit(new(RunDefers))
		fn.emit(new(Ret))
	}
	fn.finishBody()
}

// buildDecl builds SSA code for all globals, functions or methods
// declared by decl in package pkg.
//
func (b *builder) buildDecl(pkg *Package, decl ast.Decl) {
	switch decl := decl.(type) {
	case *ast.GenDecl:
		switch decl.Tok {
		// Nothing to do for CONST, IMPORT.
		case token.VAR:
			for _, spec := range decl.Specs {
				b.globalValueSpec(pkg.Init, spec.(*ast.ValueSpec), nil, nil)
			}
		case token.TYPE:
			for _, spec := range decl.Specs {
				id := spec.(*ast.TypeSpec).Name
				if isBlankIdent(id) {
					continue
				}
				nt := pkg.objectOf(id).Type().(*types.Named)
				for i, n := 0, nt.NumMethods(); i < n; i++ {
					b.buildFunction(pkg.Prog.concreteMethods[nt.Method(i)])
				}
			}
		}

	case *ast.FuncDecl:
		id := decl.Name
		if decl.Recv != nil {
			return // method declaration
		}
		if isBlankIdent(id) {
			// no-op

		} else if id.Name == "init" {
			// init() block
			if pkg.Prog.mode&LogSource != 0 {
				fmt.Fprintln(os.Stderr, "build init block @", pkg.Prog.Files.Position(decl.Pos()))
			}
			init := pkg.Init

			// A return statement within an init block is
			// treated like a "goto" to the the next init
			// block, which we stuff in the outermost
			// break label.
			next := init.newBasicBlock("init.next")
			init.targets = &targets{
				tail:   init.targets,
				_break: next,
			}
			b.stmt(init, decl.Body)
			// Run function calls deferred in this init
			// block when falling off the end of the block.
			init.emit(new(RunDefers))
			emitJump(init, next)
			init.targets = init.targets.tail
			init.currentBlock = next

		} else {
			// Package-level function.
			b.buildFunction(pkg.Prog.Value(pkg.objectOf(id)).(*Function))
		}
	}

}

// BuildAll calls Package.Build() for each package in prog.
// Building occurs in parallel unless the BuildSerially mode flag was set.
//
// BuildAll is idempotent and thread-safe.
//
func (prog *Program) BuildAll() {
	var wg sync.WaitGroup
	for _, p := range prog.Packages {
		if prog.mode&BuildSerially != 0 {
			p.Build()
		} else {
			wg.Add(1)
			go func(p *Package) {
				p.Build()
				wg.Done()
			}(p)
		}
	}
	wg.Wait()
}

// Build builds SSA code for all functions and vars in package p.
//
// Build is idempotent and thread-safe.
//
func (p *Package) Build() {
	if !atomic.CompareAndSwapInt32(&p.started, 0, 1) {
		return // already started
	}
	if p.info.Files == nil {
		p.info = nil
		return // nothing to do
	}
	if p.Prog.mode&LogSource != 0 {
		defer logStack("build %s", p)()
	}
	init := p.Init
	init.startBody()

	// Make init() skip if package is already initialized.
	initguard := p.Var("init$guard")
	doinit := init.newBasicBlock("init.start")
	done := init.newBasicBlock("init.done")
	emitIf(init, emitLoad(init, initguard), done, doinit)
	init.currentBlock = doinit
	emitStore(init, initguard, vTrue)

	// Call the init() function of each package we import.
	for _, typkg := range p.info.Imports() {
		var v Call
		v.Call.Func = p.Prog.packages[typkg].Init
		v.Call.pos = init.pos
		v.setType(types.NewTuple())
		init.emit(&v)
	}

	b := &builder{
		nTo1Vars: make(map[*ast.ValueSpec]bool),
	}

	// Visit the package's var decls and init funcs in source
	// order.  This causes init() code to be generated in
	// topological order.  We visit them transitively through
	// functions of the same package, but we don't treat functions
	// as roots.
	//
	// We also ensure all functions and methods are built, even if
	// they are unreachable.
	for _, file := range p.info.Files {
		for _, decl := range file.Decls {
			b.buildDecl(p, decl)
		}
	}

	p.info = nil // We no longer need ASTs or go/types deductions.

	// Finish up.
	emitJump(init, done)
	init.currentBlock = done
	init.emit(new(RunDefers))
	init.emit(new(Ret))
	init.finishBody()
}

// Only valid during p's create and build phases.
func (p *Package) objectOf(id *ast.Ident) types.Object {
	if o := p.info.ObjectOf(id); o != nil {
		return o
	}
	panic(fmt.Sprintf("no types.Object for ast.Ident %s @ %s",
		id.Name, p.Prog.Files.Position(id.Pos())))
}

// Only valid during p's create and build phases.
func (p *Package) typeOf(e ast.Expr) types.Type {
	return p.info.TypeOf(e)
}
