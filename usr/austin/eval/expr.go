// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"eval";
	"go/ast";
	"go/scanner";
	"go/token";
	"log";
	"os";
	"strconv";
	"strings";
)

// An exprCompiler compiles a single node in an expression.  It stores
// the whole expression's context plus information specific to this node.
// After compilation, it stores the type of the expression and its
// evaluator function.
type exprCompiler struct {
	*exprContext;
	pos token.Position;
	t Type;
	// Evaluate this node as the given type.
	evalBool func(f *Frame) bool;
	evalUint func(f *Frame) uint64;
	evalInt func(f *Frame) int64;
	// TODO(austin) evalIdealInt and evalIdealFloat shouldn't be
	// functions at all.
	evalIdealInt func() *bignum.Integer;
	evalFloat func(f *Frame) float64;
	evalIdealFloat func() *bignum.Rational;
	evalString func(f *Frame) string;
	evalArray func(f *Frame) ArrayValue;
	evalPtr func(f *Frame) Value;
	evalFunc func(f *Frame) Func;
	// Evaluate to the "address of" this value; that is, the
	// settable Value object.  nil for expressions whose address
	// cannot be taken.
	evalAddr func(f *Frame) Value;
	// Execute this expression as a statement.  Only expressions
	// that are valid expression statements should set this.
	exec func(f *Frame);
	// A short string describing this expression for error
	// messages.  Only necessary if t != nil.
	desc string;
}

func newExprCompiler(c *exprContext, pos token.Position) *exprCompiler {
	return &exprCompiler{
		exprContext: c,
		pos: pos,
		desc: "<missing description>"
	};
}

// Operator generators
// TODO(austin) Remove these forward declarations
func (a *exprCompiler) genConstant(v Value)
func (a *exprCompiler) genIdentOp(s *Scope, index int)
func (a *exprCompiler) genIndexArray(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genFuncCall(call func(f *Frame) []Value)
func (a *exprCompiler) genStarOp(v *exprCompiler)
func (a *exprCompiler) genUnaryOpNeg(v *exprCompiler)
func (a *exprCompiler) genUnaryOpNot(v *exprCompiler)
func (a *exprCompiler) genUnaryOpXor(v *exprCompiler)
func (a *exprCompiler) genBinOpAdd(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpSub(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpMul(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpQuo(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpRem(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpAnd(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpOr(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpXor(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpAndNot(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpShl(l *exprCompiler, r *exprCompiler)
func (a *exprCompiler) genBinOpShr(l *exprCompiler, r *exprCompiler)
func genAssign(lt Type, r *exprCompiler) (func(lv Value, f *Frame))

func (a *exprCompiler) copy() *exprCompiler {
	ec := newExprCompiler(a.exprContext, a.pos);
	ec.desc = a.desc;
	return ec;
}

func (a *exprCompiler) copyVisit(x ast.Expr) *exprCompiler {
	ec := newExprCompiler(a.exprContext, x.Pos());
	x.Visit(ec);
	return ec;
}

func (a *exprCompiler) diag(format string, args ...) {
	a.diagAt(&a.pos, format, args);
}

func (a *exprCompiler) diagOpType(op token.Token, vt Type) {
	a.diag("illegal operand type for '%v' operator\n\t%v", op, vt);
}

func (a *exprCompiler) diagOpTypes(op token.Token, lt Type, rt Type) {
	a.diag("illegal operand types for '%v' operator\n\t%v\n\t%v", op, lt, rt);
}

/*
 * "As" functions.  These retrieve evaluator functions from an
 * exprCompiler, panicking if the requested evaluator is nil.
 */

func (a *exprCompiler) asBool() (func(f *Frame) bool) {
	if a.evalBool == nil {
		log.Crashf("tried to get %v node as boolType", a.t);
	}
	return a.evalBool;
}

func (a *exprCompiler) asUint() (func(f *Frame) uint64) {
	if a.evalUint == nil {
		log.Crashf("tried to get %v node as uintType", a.t);
	}
	return a.evalUint;
}

func (a *exprCompiler) asInt() (func(f *Frame) int64) {
	if a.evalInt == nil {
		log.Crashf("tried to get %v node as intType", a.t);
	}
	return a.evalInt;
}

func (a *exprCompiler) asIdealInt() (func() *bignum.Integer) {
	if a.evalIdealInt == nil {
		log.Crashf("tried to get %v node as idealIntType", a.t);
	}
	return a.evalIdealInt;
}

func (a *exprCompiler) asFloat() (func(f *Frame) float64) {
	if a.evalFloat == nil {
		log.Crashf("tried to get %v node as floatType", a.t);
	}
	return a.evalFloat;
}

func (a *exprCompiler) asIdealFloat() (func() *bignum.Rational) {
	if a.evalIdealFloat == nil {
		log.Crashf("tried to get %v node as idealFloatType", a.t);
	}
	return a.evalIdealFloat;
}

func (a *exprCompiler) asString() (func(f *Frame) string) {
	if a.evalString == nil {
		log.Crashf("tried to get %v node as stringType", a.t);
	}
	return a.evalString;
}

func (a *exprCompiler) asArray() (func(f *Frame) ArrayValue) {
	if a.evalArray == nil {
		log.Crashf("tried to get %v node as ArrayType", a.t);
	}
	return a.evalArray;
}

func (a *exprCompiler) asPtr() (func(f *Frame) Value) {
	if a.evalPtr == nil {
		log.Crashf("tried to get %v node as PtrType", a.t);
	}
	return a.evalPtr;
}

func (a *exprCompiler) asFunc() (func(f *Frame) Func) {
	if a.evalFunc == nil {
		log.Crashf("tried to get %v node as FuncType", a.t);
	}
	return a.evalFunc;
}

/*
 * Common expression manipulations
 */

// a.convertTo(t) converts the value of the analyzed expression a,
// which must be a constant, ideal number, to a new analyzed
// expression with a constant value of type t.
func (a *exprCompiler) convertTo(t Type) *exprCompiler {
	if !a.t.isIdeal() {
		log.Crashf("attempted to convert from %v, expected ideal", a.t);
	}

	var rat *bignum.Rational;

	// XXX(Spec)  The spec says "It is erroneous".
	//
	// It is an error to assign a value with a non-zero fractional
	// part to an integer, or if the assignment would overflow or
	// underflow, or in general if the value cannot be represented
	// by the type of the variable.
	switch a.t {
	case IdealFloatType:
		rat = a.asIdealFloat()();
		if t.isInteger() && !rat.IsInt() {
			a.diag("constant %v truncated to integer", ratToString(rat));
			return nil;
		}
	case IdealIntType:
		i := a.asIdealInt()();
		rat = bignum.MakeRat(i, bignum.Nat(1));
	default:
		log.Crashf("unexpected ideal type %v", a.t);
	}

	// Check bounds
	if t, ok := t.rep().(BoundedType); ok {
		if rat.Cmp(t.minVal()) < 0 {
			a.diag("constant %v underflows %v", ratToString(rat), t);
			return nil;
		}
		if rat.Cmp(t.maxVal()) > 0 {
			a.diag("constant %v overflows %v", ratToString(rat), t);
			return nil;
		}
	}

	// Convert rat to type t.
	res := a.copy();
	res.t = t;
	switch t := t.rep().(type) {
	case *uintType:
		n, d := rat.Value();
		f := n.Quo(bignum.MakeInt(false, d));
		v := f.Abs().Value();
		res.evalUint = func(*Frame) uint64 { return v };
	case *intType:
		n, d := rat.Value();
		f := n.Quo(bignum.MakeInt(false, d));
		v := f.Value();
		res.evalInt = func(*Frame) int64 { return v };
	case *idealIntType:
		n, d := rat.Value();
		f := n.Quo(bignum.MakeInt(false, d));
		res.evalIdealInt = func() *bignum.Integer { return f };
	case *floatType:
		n, d := rat.Value();
		v := float64(n.Value())/float64(d.Value());
		res.evalFloat = func(*Frame) float64 { return v };
	case *idealFloatType:
		res.evalIdealFloat = func() *bignum.Rational { return rat };
	default:
		log.Crashf("cannot convert to type %T", t);
	}

	return res;
}

// mkAssign takes an optional expected l-value type, lt, and an
// r-value expression compiler, r, and returns the expected l-value
// type and a function that evaluates the r-value and assigns it to
// the l-value lv.
//
// If lt is non-nil, the returned l-value type will always be lt.  If
// lt is nil, mkAssign will infer and return the appropriate l-value
// type, or produce an error.
//
// errOp specifies the operation name to use for error messages, such
// as "assignment", or "function call".  errPosName specifies the name
// to use for positions.  errPos, if non-zero, specifies the position
// of this assignment (for tuple assignments or function arguments).
//
// If the assignment fails to typecheck, this generates an error
// message and returns nil, nil.
func mkAssign(lt Type, r *exprCompiler, errOp string, errPosName string, errPos int) (Type, func(lv Value, f *Frame)) {
	// However, when [an ideal is] (used in an expression)
	// assigned to a variable or typed constant, the destination
	// must be able to represent the assigned value.
	if r.t.isIdeal() && (lt == nil || lt.isInteger() || lt.isFloat()) {
		// If the type is absent and the corresponding
		// expression is a constant expression of ideal
		// integer or ideal float type, the type of the
		// declared variable is int or float respectively.
		if lt == nil {
			switch {
			case r.t.isInteger():
				lt = IntType;
			case r.t.isFloat():
				lt = FloatType;
			default:
				log.Crashf("unexpected ideal type %v", r.t);
			}
		}
		r = r.convertTo(lt);
		if r == nil {
			return nil, nil;
		}
	}

	// TOOD(austin) Deal with assignment special cases

	if lt == nil {
		lt = r.t;
	} else {
		// Values of any type may always be assigned to
		// variables of compatible static type.
		if lt.literal() != r.t.literal() {
			if errPos == 0 {
				r.diag("illegal operand types for %s\n\t%v\n\t%v", errOp, lt, r.t);
			} else {
				r.diag("illegal operand types in %s %d of %s\n\t%v\n\t%v", errPosName, errPos, errOp, lt, r.t);
			}
			return nil, nil;
		}
	}

	// Compile
	return lt, genAssign(lt, r);
}

/*
 * Expression visitors
 */

func (a *exprCompiler) DoBadExpr(x *ast.BadExpr) {
	// Do nothing.  Already reported by parser.
}

func (a *exprCompiler) DoIdent(x *ast.Ident) {
	def, dscope := a.scope.Lookup(x.Value);
	if def == nil {
		a.diag("%s: undefined", x.Value);
		return;
	}
	switch def := def.(type) {
	case *Constant:
		a.t = def.Type;
		a.genConstant(def.Value);
		a.desc = "constant";
	case *Variable:
		if a.constant {
			a.diag("variable %s used in constant expression", x.Value);
			return;
		}
		a.t = def.Type;
		defidx := def.Index;
		a.genIdentOp(dscope, defidx);
		a.desc = "variable";
	case Type:
		a.diag("type %v used as expression", x.Value);
	default:
		log.Crashf("name %s has unknown type %T", x.Value, def);
	}
}

func (a *exprCompiler) doIdealInt(i *bignum.Integer) {
	a.t = IdealIntType;
	a.evalIdealInt = func() *bignum.Integer { return i };
}

func (a *exprCompiler) DoIntLit(x *ast.IntLit) {
	i, _, _2 := bignum.IntFromString(string(x.Value), 0);
	a.doIdealInt(i);
	a.desc = "integer literal";
}

func (a *exprCompiler) DoCharLit(x *ast.CharLit) {
	if x.Value[0] != '\'' {
		log.Crashf("malformed character literal %s at %v passed parser", x.Value, x.Pos());
	}
	v, mb, tail, err := strconv.UnquoteChar(string(x.Value[1:len(x.Value)]), '\'');
	if err != nil || tail != "'" {
		log.Crashf("malformed character literal %s at %v passed parser", x.Value, x.Pos());
	}
	a.doIdealInt(bignum.Int(int64(v)));
	a.desc = "character literal";
}

func (a *exprCompiler) DoFloatLit(x *ast.FloatLit) {
	f, _, n := bignum.RatFromString(string(x.Value), 0);
	if n != len(x.Value) {
		log.Crashf("malformed float literal %s at %v passed parser", x.Value, x.Pos());
	}
	a.t = IdealFloatType;
	a.evalIdealFloat = func() *bignum.Rational { return f };
	a.desc = "float literal";
}

func (a *exprCompiler) doString(s string) {
	a.t = StringType;
	a.evalString = func(*Frame) string { return s };
}

func (a *exprCompiler) DoStringLit(x *ast.StringLit) {
	s, err := strconv.Unquote(string(x.Value));
	if err != nil {
		a.diag("illegal string literal, %v", err);
		return;
	}
	a.doString(s);
	a.desc = "string literal";
}

func (a *exprCompiler) DoStringList(x *ast.StringList) {
	ss := make([]string, len(x.Strings));
	for i := 0; i < len(x.Strings); i++ {
		s, err := strconv.Unquote(string(x.Strings[i].Value));
		if err != nil {
			a.diag("illegal string literal, %v", err);
			return;
		}
		ss[i] = s;
	}
	a.doString(strings.Join(ss, ""));
	a.desc = "string literal";
}

func (a *exprCompiler) DoFuncLit(x *ast.FuncLit) {
	// TODO(austin) Closures capture their entire defining frame
	// instead of just the variables they use.

	decl := a.compileFuncType(a.scope, x.Type);
	if decl == nil {
		// TODO(austin) Try compiling the body, perhaps with
		// dummy definitions for the arguments
		return;
	}

	evalFunc := a.compileFunc(a.scope, decl, x.Body);
	if evalFunc == nil {
		return;
	}

	if a.constant {
		a.diag("function literal used in constant expression");
		return;
	}

	a.t = decl.Type;
	a.evalFunc = evalFunc;
}

func (a *exprCompiler) DoCompositeLit(x *ast.CompositeLit) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoParenExpr(x *ast.ParenExpr) {
	x.X.Visit(a);
}

func (a *exprCompiler) DoSelectorExpr(x *ast.SelectorExpr) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoIndexExpr(x *ast.IndexExpr) {
	l, r := a.copyVisit(x.X), a.copyVisit(x.Index);
	if l.t == nil || r.t == nil {
		return;
	}

	// Type check object
	if lt, ok := l.t.rep().(*PtrType); ok {
		if et, ok := lt.Elem.rep().(*ArrayType); ok {
			// Automatic dereference
			nl := l.copy();
			nl.t = et;
			nl.genStarOp(l);
			l = nl;
		}
	}

	var at Type;
	intIndex := false;
	var maxIndex int64 = -1;

	switch lt := l.t.rep().(type) {
	case *ArrayType:
		at = lt.Elem;
		intIndex = true;
		maxIndex = lt.Len;

	// TODO(austin) Uncomment when there is a SliceType
	// case *SliceType:
	// 	a.t = lt.Elem;
	// 	intIndex = true;

	case *stringType:
		at = Uint8Type;
		intIndex = true;

	// TODO(austin) Uncomment when there is a MapType
	// case *MapType:
	// 	log.Crash("Index into map not implemented");

	default:
		a.diag("cannot index into %v", l.t);
		return;
	}

	// Type check index and convert to int if necessary
	if intIndex {
		// XXX(Spec) It's unclear if ideal floats with no
		// fractional part are allowed here.  6g allows it.  I
		// believe that's wrong.
		switch _ := r.t.rep().(type) {
		case *idealIntType:
			val := r.asIdealInt()();
			if val.IsNeg() || (maxIndex != -1 && val.Cmp(bignum.Int(maxIndex)) >= 0) {
				a.diag("array index out of bounds");
				return;
			}
			r = r.convertTo(IntType);
			if r == nil {
				return;
			}

		case *uintType:
			// Convert to int
			nr := r.copy();
			nr.t = IntType;
			rf := r.asUint();
			nr.evalInt = func(f *Frame) int64 {
				return int64(rf(f));
			};
			r = nr;

		case *intType:
			// Good as is

		default:
			a.diag("illegal operand type for index\n\t%v", r.t);
			return;
		}
	}

	a.t = at;

	// Compile
	switch lt := l.t.rep().(type) {
	case *ArrayType:
		a.t = lt.Elem;
		// TODO(austin) Bounds check
		a.genIndexArray(l, r);
		lf := l.asArray();
		rf := r.asInt();
		a.evalAddr = func(f *Frame) Value {
			return lf(f).Elem(rf(f));
		};

	case *stringType:
		// TODO(austin) Bounds check
		lf := l.asString();
		rf := r.asInt();
		// TODO(austin) This pulls over the whole string in a
		// remote setting, instead of just the one character.
		a.evalUint = func(f *Frame) uint64 {
			return uint64(lf(f)[rf(f)]);
		}

	default:
		log.Crashf("Compilation of index into %T not implemented", l.t);
	}
}

func (a *exprCompiler) DoTypeAssertExpr(x *ast.TypeAssertExpr) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoCallExpr(x *ast.CallExpr) {
	// TODO(austin) Type conversions look like calls, but will
	// fail in DoIdent right now.
	//
	// TODO(austin) Magic built-in functions
	//
	// TODO(austin) Variadic functions.

	// Compile children
	bad := false;
	l := a.copyVisit(x.Fun);
	if l.t == nil {
		bad = true;
	}
	as := make([]*exprCompiler, len(x.Args));
	ats := make([]Type, len(as));
	for i := 0; i < len(x.Args); i++ {
		as[i] = a.copyVisit(x.Args[i]);
		if as[i].t == nil {
			bad = true;
		}
		ats[i] = as[i].t;
	}
	if bad {
		return;
	}

	// Type check
	if a.constant {
		a.diag("function call in constant context");
		return;
	}

	// XXX(Spec) Calling a named function type is okay.  I really
	// think there needs to be a general discussion of named
	// types.  A named type creates a new, distinct type, but the
	// type of that type is still whatever it's defined to.  Thus,
	// in "type Foo int", Foo is still an integer type and in
	// "type Foo func()", Foo is a function type.
	lt, ok := l.t.rep().(*FuncType);
	if !ok {
		a.diag("cannot call non-function type %v", l.t);
		return;
	}

	if len(as) != len(lt.In) {
		msg := "too many";
		if len(as) < len(lt.In) {
			msg = "not enough";
		}
		a.diag("%s arguments to call\n\t%s\n\t%s", msg, typeListString(lt.In, nil), typeListString(ats, nil));
		return;
	}

	// The arguments must be single-valued expressions assignment
	// compatible with the parameters of F.
	afs := make([]func(lv Value, f *Frame), len(as));
	for i := 0; i < len(as); i++ {
		var at Type;
		at, afs[i] = mkAssign(lt.In[i], as[i], "function call", "argument", i + 1);
		if at == nil {
			bad = true;
		}
	}
	if bad {
		return;
	}

	nResults := len(lt.Out);
	if nResults != 1 {
		log.Crashf("Multi-valued return type not implemented");
	}
	a.t = lt.Out[0];

	// Compile
	lf := l.asFunc();
	call := func(f *Frame) []Value {
		fun := lf(f);
		fr := fun.NewFrame();
		for i, af := range afs {
			af(fr.Vars[i], f);
		}
		fun.Call(fr);
		return fr.Vars[len(afs):len(afs)+nResults];
	};
	a.genFuncCall(call);

	// Function calls, method calls, and channel operations can
	// appear in statement context.
	a.exec = func(f *Frame) { call(f) };
}

func (a *exprCompiler) DoStarExpr(x *ast.StarExpr) {
	v := a.copyVisit(x.X);
	if v.t == nil {
		return;
	}

	switch vt := v.t.rep().(type) {
	case *PtrType:
		a.t = vt.Elem;
		a.genStarOp(v);
		a.desc = "indirect expression";

	default:
		a.diagOpType(token.MUL, v.t);
	}
}

func (a *exprCompiler) genUnaryAddrOf(v *exprCompiler) {
	vf := v.evalAddr;
	a.evalPtr = func(f *Frame) Value { return vf(f) };
}

var unaryOpDescs = make(map[token.Token] string)

func (a *exprCompiler) DoUnaryExpr(x *ast.UnaryExpr) {
	v := a.copyVisit(x.X);
	if v.t == nil {
		return;
	}

	// Type check
	switch x.Op {
	case token.ADD, token.SUB:
		if !v.t.isInteger() && !v.t.isFloat() {
			a.diagOpType(x.Op, v.t);
			return;
		}
		a.t = v.t;

	case token.NOT:
		if !v.t.isBoolean() {
			a.diagOpType(x.Op, v.t);
			return;
		}
		a.t = BoolType;

	case token.XOR:
		if !v.t.isInteger() {
			a.diagOpType(x.Op, v.t);
			return;
		}
		a.t = v.t;

	case token.AND:
		// The unary prefix address-of operator & generates
		// the address of its operand, which must be a
		// variable, pointer indirection, field selector, or
		// array or slice indexing operation.
		if v.evalAddr == nil {
			a.diag("cannot take the address of %s", v.desc);
			return;
		}

		// TODO(austin) Implement "It is illegal to take the
		// address of a function result variable" once I have
		// function result variables.

		a.t = NewPtrType(v.t);

	case token.ARROW:
		log.Crashf("Unary op %v not implemented", x.Op);

	default:
		log.Crashf("unknown unary operator %v", x.Op);
	}

	var ok bool;
	a.desc, ok = unaryOpDescs[x.Op];
 	if !ok {
		a.desc = "unary " + x.Op.String() + " expression";
		unaryOpDescs[x.Op] = a.desc;
	}

	// Compile
	switch x.Op {
	case token.ADD:
		// Just compile it out
		*a = *v;

	case token.SUB:
		a.genUnaryOpNeg(v);

	case token.NOT:
		a.genUnaryOpNot(v);

	case token.XOR:
		a.genUnaryOpXor(v);

	case token.AND:
		a.genUnaryAddrOf(v);

	default:
		log.Crashf("Compilation of unary op %v not implemented", x.Op);
	}
}

var binOpDescs = make(map[token.Token] string)

func (a *exprCompiler) doBinaryExpr(op token.Token, l, r *exprCompiler) {
	// Save the original types of l.t and r.t for error messages.
	origlt := l.t;
	origrt := r.t;

	// XXX(Spec) What is the exact definition of a "named type"?

	// XXX(Spec) Arithmetic operators: "Integer types" apparently
	// means all types compatible with basic integer types, though
	// this is never explained.  Likewise for float types, etc.
	// This relates to the missing explanation of named types.

	// XXX(Spec) Operators: "If both operands are ideal numbers,
	// the conversion is to ideal floats if one of the operands is
	// an ideal float (relevant for / and %)."  How is that
	// relevant only for / and %?  If I add an ideal int and an
	// ideal float, I get an ideal float.

	if op != token.SHL && op != token.SHR {
		// Except in shift expressions, if one operand has
		// numeric type and the other operand is an ideal
		// number, the ideal number is converted to match the
		// type of the other operand.
		if (l.t.isInteger() || l.t.isFloat()) && !l.t.isIdeal() && r.t.isIdeal() {
			r = r.convertTo(l.t);
		} else if (r.t.isInteger() || r.t.isFloat()) && !r.t.isIdeal() && l.t.isIdeal() {
			l = l.convertTo(r.t);
		}
		if l == nil || r == nil {
			return;
		}

		// Except in shift expressions, if both operands are
		// ideal numbers and one is an ideal float, the other
		// is converted to ideal float.
		if l.t.isIdeal() && r.t.isIdeal() {
			if l.t.isInteger() && r.t.isFloat() {
				l = l.convertTo(r.t);
			} else if l.t.isFloat() && r.t.isInteger() {
				r = r.convertTo(l.t);
			}
			if l == nil || r == nil {
				return;
			}
		}
	}

	// XXX(Spec) "The operand types in binary operations must be
	// compatible" should say the types must be *identical*.

	// Useful type predicates
	same := func() bool {
		return l.t == r.t;
	};
	integers := func() bool {
		return l.t.isInteger() && r.t.isInteger();
	};
	floats := func() bool {
		return l.t.isFloat() && r.t.isFloat();
	};
	strings := func() bool {
		// TODO(austin) Deal with named types
		return l.t == StringType && r.t == StringType;
	};
	booleans := func() bool {
		return l.t.isBoolean() && r.t.isBoolean();
	};

	// Type check
	switch op {
	case token.ADD:
		if !same() || (!integers() && !floats() && !strings()) {
			a.diagOpTypes(op, origlt, origrt);
			return;
		}
		a.t = l.t;

	case token.SUB, token.MUL, token.QUO:
		if !same() || (!integers() && !floats()) {
			a.diagOpTypes(op, origlt, origrt);
			return;
		}
		a.t = l.t;

	case token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
		if !same() || !integers() {
			a.diagOpTypes(op, origlt, origrt);
			return;
		}
		a.t = l.t;

	case token.SHL, token.SHR:
		// XXX(Spec) Is it okay for the right operand to be an
		// ideal float with no fractional part?  "The right
		// operand in a shift operation must be always be of
		// unsigned integer type or an ideal number that can
		// be safely converted into an unsigned integer type
		// (§Arithmetic operators)" suggests so and 6g agrees.

		if !l.t.isInteger() || !(r.t.isInteger() || r.t.isIdeal()) {
			a.diagOpTypes(op, origlt, origrt);
			return;
		}

		// The right operand in a shift operation must be
		// always be of unsigned integer type or an ideal
		// number that can be safely converted into an
		// unsigned integer type.
		if r.t.isIdeal() {
			r2 := r.convertTo(UintType);
			if r2 == nil {
				return;
			}

			// If the left operand is not ideal, convert
			// the right to not ideal.
			if !l.t.isIdeal() {
				r = r2;
			}

			// If both are ideal, but the right side isn't
			// an ideal int, convert it to simplify things.
			if l.t.isIdeal() && !r.t.isInteger() {
				r = r.convertTo(IdealIntType);
				if r == nil {
					log.Crashf("conversion to uintType succeeded, but conversion to idealIntType failed");
				}
			}
		} else if _, ok := r.t.rep().(*uintType); !ok {
			a.diag("right operand of shift must be unsigned");
			return;
		}

		if l.t.isIdeal() && !r.t.isIdeal() {
			// XXX(Spec) What is the meaning of "ideal >>
			// non-ideal"?  Russ says the ideal should be
			// converted to an int.  6g propagates the
			// type down from assignments as a hint.

			l = l.convertTo(IntType);
			if l == nil {
				return;
			}
		}

		// At this point, we should have one of three cases:
		// 1) uint SHIFT uint
		// 2) int SHIFT uint
		// 3) ideal int SHIFT ideal int

		a.t = l.t;

	case token.LOR, token.LAND:
		if !booleans() {
			return;
		}
		// XXX(Spec) There's no mention of *which* boolean
		// type the logical operators return.  From poking at
		// 6g, it appears to be the named boolean type, NOT
		// the type of the left operand, and NOT an unnamed
		// boolean type.

		a.t = BoolType;

	case token.ARROW:
		// The operands in channel sends differ in type: one
		// is always a channel and the other is a variable or
		// value of the channel's element type.
		log.Crash("Binary op <- not implemented");
		a.t = BoolType;

	case token.LSS, token.GTR, token.LEQ, token.GEQ:
		// ... booleans may be compared only for equality or
		// inequality.

		if l.t.isBoolean() || r.t.isBoolean() {
			a.diagOpTypes(op, origlt, origrt);
			return;
		}

		fallthrough;
	case token.EQL, token.NEQ:
		// When comparing two operands of channel type, the
		// channel value types must be compatible but the
		// channel direction is ignored.

		// XXX(Spec) Operators: "When comparing two operands
		// of channel type, the channel value types must be
		// compatible but the channel direction is ignored."
		// By "compatible" this really means "comparison
		// compatible".  Really, the rules for type checking
		// comparison operators are entirely different from
		// other binary operators, but this just barely hints
		// at that.

		// XXX(Spec) Comparison operators: "All comparison
		// operators apply to basic types except bools."
		// "except bools" is really weird here, since this is
		// actually explained in the Comparison compatibility
		// section.
		log.Crashf("Binary op %v not implemented", op);
		a.t = BoolType;

	default:
		log.Crashf("unknown binary operator %v", op);
	}

	var ok bool;
	a.desc, ok = binOpDescs[op];
	if !ok {
		a.desc = op.String() + " expression";
		binOpDescs[op] = a.desc;
	}

	// Compile
	switch op {
	case token.ADD:
		a.genBinOpAdd(l, r);

	case token.SUB:
		a.genBinOpSub(l, r);

	case token.MUL:
		a.genBinOpMul(l, r);

	case token.QUO:
		// TODO(austin) What if divisor is zero?
		// TODO(austin) Clear higher bits that may have
		// accumulated in our temporary.
		a.genBinOpQuo(l, r);

	case token.REM:
		// TODO(austin) What if divisor is zero?
		// TODO(austin) Clear higher bits that may have
		// accumulated in our temporary.
		a.genBinOpRem(l, r);

	case token.AND:
		a.genBinOpAnd(l, r);

	case token.OR:
		a.genBinOpOr(l, r);

	case token.XOR:
		a.genBinOpXor(l, r);

	case token.AND_NOT:
		a.genBinOpAndNot(l, r);

	case token.SHL:
		if l.t.isIdeal() {
			lv := l.asIdealInt()();
			rv := r.asIdealInt()();
			const maxShift = 99999;
			if rv.Cmp(bignum.Int(maxShift)) > 0 {
				a.diag("left shift by %v; exceeds implementation limit of %v", rv, maxShift);
				a.t = nil;
				return;
			}
			val := lv.Shl(uint(rv.Value()));
			a.evalIdealInt = func() *bignum.Integer { return val };
		} else {
			a.genBinOpShl(l, r);
		}

	case token.SHR:
		if l.t.isIdeal() {
			lv := l.asIdealInt()();
			rv := r.asIdealInt()();
			val := lv.Shr(uint(rv.Value()));
			a.evalIdealInt = func() *bignum.Integer { return val };
		} else {
			a.genBinOpShr(l, r);
		}

	default:
		log.Crashf("Compilation of binary op %v not implemented", op);
	}
}

func (a *exprCompiler) DoBinaryExpr(x *ast.BinaryExpr) {
	l, r := a.copyVisit(x.X), a.copyVisit(x.Y);
	if l.t == nil || r.t == nil {
		return;
	}

	a.doBinaryExpr(x.Op, l, r);
}

func (a *exprCompiler) DoKeyValueExpr(x *ast.KeyValueExpr) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoEllipsis(x *ast.Ellipsis) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoArrayType(x *ast.ArrayType) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoStructType(x *ast.StructType) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoFuncType(x *ast.FuncType) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoInterfaceType(x *ast.InterfaceType) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoMapType(x *ast.MapType) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoChanType(x *ast.ChanType) {
	log.Crash("Not implemented");
}

// TODO(austin) This is a hack to eliminate a circular dependency
// between type.go and expr.go
func (a *compiler) compileArrayLen(scope *Scope, expr ast.Expr) (int64, bool) {
	lenExpr := a.compileExpr(scope, expr, true);
	if lenExpr == nil {
		return 0, false;
	}
	if !lenExpr.t.isInteger() {
		a.diagAt(expr, "array size must be an integer");
		return 0, false;
	}

	if lenExpr.t.isIdeal() {
		lenExpr = lenExpr.convertTo(IntType);
		if lenExpr == nil {
			return 0, false;
		}
	}

	switch _ := lenExpr.t.rep().(type) {
	case *intType:
		return lenExpr.evalInt(nil), true;
	case *uintType:
		return int64(lenExpr.evalUint(nil)), true;
	}
	log.Crashf("unexpected integer type %T", lenExpr.t);
	return 0, false;
}

func (a *compiler) compileExpr(scope *Scope, expr ast.Expr, constant bool) *exprCompiler {
	ec := newExprCompiler(&exprContext{a, scope, constant}, expr.Pos());
	expr.Visit(ec);
	if ec.t == nil {
		return nil;
	}
	return ec;
}

// extractEffect separates out any effects that the expression may
// have, returning a function that will perform those effects and a
// new exprCompiler that is guaranteed to be side-effect free.  These
// are the moral equivalents of "temp := &expr" and "*temp".
//
// Implementation limit: The expression must be addressable.
func (a *exprCompiler) extractEffect() (func(f *Frame), *exprCompiler) {
	if a.evalAddr == nil {
		// This is a much easier case, but the code is
		// completely different.
		log.Crash("extractEffect only implemented for addressable expressions");
	}

	// Create temporary
	tempScope := a.scope;
	tempType := NewPtrType(a.t);
	// TODO(austin) These temporaries accumulate in the scope.
	temp := tempScope.DefineTemp(tempType);
	tempIdx := temp.Index;

	// Generate "temp := &e"
	addr := a.copy();
	addr.t = tempType;
	addr.genUnaryAddrOf(a);

	_, assign := mkAssign(tempType, addr, "", "", 0);
	if assign == nil {
		log.Crashf("extractEffect: mkAssign type check failed");
	}

	effect := func(f *Frame) {
		tempVal := f.Get(tempScope, tempIdx);
		assign(tempVal, f);
	};

	// Generate "*temp"
	getTemp := a.copy();
	getTemp.t = tempType;
	getTemp.genIdentOp(tempScope, tempIdx);

	deref := a.copy();
	deref.t = a.t;
	deref.genStarOp(getTemp);

	return effect, deref;
}

/*
 * Testing interface
 */

type Expr struct {
	t Type;
	f func(f *Frame, out Value);
}

func (expr *Expr) Eval(f *Frame) Value {
	v := expr.t.Zero();
	expr.f(f, v);
	return v;
}

func CompileExpr(scope *Scope, expr ast.Expr) (*Expr, os.Error) {
	errors := scanner.NewErrorVector();
	cc := &compiler{errors};

	ec := cc.compileExpr(scope, expr, false);
	if ec == nil {
		return nil, errors.GetError(scanner.Sorted);
	}
	switch t := ec.t.rep().(type) {
	case *boolType:
		return &Expr{t, func(f *Frame, out Value) { out.(BoolValue).Set(ec.evalBool(f)) }}, nil;
	case *uintType:
		return &Expr{t, func(f *Frame, out Value) { out.(UintValue).Set(ec.evalUint(f)) }}, nil;
	case *intType:
		return &Expr{t, func(f *Frame, out Value) { out.(IntValue).Set(ec.evalInt(f)) }}, nil;
	case *idealIntType:
		return &Expr{t, func(f *Frame, out Value) { out.(*idealIntV).V = ec.evalIdealInt() }}, nil;
	case *floatType:
		return &Expr{t, func(f *Frame, out Value) { out.(FloatValue).Set(ec.evalFloat(f)) }}, nil;
	case *idealFloatType:
		return &Expr{t, func(f *Frame, out Value) { out.(*idealFloatV).V = ec.evalIdealFloat() }}, nil;
	case *stringType:
		return &Expr{t, func(f *Frame, out Value) { out.(StringValue).Set(ec.evalString(f)) }}, nil;
	case *PtrType:
		return &Expr{t, func(f *Frame, out Value) { out.(PtrValue).Set(ec.evalPtr(f)) }}, nil;
	case *FuncType:
		return &Expr{t, func(f *Frame, out Value) { out.(FuncValue).Set(ec.evalFunc(f)) }}, nil;
	}
	log.Crashf("unexpected type %v", ec.t);
	panic();
}

/*
 * Operator generators
 * Everything below here is MACHINE GENERATED by gen.py genOps
 */

func (a *exprCompiler) genConstant(v Value) {
	switch _ := a.t.rep().(type) {
	case *boolType:
		val := v.(BoolValue).Get();
		a.evalBool = func(f *Frame) bool { return val };
	case *uintType:
		val := v.(UintValue).Get();
		a.evalUint = func(f *Frame) uint64 { return val };
	case *intType:
		val := v.(IntValue).Get();
		a.evalInt = func(f *Frame) int64 { return val };
	case *idealIntType:
		val := v.(IdealIntValue).Get();
		a.evalIdealInt = func() *bignum.Integer { return val };
	case *floatType:
		val := v.(FloatValue).Get();
		a.evalFloat = func(f *Frame) float64 { return val };
	case *idealFloatType:
		val := v.(IdealFloatValue).Get();
		a.evalIdealFloat = func() *bignum.Rational { return val };
	case *stringType:
		val := v.(StringValue).Get();
		a.evalString = func(f *Frame) string { return val };
	case *ArrayType:
		val := v.(ArrayValue).Get();
		a.evalArray = func(f *Frame) ArrayValue { return val };
	case *PtrType:
		val := v.(PtrValue).Get();
		a.evalPtr = func(f *Frame) Value { return val };
	case *FuncType:
		val := v.(FuncValue).Get();
		a.evalFunc = func(f *Frame) Func { return val };
	default:
		log.Crashf("unexpected constant type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genIdentOp(s *Scope, index int) {
	a.evalAddr = func(f *Frame) Value { return f.Get(s, index) };
	switch _ := a.t.rep().(type) {
	case *boolType:
		a.evalBool = func(f *Frame) bool { return f.Get(s, index).(BoolValue).Get() };
	case *uintType:
		a.evalUint = func(f *Frame) uint64 { return f.Get(s, index).(UintValue).Get() };
	case *intType:
		a.evalInt = func(f *Frame) int64 { return f.Get(s, index).(IntValue).Get() };
	case *floatType:
		a.evalFloat = func(f *Frame) float64 { return f.Get(s, index).(FloatValue).Get() };
	case *stringType:
		a.evalString = func(f *Frame) string { return f.Get(s, index).(StringValue).Get() };
	case *ArrayType:
		a.evalArray = func(f *Frame) ArrayValue { return f.Get(s, index).(ArrayValue).Get() };
	case *PtrType:
		a.evalPtr = func(f *Frame) Value { return f.Get(s, index).(PtrValue).Get() };
	case *FuncType:
		a.evalFunc = func(f *Frame) Func { return f.Get(s, index).(FuncValue).Get() };
	default:
		log.Crashf("unexpected identifier type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genIndexArray(l *exprCompiler, r *exprCompiler) {
	lf := l.asArray();
	rf := r.asInt();
	switch _ := a.t.rep().(type) {
	case *boolType:
		a.evalBool = func(f *Frame) bool { return lf(f).Elem(rf(f)).(BoolValue).Get() };
	case *uintType:
		a.evalUint = func(f *Frame) uint64 { return lf(f).Elem(rf(f)).(UintValue).Get() };
	case *intType:
		a.evalInt = func(f *Frame) int64 { return lf(f).Elem(rf(f)).(IntValue).Get() };
	case *floatType:
		a.evalFloat = func(f *Frame) float64 { return lf(f).Elem(rf(f)).(FloatValue).Get() };
	case *stringType:
		a.evalString = func(f *Frame) string { return lf(f).Elem(rf(f)).(StringValue).Get() };
	case *ArrayType:
		a.evalArray = func(f *Frame) ArrayValue { return lf(f).Elem(rf(f)).(ArrayValue).Get() };
	case *PtrType:
		a.evalPtr = func(f *Frame) Value { return lf(f).Elem(rf(f)).(PtrValue).Get() };
	case *FuncType:
		a.evalFunc = func(f *Frame) Func { return lf(f).Elem(rf(f)).(FuncValue).Get() };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genFuncCall(call func(f *Frame) []Value) {
	switch _ := a.t.rep().(type) {
	case *boolType:
		a.evalBool = func(f *Frame) bool { return call(f)[0].(BoolValue).Get() };
	case *uintType:
		a.evalUint = func(f *Frame) uint64 { return call(f)[0].(UintValue).Get() };
	case *intType:
		a.evalInt = func(f *Frame) int64 { return call(f)[0].(IntValue).Get() };
	case *floatType:
		a.evalFloat = func(f *Frame) float64 { return call(f)[0].(FloatValue).Get() };
	case *stringType:
		a.evalString = func(f *Frame) string { return call(f)[0].(StringValue).Get() };
	case *ArrayType:
		a.evalArray = func(f *Frame) ArrayValue { return call(f)[0].(ArrayValue).Get() };
	case *PtrType:
		a.evalPtr = func(f *Frame) Value { return call(f)[0].(PtrValue).Get() };
	case *FuncType:
		a.evalFunc = func(f *Frame) Func { return call(f)[0].(FuncValue).Get() };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genStarOp(v *exprCompiler) {
	vf := v.asPtr();
	a.evalAddr = func(f *Frame) Value { return vf(f) };
	switch _ := a.t.rep().(type) {
	case *boolType:
		a.evalBool = func(f *Frame) bool { return vf(f).(BoolValue).Get() };
	case *uintType:
		a.evalUint = func(f *Frame) uint64 { return vf(f).(UintValue).Get() };
	case *intType:
		a.evalInt = func(f *Frame) int64 { return vf(f).(IntValue).Get() };
	case *floatType:
		a.evalFloat = func(f *Frame) float64 { return vf(f).(FloatValue).Get() };
	case *stringType:
		a.evalString = func(f *Frame) string { return vf(f).(StringValue).Get() };
	case *ArrayType:
		a.evalArray = func(f *Frame) ArrayValue { return vf(f).(ArrayValue).Get() };
	case *PtrType:
		a.evalPtr = func(f *Frame) Value { return vf(f).(PtrValue).Get() };
	case *FuncType:
		a.evalFunc = func(f *Frame) Func { return vf(f).(FuncValue).Get() };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genUnaryOpNeg(v *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		vf := v.asUint();
		a.evalUint = func(f *Frame) uint64 { return -vf(f) };
	case *intType:
		vf := v.asInt();
		a.evalInt = func(f *Frame) int64 { return -vf(f) };
	case *idealIntType:
		vf := v.asIdealInt();
		val := vf().Neg();
		a.evalIdealInt = func() *bignum.Integer { return val };
	case *floatType:
		vf := v.asFloat();
		a.evalFloat = func(f *Frame) float64 { return -vf(f) };
	case *idealFloatType:
		vf := v.asIdealFloat();
		val := vf().Neg();
		a.evalIdealFloat = func() *bignum.Rational { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genUnaryOpNot(v *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *boolType:
		vf := v.asBool();
		a.evalBool = func(f *Frame) bool { return !vf(f) };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genUnaryOpXor(v *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		vf := v.asUint();
		a.evalUint = func(f *Frame) uint64 { return ^vf(f) };
	case *intType:
		vf := v.asInt();
		a.evalInt = func(f *Frame) int64 { return ^vf(f) };
	case *idealIntType:
		vf := v.asIdealInt();
		val := vf().Neg().Sub(bignum.Int(1));
		a.evalIdealInt = func() *bignum.Integer { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpAdd(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) + rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) + rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Add(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalFloat = func(f *Frame) float64 { return lf(f) + rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Add(rf());
		a.evalIdealFloat = func() *bignum.Rational { return val };
	case *stringType:
		lf := l.asString();
		rf := r.asString();
		a.evalString = func(f *Frame) string { return lf(f) + rf(f) };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpSub(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) - rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) - rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Sub(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalFloat = func(f *Frame) float64 { return lf(f) - rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Sub(rf());
		a.evalIdealFloat = func() *bignum.Rational { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpMul(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) * rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) * rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Mul(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalFloat = func(f *Frame) float64 { return lf(f) * rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Mul(rf());
		a.evalIdealFloat = func() *bignum.Rational { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpQuo(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) / rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) / rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Quo(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalFloat = func(f *Frame) float64 { return lf(f) / rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Quo(rf());
		a.evalIdealFloat = func() *bignum.Rational { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpRem(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) % rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) % rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Rem(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpAnd(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) & rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) & rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().And(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpOr(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) | rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) | rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Or(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpXor(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) ^ rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) ^ rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Xor(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpAndNot(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) &^ rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { return lf(f) &^ rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().AndNot(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpShl(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) << rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asUint();
		a.evalInt = func(f *Frame) int64 { return lf(f) << rf(f) };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *exprCompiler) genBinOpShr(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.rep().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) >> rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asUint();
		a.evalInt = func(f *Frame) int64 { return lf(f) >> rf(f) };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func genAssign(lt Type, r *exprCompiler) (func(lv Value, f *Frame)) {
	switch _ := lt.rep().(type) {
	case *boolType:
		rf := r.asBool();
		return func(lv Value, f *Frame) { lv.(BoolValue).Set(rf(f)) };
	case *uintType:
		rf := r.asUint();
		return func(lv Value, f *Frame) { lv.(UintValue).Set(rf(f)) };
	case *intType:
		rf := r.asInt();
		return func(lv Value, f *Frame) { lv.(IntValue).Set(rf(f)) };
	case *floatType:
		rf := r.asFloat();
		return func(lv Value, f *Frame) { lv.(FloatValue).Set(rf(f)) };
	case *stringType:
		rf := r.asString();
		return func(lv Value, f *Frame) { lv.(StringValue).Set(rf(f)) };
	case *ArrayType:
		rf := r.asArray();
		return func(lv Value, f *Frame) { lv.Assign(rf(f)) };
	case *PtrType:
		rf := r.asPtr();
		return func(lv Value, f *Frame) { lv.(PtrValue).Set(rf(f)) };
	case *FuncType:
		rf := r.asFunc();
		return func(lv Value, f *Frame) { lv.(FuncValue).Set(rf(f)) };
	default:
		log.Crashf("unexpected left operand type %v at %v", lt, r.pos);
	}
	panic();
}
