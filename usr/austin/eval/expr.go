// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"eval";
	"fmt";
	"go/ast";
	"go/scanner";
	"go/token";
	"log";
	"os";
	"strconv";
	"strings";
)

// An exprContext stores information used throughout the compilation
// of an entire expression.
type exprContext struct {
	scope *Scope;
	constant bool;
	errors scanner.ErrorHandler;
}

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
	evalIdealInt func() *bignum.Integer;
	evalFloat func(f *Frame) float64;
	evalIdealFloat func() *bignum.Rational;
	evalString func(f *Frame) string;
	evalArray func(f *Frame) ArrayValue;
	evalPtr func(f *Frame) Value;
	// Evaluate to the "address of" this value; that is, the
	// settable Value object.  nil for expressions whose address
	// cannot be taken.
	evalAddr func(f *Frame) Value;
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
func (a *exprCompiler) genIdentOp(t Type, s *Scope, index int)
func (a *exprCompiler) genIndexArray(l *exprCompiler, r *exprCompiler)
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
	a.errors.Error(a.pos, fmt.Sprintf(format, args));
}

func (a *exprCompiler) diagOpType(op token.Token, vt Type) {
	a.diag("illegal operand type for '%v' operator\n\t%v", op, vt);
}

func (a *exprCompiler) diagOpTypes(op token.Token, lt Type, rt Type) {
	a.diag("illegal operand types for '%v' operator\n\t%v\n\t%v", op, lt, rt);
}

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

// TODO(austin) Move convertTo somewhere more reasonable
func (a *exprCompiler) convertTo(t Type) *exprCompiler

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
		switch _ := a.t.literal().(type) {
		case *idealIntType:
			val := def.Value.(IdealIntValue).Get();
			a.evalIdealInt = func() *bignum.Integer { return val; };
		case *idealFloatType:
			val := def.Value.(IdealFloatValue).Get();
			a.evalIdealFloat = func() *bignum.Rational { return val; };
		default:
			log.Crashf("unexpected constant type: %v", a.t);
		}
		a.desc = "constant";
	case *Variable:
		if a.constant {
			a.diag("expression must be a constant");
			return;
		}
		a.t = def.Type;
		defidx := def.Index;
		a.genIdentOp(def.Type, dscope, defidx);
		a.evalAddr = func(f *Frame) Value {
			return f.Get(dscope, defidx);
		};
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
	log.Crash("Not implemented");
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
	if lt, ok := l.t.literal().(*PtrType); ok {
		if et, ok := lt.Elem.literal().(*ArrayType); ok {
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

	switch lt := l.t.literal().(type) {
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
		switch _ := r.t.literal().(type) {
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
	switch lt := l.t.literal().(type) {
	case *ArrayType:
		a.t = lt.Elem;
		// TODO(austin) Bounds check
		a.genIndexArray(l, r);
		lf := l.asArray();
		rf := r.asInt();
		a.evalAddr = func(f *Frame) Value {
			return lf(f).Elem(rf(f));
		};

	default:
		log.Crashf("Compilation of index into %T not implemented", l.t.literal());
	}
}

func (a *exprCompiler) DoTypeAssertExpr(x *ast.TypeAssertExpr) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoCallExpr(x *ast.CallExpr) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoStarExpr(x *ast.StarExpr) {
	v := a.copyVisit(x.X);
	if v.t == nil {
		return;
	}

	switch vt := v.t.literal().(type) {
	case *PtrType:
		// TODO(austin) If this is vt.Elem() I get a
		// "call of a non-function: Type" error
		a.t = vt.Elem;
		a.genStarOp(v);
		vf := v.asPtr();
		a.evalAddr = func(f *Frame) Value { return vf(f) };
		a.desc = "* expression";

	default:
		a.diagOpType(token.MUL, v.t);
	}
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
		// TODO(austin) Unnamed bool?  Named bool?
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
		vf := v.evalAddr;
		a.evalPtr = func(f *Frame) Value { return vf(f) };

	default:
		log.Crashf("Compilation of unary op %v not implemented", x.Op);
	}
}

// a.convertTo(t) converts the value of the analyzed expression a,
// which must be a constant, ideal number, to a new analyzed
// expression with a constant value of type t.
func (a *exprCompiler) convertTo(t Type) *exprCompiler {
	if !a.t.isIdeal() {
		log.Crashf("attempted to convert from %v, expected ideal", a.t);
	}

	var rat *bignum.Rational;

	// It is erroneous to assign a value with a non-zero
	// fractional part to an integer, or if the assignment would
	// overflow or underflow, or in general if the value cannot be
	// represented by the type of the variable.
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
	if t, ok := t.(BoundedType); ok {
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
	switch t := t.(type) {
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

var binOpDescs = make(map[token.Token] string)

func (a *exprCompiler) DoBinaryExpr(x *ast.BinaryExpr) {
	l, r := a.copyVisit(x.X), a.copyVisit(x.Y);
	if l.t == nil || r.t == nil {
		return;
	}

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

	if x.Op != token.SHL && x.Op != token.SHR {
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

	// Useful type predicates
	// TODO(austin) The spec is wrong here.  The types must be
	// identical, not compatible.
	compat := func() bool {
		return l.t.compatible(r.t);
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
		// TODO(austin) Deal with named types
		return l.t == BoolType && r.t == BoolType;
	};

	// Type check
	switch x.Op {
	case token.ADD:
		if !compat() || (!integers() && !floats() && !strings()) {
			a.diagOpTypes(x.Op, origlt, origrt);
			return;
		}
		a.t = l.t;

	case token.SUB, token.MUL, token.QUO:
		if !compat() || (!integers() && !floats()) {
			a.diagOpTypes(x.Op, origlt, origrt);
			return;
		}
		a.t = l.t;

	case token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
		if !compat() || !integers() {
			a.diagOpTypes(x.Op, origlt, origrt);
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
			a.diagOpTypes(x.Op, origlt, origrt);
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
		} else if _, ok := r.t.literal().(*uintType); !ok {
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

		// TODO(austin) Named bool type
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
		if l.t.literal() == BoolType || r.t.literal() == BoolType {
			a.diagOpTypes(x.Op, origlt, origrt);
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
		log.Crashf("Binary op %v not implemented", x.Op);
		// TODO(austin) Unnamed bool?  Named bool?
		a.t = BoolType;

	default:
		log.Crashf("unknown binary operator %v", x.Op);
	}

	var ok bool;
	a.desc, ok = binOpDescs[x.Op];
	if !ok {
		a.desc = x.Op.String() + " expression";
		binOpDescs[x.Op] = a.desc;
	}

	// Compile
	switch x.Op {
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
		log.Crashf("Compilation of binary op %v not implemented", x.Op);
	}
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

func compileExpr(expr ast.Expr, scope *Scope, errors scanner.ErrorHandler) *exprCompiler {
	ec := newExprCompiler(&exprContext{scope, false, errors}, expr.Pos());
	expr.Visit(ec);
	if ec.t == nil {
		return nil;
	}
	return ec;
}

/*
 * Public interface
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

func CompileExpr(expr ast.Expr, scope *Scope) (*Expr, os.Error) {
	errors := scanner.NewErrorVector();

	ec := compileExpr(expr, scope, errors);
	if ec == nil {
		return nil, errors.GetError(scanner.Sorted);
	}
	switch t := ec.t.(type) {
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
	}
	log.Crashf("unexpected type %v", ec.t);
	panic();
}

/*
 * Operator generators
 * Everything below here is MACHINE GENERATED by gen.py genOps
 */

func (a *exprCompiler) genIdentOp(t Type, s *Scope, index int) {
	switch _ := t.literal().(type) {
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
	default:
		log.Crashf("unexpected identifier type %v at %v", t.literal(), a.pos);
	}
}

func (a *exprCompiler) genIndexArray(l *exprCompiler, r *exprCompiler) {
	lf := l.asArray();
	rf := r.asInt();
	switch _ := a.t.literal().(type) {
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
	default:
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genStarOp(v *exprCompiler) {
	vf := v.asPtr();
	switch _ := a.t.literal().(type) {
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
	default:
		log.Crashf("unexpected result type %v at %v", v.t.literal().(*PtrType).Elem.literal(), a.pos);
	}
}

func (a *exprCompiler) genUnaryOpNeg(v *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", v.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genUnaryOpNot(v *exprCompiler) {
	switch _ := a.t.literal().(type) {
	case *boolType:
		vf := v.asBool();
		a.evalBool = func(f *Frame) bool { return !vf(f) };
	default:
		log.Crashf("unexpected result type %v at %v", v.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genUnaryOpXor(v *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", v.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpAdd(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpSub(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpMul(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpQuo(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpRem(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpAnd(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpOr(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpXor(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpAndNot(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
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
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpShl(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) << rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asUint();
		a.evalInt = func(f *Frame) int64 { return lf(f) << rf(f) };
	default:
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}

func (a *exprCompiler) genBinOpShr(l *exprCompiler, r *exprCompiler) {
	switch _ := a.t.literal().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { return lf(f) >> rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asUint();
		a.evalInt = func(f *Frame) int64 { return lf(f) >> rf(f) };
	default:
		log.Crashf("unexpected result type %v at %v", l.t.literal(), a.pos);
	}
}
