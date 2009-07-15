// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"eval";
	"go/ast";
	"go/token";
	"log";
	"strconv";
	"strings";
)

// An exprContext stores information used throughout the compilation
// of an entire expression.
type exprContext struct {
	scope *Scope;
	constant bool;
	// TODO(austin) Error list
}

// An exprCompiler compiles a single node in an expression.  It stores
// the whole expression's context plus information specific to this node.
// After compilation, it stores the type of the expression and its
// evaluator function.
type exprCompiler struct {
	*exprContext;
	pos token.Position;
	t Type;
	// TODO(austin) Should there be separate f's for each specific
	// Value interface?  We spend a lot of time calling f's and
	// just blindly casting the result since we already know its type.
	f func (f *Frame) Value;
	// A short string describing this expression for error
	// messages.  Only necessary if t != nil.
	desc string;
	// True if the address-of operator can be applied to this
	// result.
	addressable bool;
}

func newExprCompiler(c *exprContext, pos token.Position) *exprCompiler {
	return &exprCompiler{c, pos, nil, nil, "<missing description>", false};
}

func (a *exprCompiler) fork(x ast.Expr) *exprCompiler {
	ec := newExprCompiler(a.exprContext, x.Pos());
	x.Visit(ec);
	return ec;
}

func (a *exprCompiler) diag(format string, args ...) {
	diag(a.pos, format, args);
}

func (a *exprCompiler) diagOpType(op token.Token, vt Type) {
	a.diag("illegal operand type for '%v' operator\n\t%v", op, vt);
}

func (a *exprCompiler) diagOpTypes(op token.Token, lt Type, rt Type) {
	a.diag("illegal operand types for '%v' operator\n\t%v\n\t%v", op, lt, rt);
}

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
		a.f = func (*Frame) Value { return def.Value };
		a.desc = "constant";
	case *Variable:
		if a.constant {
			a.diag("expression must be a constant");
			return;
		}
		a.t = def.Type;
		defidx := def.Index;
		a.f = func (f *Frame) Value {
			// TODO(austin) Make Frame do this?
			for f.Scope != dscope {
				f = f.Outer;
			}
			return f.Vars[defidx];
		};
		a.desc = "variable";
		a.addressable = true;
	case Type:
		a.diag("type %v used as expression", x.Value);
	default:
		log.Crashf("name %s has unknown type %T", x.Value, def);
	}
}

func (a *exprCompiler) doIdealInt(i *bignum.Integer) {
	a.t = IdealIntType;
	val := &idealIntV{i};
	a.f = func (*Frame) Value { return val };
}

func (a *exprCompiler) DoIntLit(x *ast.IntLit) {
	i, _, _2 := bignum.IntFromString(string(x.Value), 0);
	a.doIdealInt(i);
	a.desc = "integer literal";
}

func (a *exprCompiler) DoCharLit(x *ast.CharLit) {
	if x.Value[0] != '\'' {
		// Shouldn't get past the parser
		log.Crashf("unexpected character literal %s at %v", x.Value, x.Pos());
	}
	v, mb, tail, err := strconv.UnquoteChar(string(x.Value[1:len(x.Value)]), '\'');
	if err != nil {
		a.diag("illegal character literal, %v", err);
		return;
	}
	if tail != "'" {
		a.diag("character literal must contain only one character");
		return;
	}
	a.doIdealInt(bignum.Int(int64(v)));
	a.desc = "character literal";
}

func (a *exprCompiler) DoFloatLit(x *ast.FloatLit) {
	a.t = IdealFloatType;
	i, _, _2 := bignum.RatFromString(string(x.Value), 0);
	val := &idealFloatV{i};
	a.f = func (*Frame) Value { return val };
	a.desc = "float literal";
}

func (a *exprCompiler) doString(s string) {
	a.t = StringType;
	val := stringV(s);
	a.f = func (*Frame) Value { return &val };
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
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoTypeAssertExpr(x *ast.TypeAssertExpr) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoCallExpr(x *ast.CallExpr) {
	log.Crash("Not implemented");
}

func (a *exprCompiler) DoStarExpr(x *ast.StarExpr) {
	v := a.fork(x.X);
	if v.t == nil {
		return;
	}

	switch vt := v.t.(type) {
	case *PtrType:
		a.t = vt.Elem();
		vf := v.f;
		a.f = func (f *Frame) Value { return vf(f).(PtrValue).Get() };
		a.desc = "* expression";
		a.addressable = true;

	default:
		a.diagOpType(token.MUL, v.t);
	}
}

func (a *exprCompiler) DoUnaryExpr(x *ast.UnaryExpr) {
	switch x.Op {
	case token.SUB:
		// Negation
		v := a.fork(x.X);
		if v.t == nil {
			return;
		}

		a.t = v.t;
		vf := v.f;
		switch vt := v.t.literal().(type) {
		case *uintType:
			a.f = func (f *Frame) Value {
				return vt.value(-vf(f).(UintValue).Get());
			};
		case *intType:
			a.f = func (f *Frame) Value {
				return vt.value(-vf(f).(IntValue).Get());
			};
		case *idealIntType:
			val := vt.value(vf(nil).(IdealIntValue).Get().Neg());
			a.f = func (f *Frame) Value { return val };
		case *floatType:
			a.f = func (f *Frame) Value {
				return vt.value(-vf(f).(FloatValue).Get());
			};
		case *idealFloatType:
			val := vt.value(vf(nil).(IdealFloatValue).Get().Neg());
			a.f = func (f *Frame) Value { return val };
		default:
			a.t = nil;
			a.diagOpType(x.Op, v.t);
			return;
		}

	case token.AND:
		// Address-of
		v := a.fork(x.X);
		if v.t == nil {
			return;
		}

		// The unary prefix address-of operator & generates
		// the address of its operand, which must be a
		// variable, pointer indirection, field selector, or
		// array or slice indexing operation.
		if !v.addressable {
			a.diag("cannot take the address of %s", v.desc);
			return;
		}

		// TODO(austin) Implement "It is illegal to take the
		// address of a function result variable" once I have
		// function result variables.

		at := NewPtrType(v.t);
		a.t = at;

		vf := v.f;
		a.f = func (f *Frame) Value { return at.value(vf(f)) };
		a.desc = "& expression";

	default:
		log.Crashf("Unary op %v not implemented", x.Op);
	}
}

// a.convertTo(t) converts the value of the analyzed expression a,
// which must be a constant, ideal number, to a new analyzed
// expression with a constant value of type t.
func (a *exprCompiler) convertTo(t Type) *exprCompiler {
	if !a.t.isIdeal() {
		log.Crashf("attempted to convert from %v, expected ideal", a.t);
	}

	val := a.f(nil);
	var rat *bignum.Rational;

	// It is erroneous to assign a value with a non-zero
	// fractional part to an integer, or if the assignment would
	// overflow or underflow, or in general if the value cannot be
	// represented by the type of the variable.
	switch a.t {
	case IdealFloatType:
		rat = val.(IdealFloatValue).Get();
		if t.isInteger() && !rat.IsInt() {
			a.diag("constant %v truncated to integer", ratToString(rat));
			return nil;
		}
	case IdealIntType:
		rat = bignum.MakeRat(val.(IdealIntValue).Get(), bignum.Nat(1));
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
	switch t := t.(type) {
	case *uintType:
		n, d := rat.Value();
		f := n.Quo(bignum.MakeInt(false, d));
		v := f.Abs().Value();
		val = t.value(v);
	case *intType:
		n, d := rat.Value();
		f := n.Quo(bignum.MakeInt(false, d));
		v := f.Value();
		val = t.value(v);
	case *idealIntType:
		n, d := rat.Value();
		f := n.Quo(bignum.MakeInt(false, d));
		val = t.value(f);
	case *floatType:
		n, d := rat.Value();
		v := float64(n.Value())/float64(d.Value());
		val = t.value(v);
	case *idealFloatType:
		val = t.value(rat);
	default:
		log.Crashf("cannot convert to type %T", t);
	}

	res := newExprCompiler(a.exprContext, a.pos);
	res.t = t;
	res.f = func (*Frame) Value { return val };
	res.desc = a.desc;
	return res;
}

var opDescs = make(map[token.Token] string)

func (a *exprCompiler) DoBinaryExpr(x *ast.BinaryExpr) {
	l, r := a.fork(x.X), a.fork(x.Y);
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

	// Except in shift expressions, if one operand has numeric
	// type and the other operand is an ideal number, the ideal
	// number is converted to match the type of the other operand.
	if x.Op != token.SHL && x.Op != token.SHR {
		if l.t.isInteger() && !l.t.isIdeal() && r.t.isIdeal() {
			r = r.convertTo(l.t);
		} else if r.t.isInteger() && !r.t.isIdeal() && l.t.isIdeal() {
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
		// The right operand in a shift operation must be
		// always be of unsigned integer type or an ideal
		// number that can be safely converted into an
		// unsigned integer type.
		if r.t.isIdeal() {
			r = r.convertTo(UintType);
			if r == nil {
				return;
			}
		}

		if !integers() {
			a.diagOpTypes(x.Op, origlt, origrt);
			return;
		}
		if _, ok := r.t.literal().(*uintType); !ok {
			a.diag("right operand of shift must be unsigned");
			return;
		}
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
	a.desc, ok = opDescs[x.Op];
	if !ok {
		a.desc = x.Op.String() + " expression";
		opDescs[x.Op] = a.desc;
	}

	// Compile
	// TODO(austin) There has got to be a better way to do this.
	lf := l.f;
	rf := r.f;
	switch x.Op {
	case token.ADD:
		switch lt := l.t.literal().(type) {
		case *uintType:
			// TODO(austin) lt.value allocates.  It would
			// be awesome if we could avoid that for
			// intermediate values.  That might be
			// possible if we pass the closure a place to
			// store its result.
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(UintValue).Get() + rf(f).(UintValue).Get());
			};
		case *intType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(IntValue).Get() + rf(f).(IntValue).Get());
			};
		case *idealIntType:
			val := lt.value(lf(nil).(IdealIntValue).Get().Add(rf(nil).(IdealIntValue).Get()));
			a.f = func (f *Frame) Value { return val };
		case *floatType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(FloatValue).Get() + rf(f).(FloatValue).Get());
			};
		case *idealFloatType:
			val := lt.value(lf(nil).(IdealFloatValue).Get().Add(rf(nil).(IdealFloatValue).Get()));
			a.f = func (f *Frame) Value { return val };
		case *stringType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(StringValue).Get() + rf(f).(StringValue).Get());
			};
		default:
			// Shouldn't have passed type checking
			log.Crashf("unexpected left operand type %v at %v", l.t.literal(), x.Pos());
		}

	case token.SUB:
		switch lt := l.t.literal().(type) {
		case *uintType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(UintValue).Get() - rf(f).(UintValue).Get());
			};
		case *intType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(IntValue).Get() - rf(f).(IntValue).Get());
			};
		case *idealIntType:
			val := lt.value(lf(nil).(IdealIntValue).Get().Sub(rf(nil).(IdealIntValue).Get()));
			a.f = func (f *Frame) Value { return val };
		case *floatType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(FloatValue).Get() - rf(f).(FloatValue).Get());
			};
		case *idealFloatType:
			val := lt.value(lf(nil).(IdealFloatValue).Get().Sub(rf(nil).(IdealFloatValue).Get()));
			a.f = func (f *Frame) Value { return val };
		default:
			// Shouldn't have passed type checking
			log.Crashf("unexpected left operand type %v at %v", l.t.literal(), x.Pos());
		}

	case token.QUO:
		// TODO(austin) What if divisor is zero?
		switch lt := l.t.literal().(type) {
		case *uintType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(UintValue).Get() / rf(f).(UintValue).Get());
			};
		case *intType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(IntValue).Get() / rf(f).(IntValue).Get());
			};
		case *idealIntType:
			val := lt.value(lf(nil).(IdealIntValue).Get().Quo(rf(nil).(IdealIntValue).Get()));
			a.f = func (f *Frame) Value { return val };
		case *floatType:
			a.f = func (f *Frame) Value {
				return lt.value(lf(f).(FloatValue).Get() / rf(f).(FloatValue).Get());
			};
		case *idealFloatType:
			val := lt.value(lf(nil).(IdealFloatValue).Get().Quo(rf(nil).(IdealFloatValue).Get()));
			a.f = func (f *Frame) Value { return val };
		default:
			// Shouldn't have passed type checking
			log.Crashf("unexpected left operand type %v at %v", l.t.literal(), x.Pos());
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

func compileExpr(expr ast.Expr, scope *Scope) *exprCompiler {
	ec := newExprCompiler(&exprContext{scope, false}, expr.Pos());
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
	f func (f *Frame) Value;
}

func (expr *Expr) Eval(f *Frame) Value {
	return expr.f(f);
}

func CompileExpr(expr ast.Expr, scope *Scope) *Expr {
	ec := compileExpr(expr, scope);
	if ec == nil {
		return nil;
	}
	return &Expr{ec.f};
}
