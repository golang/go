// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"go/ast";
	"go/scanner";
	"go/token";
	"log";
	"os";
	"strconv";
	"strings";
)

// An expr is the result of compiling an expression.  It stores the
// type of the expression and its evaluator function.
type expr struct {
	*exprInfo;
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
	evalStruct func(f *Frame) StructValue;
	evalPtr func(f *Frame) Value;
	evalFunc func(f *Frame) Func;
	evalSlice func(f *Frame) Slice;
	evalMap func(f *Frame) Map;
	evalMulti func(f *Frame) []Value;
	// Map index expressions permit special forms of assignment,
	// for which we need to know the Map and key.
	evalMapValue func(f *Frame) (Map, interface{});
	// Evaluate to the "address of" this value; that is, the
	// settable Value object.  nil for expressions whose address
	// cannot be taken.
	evalAddr func(f *Frame) Value;
	// Execute this expression as a statement.  Only expressions
	// that are valid expression statements should set this.
	exec func(f *Frame);
	// A short string describing this expression for error
	// messages.
	desc string;
}

// exprInfo stores information needed to compile any expression node.
// Each expr also stores its exprInfo so further expressions can be
// compiled from it.
type exprInfo struct {
	*compiler;
	pos token.Position;
}

func (a *exprInfo) newExpr(t Type, desc string) *expr {
	return &expr{exprInfo: a, t: t, desc: desc};
}

func (a *exprInfo) diag(format string, args ...) {
	a.diagAt(&a.pos, format, args);
}

func (a *exprInfo) diagOpType(op token.Token, vt Type) {
	a.diag("illegal operand type for '%v' operator\n\t%v", op, vt);
}

func (a *exprInfo) diagOpTypes(op token.Token, lt Type, rt Type) {
	a.diag("illegal operand types for '%v' operator\n\t%v\n\t%v", op, lt, rt);
}

/*
 * "As" functions.  These retrieve evaluator functions from an
 * expr, panicking if the requested evaluator is nil.
 */

func (a *expr) asBool() (func(f *Frame) bool) {
	if a.evalBool == nil {
		log.Crashf("tried to get %v node as boolType", a.t);
	}
	return a.evalBool;
}

func (a *expr) asUint() (func(f *Frame) uint64) {
	if a.evalUint == nil {
		log.Crashf("tried to get %v node as uintType", a.t);
	}
	return a.evalUint;
}

func (a *expr) asInt() (func(f *Frame) int64) {
	if a.evalInt == nil {
		log.Crashf("tried to get %v node as intType", a.t);
	}
	return a.evalInt;
}

func (a *expr) asIdealInt() (func() *bignum.Integer) {
	if a.evalIdealInt == nil {
		log.Crashf("tried to get %v node as idealIntType", a.t);
	}
	return a.evalIdealInt;
}

func (a *expr) asFloat() (func(f *Frame) float64) {
	if a.evalFloat == nil {
		log.Crashf("tried to get %v node as floatType", a.t);
	}
	return a.evalFloat;
}

func (a *expr) asIdealFloat() (func() *bignum.Rational) {
	if a.evalIdealFloat == nil {
		log.Crashf("tried to get %v node as idealFloatType", a.t);
	}
	return a.evalIdealFloat;
}

func (a *expr) asString() (func(f *Frame) string) {
	if a.evalString == nil {
		log.Crashf("tried to get %v node as stringType", a.t);
	}
	return a.evalString;
}

func (a *expr) asArray() (func(f *Frame) ArrayValue) {
	if a.evalArray == nil {
		log.Crashf("tried to get %v node as ArrayType", a.t);
	}
	return a.evalArray;
}

func (a *expr) asStruct() (func(f *Frame) StructValue) {
	if a.evalStruct == nil {
		log.Crashf("tried to get %v node as StructType", a.t);
	}
	return a.evalStruct;
}

func (a *expr) asPtr() (func(f *Frame) Value) {
	if a.evalPtr == nil {
		log.Crashf("tried to get %v node as PtrType", a.t);
	}
	return a.evalPtr;
}

func (a *expr) asFunc() (func(f *Frame) Func) {
	if a.evalFunc == nil {
		log.Crashf("tried to get %v node as FuncType", a.t);
	}
	return a.evalFunc;
}

func (a *expr) asSlice() (func(f *Frame) Slice) {
	if a.evalSlice == nil {
		log.Crashf("tried to get %v node as SliceType", a.t);
	}
	return a.evalSlice;
}

func (a *expr) asMap() (func(f *Frame) Map) {
	if a.evalMap == nil {
		log.Crashf("tried to get %v node as MapType", a.t);
	}
	return a.evalMap;
}

func (a *expr) asMulti() (func(f *Frame) []Value) {
	if a.evalMulti == nil {
		log.Crashf("tried to get %v node as MultiType", a.t);
	}
	return a.evalMulti;
}

func (a *expr) asInterface() (func(f *Frame) interface {}) {
	switch _ := a.t.lit().(type) {
	case *boolType:
		sf := a.asBool();
		return func(f *Frame) interface {} { return sf(f) };
	case *uintType:
		sf := a.asUint();
		return func(f *Frame) interface {} { return sf(f) };
	case *intType:
		sf := a.asInt();
		return func(f *Frame) interface {} { return sf(f) };
	case *floatType:
		sf := a.asFloat();
		return func(f *Frame) interface {} { return sf(f) };
	case *stringType:
		sf := a.asString();
		return func(f *Frame) interface {} { return sf(f) };
	case *PtrType:
		sf := a.asPtr();
		return func(f *Frame) interface {} { return sf(f) };
	case *FuncType:
		sf := a.asFunc();
		return func(f *Frame) interface {} { return sf(f) };
	case *MapType:
		sf := a.asMap();
		return func(f *Frame) interface {} { return sf(f) };
	default:
		log.Crashf("unexpected expression node type %v at %v", a.t, a.pos);
	}
	panic();
}

/*
 * Common expression manipulations
 */

// a.convertTo(t) converts the value of the analyzed expression a,
// which must be a constant, ideal number, to a new analyzed
// expression with a constant value of type t.
//
// TODO(austin) Rename to resolveIdeal or something?
func (a *expr) convertTo(t Type) *expr {
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
	if t, ok := t.lit().(BoundedType); ok {
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
	res := a.newExpr(t, a.desc);
	switch t := t.lit().(type) {
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

/*
 * Assignments
 */

// An assignCompiler compiles assignment operations.  Anything other
// than short declarations should use the compileAssign wrapper.
//
// There are three valid types of assignment:
// 1) T = T
//    Assigning a single expression with single-valued type to a
//    single-valued type.
// 2) MT = T, T, ...
//    Assigning multiple expressions with single-valued types to a
//    multi-valued type.
// 3) MT = MT
//    Assigning a single expression with multi-valued type to a
//    multi-valued type.
type assignCompiler struct {
	*compiler;
	pos token.Position;
	// The RHS expressions.  This may include nil's for
	// expressions that failed to compile.
	rs []*expr;
	// The (possibly unary) MultiType of the RHS.
	rmt *MultiType;
	// Whether this is an unpack assignment (case 3).
	isUnpack bool;
	// Whether map special assignment forms are allowed.
	allowMap bool;
	// Whether this is a "r, ok = a[x]" assignment.
	isMapUnpack bool;
	// The operation name to use in error messages, such as
	// "assignment" or "function call".
	errOp string;
	// The name to use for positions in error messages, such as
	// "argument".
	errPosName string;
}

// Type check the RHS of an assignment, returning a new assignCompiler
// and indicating if the type check succeeded.  This always returns an
// assignCompiler with rmt set, but if type checking fails, slots in
// the MultiType may be nil.  If rs contains nil's, type checking will
// fail and these expressions given a nil type.
func (a *compiler) checkAssign(pos token.Position, rs []*expr, errOp, errPosName string) (*assignCompiler, bool) {
	c := &assignCompiler{
		compiler: a,
		pos: pos,
		rs: rs,
		errOp: errOp,
		errPosName: errPosName,
	};

	// Is this an unpack?
	if len(rs) == 1 && rs[0] != nil {
		if rmt, isUnpack := rs[0].t.(*MultiType); isUnpack {
			c.rmt = rmt;
			c.isUnpack = true;
			return c, true;
		}
	}

	// Create MultiType for RHS and check that all RHS expressions
	// are single-valued.
	rts := make([]Type, len(rs));
	ok := true;
	for i, r := range rs {
		if r == nil {
			ok = false;
			continue;
		}

		if _, isMT := r.t.(*MultiType); isMT {
			r.diag("multi-valued expression not allowed in %s", errOp);
			ok = false;
			continue;
		}

		rts[i] = r.t;
	}

	c.rmt = NewMultiType(rts);
	return c, ok;
}

func (a *assignCompiler) allowMapForms(nls int) {
	a.allowMap = true;

	// Update unpacking info if this is r, ok = a[x]
	if nls == 2 && len(a.rs) == 1 && a.rs[0].evalMapValue != nil {
		a.isUnpack = true;
		a.rmt = NewMultiType([]Type {a.rs[0].t, BoolType});
		a.isMapUnpack = true;
	}
}

// compile type checks and compiles an assignment operation, returning
// a function that expects an l-value and the frame in which to
// evaluate the RHS expressions.  The l-value must have exactly the
// type given by lt.  Returns nil if type checking fails.
func (a *assignCompiler) compile(b *block, lt Type) (func(lv Value, f *Frame)) {
	lmt, isMT := lt.(*MultiType);
	rmt, isUnpack := a.rmt, a.isUnpack;

	// Create unary MultiType for single LHS
	if !isMT {
		lmt = NewMultiType([]Type{lt});
	}

	// Check that the assignment count matches
	lcount := len(lmt.Elems);
	rcount := len(rmt.Elems);
	if lcount != rcount {
		msg := "not enough";
		pos := a.pos;
		if rcount > lcount {
			msg = "too many";
			if lcount > 0 {
				pos = a.rs[lcount-1].pos;
			}
		}
		a.diagAt(&pos, "%s %ss for %s\n\t%s\n\t%s", msg, a.errPosName, a.errOp, lt, rmt);
		return nil;
	}

	bad := false;

	// If this is an unpack, create a temporary to store the
	// multi-value and replace the RHS with expressions to pull
	// out values from the temporary.  Technically, this is only
	// necessary when we need to perform assignment conversions.
	var effect func(f *Frame);
	if isUnpack {
		// This leaks a slot, but is definitely safe.
		temp := b.DefineSlot(a.rmt);
		tempIdx := temp.Index;
		if a.isMapUnpack {
			rf := a.rs[0].evalMapValue;
			vt := a.rmt.Elems[0];
			effect = func(f *Frame) {
				m, k := rf(f);
				v := m.Elem(k);
				found := boolV(true);
				if v == nil {
					found = boolV(false);
					v = vt.Zero();
				}
				f.Vars[tempIdx] = multiV([]Value {v, &found});
			};
		} else {
			rf := a.rs[0].asMulti();
			effect = func(f *Frame) {
				f.Vars[tempIdx] = multiV(rf(f));
			};
		}
		orig := a.rs[0];
		a.rs = make([]*expr, len(a.rmt.Elems));
		for i, t := range a.rmt.Elems {
			if t.isIdeal() {
				log.Crashf("Right side of unpack contains ideal: %s", rmt);
			}
			a.rs[i] = orig.newExpr(t, orig.desc);
			index := i;
			a.rs[i].genValue(func(f *Frame) Value { return f.Vars[tempIdx].(multiV)[index] });
		}
	}
	// Now len(a.rs) == len(a.rmt) and we've reduced any unpacking
	// to multi-assignment.

	// TODO(austin) Deal with assignment special cases.

	// Values of any type may always be assigned to variables of
	// compatible static type.
	for i, lt := range lmt.Elems {
		rt := rmt.Elems[i];

		// When [an ideal is] (used in an expression) assigned
		// to a variable or typed constant, the destination
		// must be able to represent the assigned value.
		if rt.isIdeal() {
			a.rs[i] = a.rs[i].convertTo(lmt.Elems[i]);
			if a.rs[i] == nil {
				bad = true;
				continue;
			}
			rt = a.rs[i].t;
		}

		// A pointer p to an array can be assigned to a slice
		// variable v with compatible element type if the type
		// of p or v is unnamed.
		if rpt, ok := rt.lit().(*PtrType); ok {
			if at, ok := rpt.Elem.lit().(*ArrayType); ok {
				if lst, ok := lt.lit().(*SliceType); ok {
					if lst.Elem.compat(at.Elem, false) && (rt.lit() == Type(rt) || lt.lit() == Type(lt)) {
						rf := a.rs[i].asPtr();
						a.rs[i] = a.rs[i].newExpr(lt, a.rs[i].desc);
						len := at.Len;
						a.rs[i].evalSlice = func(f *Frame) Slice {
							return Slice{rf(f).(ArrayValue), len, len};
						};
						rt = a.rs[i].t;
					}
				}
			}
		}

		if !lt.compat(rt, false) {
			if len(a.rs) == 1 {
				a.rs[0].diag("illegal operand types for %s\n\t%v\n\t%v", a.errOp, lt, rt);
			} else {
				a.rs[i].diag("illegal operand types in %s %d of %s\n\t%v\n\t%v", a.errPosName, i+1, a.errOp, lt, rt);
			}
			bad = true;
		}
	}
	if bad {
		return nil;
	}

	// Compile
	if !isMT {
		// Case 1
		return genAssign(lt, a.rs[0]);
	}
	// Case 2 or 3
	as := make([]func(lv Value, f *Frame), len(a.rs));
	for i, r := range a.rs {
		as[i] = genAssign(lmt.Elems[i], r);
	}
	return func(lv Value, f *Frame) {
		if effect != nil {
			effect(f);
		}
		lmv := lv.(multiV);
		for i, a := range as {
			a(lmv[i], f);
		}
	};
}

// compileAssign compiles an assignment operation without the full
// generality of an assignCompiler.  See assignCompiler for a
// description of the arguments.
func (a *compiler) compileAssign(pos token.Position, b *block, lt Type, rs []*expr, errOp, errPosName string) (func(lv Value, f *Frame)) {
	ac, ok := a.checkAssign(pos, rs, errOp, errPosName);
	if !ok {
		return nil;
	}
	return ac.compile(b, lt);
}

/*
 * Expression compiler
 */

// An exprCompiler stores information used throughout the compilation
// of a single expression.  It does not embed funcCompiler because
// expressions can appear at top level.
type exprCompiler struct {
	*compiler;
	// The block this expression is being compiled in.
	block *block;
	// Whether this expression is used in a constant context.
	constant bool;
}

func (a *exprCompiler) compile(x ast.Expr) *expr {
	ei := &exprInfo{a.compiler, x.Pos()};

	switch x := x.(type) {
	// Literals
	case *ast.CharLit:
		return ei.compileCharLit(string(x.Value));

	case *ast.CompositeLit:
		goto notimpl;

	case *ast.FloatLit:
		return ei.compileFloatLit(string(x.Value));

	case *ast.FuncLit:
		decl := ei.compileFuncType(a.block, x.Type);
		if decl == nil {
			// TODO(austin) Try compiling the body,
			// perhaps with dummy argument definitions
			return nil;
		}
		fn := ei.compileFunc(a.block, decl, x.Body);
		if fn == nil {
			return nil;
		}
		if a.constant {
			a.diagAt(x, "function literal used in constant expression");
			return nil;
		}
		return ei.compileFuncLit(decl, fn);

	case *ast.IntLit:
		return ei.compileIntLit(string(x.Value));

	case *ast.StringLit:
		return ei.compileStringLit(string(x.Value));

	// Types
	case *ast.ArrayType:
		goto notimpl;

	case *ast.ChanType:
		goto notimpl;

	case *ast.Ellipsis:
		goto notimpl;

	case *ast.FuncType:
		goto notimpl;

	case *ast.InterfaceType:
		goto notimpl;

	case *ast.MapType:
		goto notimpl;

	// Remaining expressions
	case *ast.BadExpr:
		// Error already reported by parser
		a.silentErrors++;
		return nil;

	case *ast.BinaryExpr:
		l, r := a.compile(x.X), a.compile(x.Y);
		if l == nil || r == nil {
			return nil;
		}
		return ei.compileBinaryExpr(x.Op, l, r);

	case *ast.CallExpr:
		l := a.compile(x.Fun);
		args := make([]*expr, len(x.Args));
		bad := false;
		for i, arg := range x.Args {
			args[i] = a.compile(arg);
			if args[i] == nil {
				bad = true;
			}
		}
		if l == nil || bad {
			return nil;
		}
		if a.constant {
			a.diagAt(x, "function call in constant context");
			return nil;
		}
		return ei.compileCallExpr(a.block, l, args);

	case *ast.Ident:
		return ei.compileIdent(a.block, a.constant, x.Value);

	case *ast.IndexExpr:
		if x.End != nil {
			a.diagAt(x, "slice expression not implemented");
			return nil;
		}
		l, r := a.compile(x.X), a.compile(x.Index);
		if l == nil || r == nil {
			return nil;
		}
		return ei.compileIndexExpr(l, r);

	case *ast.KeyValueExpr:
		goto notimpl;

	case *ast.ParenExpr:
		return a.compile(x.X);

	case *ast.SelectorExpr:
		v := a.compile(x.X);
		if v == nil {
			return nil;
		}
		return ei.compileSelectorExpr(v, x.Sel.Value);

	case *ast.StarExpr:
		v := a.compile(x.X);
		if v == nil {
			return nil;
		}
		return ei.compileStarExpr(v);

	case *ast.StringList:
		strings := make([]*expr, len(x.Strings));
		bad := false;
		for i, s := range x.Strings {
			strings[i] = a.compile(s);
			if strings[i] == nil {
				bad = true;
			}
		}
		if bad {
			return nil;
		}
		return ei.compileStringList(strings);

	case *ast.StructType:
		goto notimpl;

	case *ast.TypeAssertExpr:
		goto notimpl;

	case *ast.UnaryExpr:
		v := a.compile(x.X);
		if v == nil {
			return nil;
		}
		return ei.compileUnaryExpr(x.Op, v);
	}
	log.Crashf("unexpected ast node type %T", x);
	panic();

notimpl:
	a.diagAt(x, "%T expression node not implemented", x);
	return nil;
}

func (a *exprInfo) compileIdent(b *block, constant bool, name string) *expr {
	level, def := b.Lookup(name);
	if def == nil {
		a.diag("%s: undefined", name);
		return nil;
	}
	switch def := def.(type) {
	case *Constant:
		expr := a.newExpr(def.Type, "constant");
		expr.genConstant(def.Value);
		return expr;
	case *Variable:
		if constant {
			a.diag("variable %s used in constant expression", name);
			return nil;
		}
		return a.compileVariable(level, def);
	case Type:
		a.diag("type %v used as expression", name);
		return nil;
	}
	log.Crashf("name %s has unknown type %T", name, def);
	panic();
}

func (a *exprInfo) compileVariable(level int, v *Variable) *expr {
	if v.Type == nil {
		// Placeholder definition from an earlier error
		a.silentErrors++;
		return nil;
	}
	expr := a.newExpr(v.Type, "variable");
	expr.genIdentOp(level, v.Index);
	return expr;
}

func (a *exprInfo) compileIdealInt(i *bignum.Integer, desc string) *expr {
	expr := a.newExpr(IdealIntType, desc);
	expr.evalIdealInt = func() *bignum.Integer { return i };
	return expr;
}

func (a *exprInfo) compileIntLit(lit string) *expr {
	i, _, _2 := bignum.IntFromString(lit, 0);
	return a.compileIdealInt(i, "integer literal");
}

func (a *exprInfo) compileCharLit(lit string) *expr {
	if lit[0] != '\'' {
		log.Crashf("malformed character literal %s at %v passed parser", lit, a.pos);
	}
	v, mb, tail, err := strconv.UnquoteChar(lit[1:len(lit)], '\'');
	if err != nil || tail != "'" {
		log.Crashf("malformed character literal %s at %v passed parser", lit, a.pos);
	}
	return a.compileIdealInt(bignum.Int(int64(v)), "character literal");
}

func (a *exprInfo) compileFloatLit(lit string) *expr {
	f, _, n := bignum.RatFromString(lit, 0);
	if n != len(lit) {
		log.Crashf("malformed float literal %s at %v passed parser", lit, a.pos);
	}
	expr := a.newExpr(IdealFloatType, "float literal");
	expr.evalIdealFloat = func() *bignum.Rational { return f };
	return expr;
}

func (a *exprInfo) compileString(s string) *expr {
	// Ideal strings don't have a named type but they are
	// compatible with type string.

	// TODO(austin) Use unnamed string type.
	expr := a.newExpr(StringType, "string literal");
	expr.evalString = func(*Frame) string { return s };
	return expr;
}

func (a *exprInfo) compileStringLit(lit string) *expr {
	s, err := strconv.Unquote(lit);
	if err != nil {
		a.diag("illegal string literal, %v", err);
		return nil;
	}
	return a.compileString(s);
}

func (a *exprInfo) compileStringList(list []*expr) *expr {
	ss := make([]string, len(list));
	for i, s := range list {
		ss[i] = s.asString()(nil);
	}
	return a.compileString(strings.Join(ss, ""));
}

func (a *exprInfo) compileFuncLit(decl *FuncDecl, fn func(f *Frame) Func) *expr {
	expr := a.newExpr(decl.Type, "function literal");
	expr.evalFunc = fn;
	return expr;
}

func (a *exprInfo) compileSelectorExpr(v *expr, name string) *expr {
	// mark marks a field that matches the selector name.  It
	// tracks the best depth found so far and whether more than
	// one field has been found at that depth.
	bestDepth := -1;
	ambig := false;
	amberr := "";
	mark := func(depth int, pathName string) {
		switch {
		case bestDepth == -1 || depth < bestDepth:
			bestDepth = depth;
			ambig = false;
			amberr = "";

		case depth == bestDepth:
			ambig = true;

		default:
			log.Crashf("Marked field at depth %d, but already found one at depth %d", depth, bestDepth);
		}
		amberr += "\n\t" + pathName[1:len(pathName)];
	};

	visited := make(map[Type] bool);

	// find recursively searches for the named field, starting at
	// type t.  If it finds the named field, it returns a function
	// which takes an expr that represents a value of type 't' and
	// returns an expr that retrieves the named field.  We delay
	// expr construction to avoid producing lots of useless expr's
	// as we search.
	//
	// TODO(austin) Now that the expression compiler works on
	// semantic values instead of AST's, there should be a much
	// better way of doing this.
	var find func(Type, int, string) (func (*expr) *expr);
	find = func(t Type, depth int, pathName string) (func (*expr) *expr) {
		// Don't bother looking if we've found something shallower
		if bestDepth != -1 && bestDepth < depth {
			return nil;
		}

		// Don't check the same type twice and avoid loops
		if _, ok := visited[t]; ok {
			return nil;
		}
		visited[t] = true;

		// Implicit dereference
		deref := false;
		if ti, ok := t.(*PtrType); ok {
			deref = true;
			t = ti.Elem;
		}

		// If it's a named type, look for methods
		if ti, ok := t.(*NamedType); ok {
			method, ok := ti.methods[name];
			if ok {
				mark(depth, pathName + "." + name);
				log.Crash("Methods not implemented");
			}
			t = ti.def;
		}

		// If it's a struct type, check fields and embedded types
		var builder func(*expr) *expr;
		if t, ok := t.(*StructType); ok {
			for i, f := range t.Elems {
				var sub func(*expr) *expr;
				switch {
				case f.Name == name:
					mark(depth, pathName + "." + name);
					sub = func(e *expr) *expr { return e };

				case f.Anonymous:
					sub = find(f.Type, depth+1, pathName + "." + f.Name);
					if sub == nil {
						continue;
					}

				default:
					continue;
				}

				// We found something.  Create a
				// builder for accessing this field.
				ft := f.Type;
				index := i;
				builder = func(parent *expr) *expr {
					if deref {
						parent = a.compileStarExpr(parent);
					}
					expr := a.newExpr(ft, "selector expression");
					pf := parent.asStruct();
					evalAddr := func(f *Frame) Value {
						return pf(f).Field(index);
					};
					expr.genValue(evalAddr);
					return sub(expr);
				};
			}
		}

		return builder;
	};

	builder := find(v.t, 0, "");
	if builder == nil {
		a.diag("type %v has no field or method %s", v.t, name);
		return nil;
	}
	if ambig {
		a.diag("field %s is ambiguous in type %v%s", name, v.t, amberr);
		return nil;
	}

	return builder(v);
}

func (a *exprInfo) compileIndexExpr(l, r *expr) *expr {
	// Type check object
	if lt, ok := l.t.lit().(*PtrType); ok {
		if et, ok := lt.Elem.lit().(*ArrayType); ok {
			// Automatic dereference
			l = a.compileStarExpr(l);
			if l == nil {
				return nil;
			}
		}
	}

	var at Type;
	intIndex := false;
	var maxIndex int64 = -1;

	switch lt := l.t.lit().(type) {
	case *ArrayType:
		at = lt.Elem;
		intIndex = true;
		maxIndex = lt.Len;

	case *SliceType:
		at = lt.Elem;
		intIndex = true;

	case *stringType:
		at = Uint8Type;
		intIndex = true;

	case *MapType:
		at = lt.Elem;
		if r.t.isIdeal() {
			r = r.convertTo(lt.Key);
			if r == nil {
				return nil;
			}
		}
		if !lt.Key.compat(r.t, false) {
			a.diag("cannot use %s as index into %s", r.t, lt);
			return nil;
		}

	default:
		a.diag("cannot index into %v", l.t);
		return nil;
	}

	// Type check index and convert to int if necessary
	if intIndex {
		// XXX(Spec) It's unclear if ideal floats with no
		// fractional part are allowed here.  6g allows it.  I
		// believe that's wrong.
		switch _ := r.t.lit().(type) {
		case *idealIntType:
			val := r.asIdealInt()();
			if val.IsNeg() {
				a.diag("negative index: %s", val);
				return nil;
			}
			if maxIndex != -1 && val.Cmp(bignum.Int(maxIndex)) >= 0 {
				a.diag("index %s exceeds length %d", val, maxIndex);
				return nil;
			}
			r = r.convertTo(IntType);
			if r == nil {
				return nil;
			}

		case *uintType:
			// Convert to int
			nr := a.newExpr(IntType, r.desc);
			rf := r.asUint();
			nr.evalInt = func(f *Frame) int64 {
				return int64(rf(f));
			};
			r = nr;

		case *intType:
			// Good as is

		default:
			a.diag("illegal operand type for index\n\t%v", r.t);
			return nil;
		}
	}

	expr := a.newExpr(at, "index expression");

	// Compile
	switch lt := l.t.lit().(type) {
	case *ArrayType:
		lf := l.asArray();
		rf := r.asInt();
		bound := lt.Len;
		expr.genValue(func(f *Frame) Value {
			l, r := lf(f), rf(f);
			if r < 0 || r >= bound {
				Abort(IndexOutOfBounds{r, bound});
			}
			return l.Elem(r);
		});

	case *SliceType:
		lf := l.asSlice();
		rf := r.asInt();
		expr.genValue(func(f *Frame) Value {
			l, r := lf(f), rf(f);
			if l.Base == nil {
				Abort(NilPointer{});
			}
			if r < 0 || r >= l.Len {
				Abort(IndexOutOfBounds{r, l.Len});
			}
			return l.Base.Elem(r);
		});

	case *stringType:
		lf := l.asString();
		rf := r.asInt();
		// TODO(austin) This pulls over the whole string in a
		// remote setting, instead of just the one character.
		expr.evalUint = func(f *Frame) uint64 {
			l, r := lf(f), rf(f);
			if r < 0 || r >= int64(len(l)) {
				Abort(IndexOutOfBounds{r, int64(len(l))});
			}
			return uint64(l[r]);
		}

	case *MapType:
		lf := l.asMap();
		rf := r.asInterface();
		expr.genValue(func(f *Frame) Value {
			m := lf(f);
			k := rf(f);
			e := m.Elem(k);
			if e == nil {
				Abort(KeyNotFound{k});
			}
			return e;
		});
		// genValue makes things addressable, but map values
		// aren't addressable.
		expr.evalAddr = nil;
		expr.evalMapValue = func(f *Frame) (Map, interface{}) {
			// TODO(austin) Key check?
			return lf(f), rf(f);
		};

	default:
		log.Crashf("unexpected left operand type %T", l.t.lit());
	}

	return expr;
}

func (a *exprInfo) compileCallExpr(b *block, l *expr, as []*expr) *expr {
	// TODO(austin) Type conversions look like calls, but will
	// fail in DoIdent right now.
	//
	// TODO(austin) Magic built-in functions
	//
	// TODO(austin) Variadic functions.

	// Type check

	// XXX(Spec) Calling a named function type is okay.  I really
	// think there needs to be a general discussion of named
	// types.  A named type creates a new, distinct type, but the
	// type of that type is still whatever it's defined to.  Thus,
	// in "type Foo int", Foo is still an integer type and in
	// "type Foo func()", Foo is a function type.
	lt, ok := l.t.lit().(*FuncType);
	if !ok {
		a.diag("cannot call non-function type %v", l.t);
		return nil;
	}

	// The arguments must be single-valued expressions assignment
	// compatible with the parameters of F.
	//
	// XXX(Spec) The spec is wrong.  It can also be a single
	// multi-valued expression.
	nin := len(lt.In);
	assign := a.compileAssign(a.pos, b, NewMultiType(lt.In), as, "function call", "argument");
	if assign == nil {
		return nil;
	}

	var t Type;
	nout := len(lt.Out);
	switch nout {
	case 0:
		t = EmptyType;
	case 1:
		t = lt.Out[0];
	default:
		t = NewMultiType(lt.Out);
	}
	expr := a.newExpr(t, "function call");

	// Gather argument and out types to initialize frame variables
	vts := make([]Type, nin + nout);
	for i, t := range lt.In {
		vts[i] = t;
	}
	for i, t := range lt.Out {
		vts[i+nin] = t;
	}

	// Compile
	lf := l.asFunc();
	call := func(f *Frame) []Value {
		fun := lf(f);
		fr := fun.NewFrame();
		for i, t := range vts {
			fr.Vars[i] = t.Zero();
		}
		assign(multiV(fr.Vars[0:nin]), f);
		fun.Call(fr);
		return fr.Vars[nin:nin+nout];
	};
	expr.genFuncCall(call);

	return expr;
}

func (a *exprInfo) compileStarExpr(v *expr) *expr {
	switch vt := v.t.lit().(type) {
	case *PtrType:
		expr := a.newExpr(vt.Elem, "indirect expression");
		vf := v.asPtr();
		expr.genValue(func(f *Frame) Value {
			v := vf(f);
			if v == nil {
				Abort(NilPointer{});
			}
			return v;
		});
		return expr;
	}

	a.diagOpType(token.MUL, v.t);
	return nil;
}

var unaryOpDescs = make(map[token.Token] string)

func (a *exprInfo) compileUnaryExpr(op token.Token, v *expr) *expr {
	// Type check
	var t Type;
	switch op {
	case token.ADD, token.SUB:
		if !v.t.isInteger() && !v.t.isFloat() {
			a.diagOpType(op, v.t);
			return nil;
		}
		t = v.t;

	case token.NOT:
		if !v.t.isBoolean() {
			a.diagOpType(op, v.t);
			return nil;
		}
		t = BoolType;

	case token.XOR:
		if !v.t.isInteger() {
			a.diagOpType(op, v.t);
			return nil;
		}
		t = v.t;

	case token.AND:
		// The unary prefix address-of operator & generates
		// the address of its operand, which must be a
		// variable, pointer indirection, field selector, or
		// array or slice indexing operation.
		if v.evalAddr == nil {
			a.diag("cannot take the address of %s", v.desc);
			return nil;
		}

		// TODO(austin) Implement "It is illegal to take the
		// address of a function result variable" once I have
		// function result variables.

		t = NewPtrType(v.t);

	case token.ARROW:
		log.Crashf("Unary op %v not implemented", op);

	default:
		log.Crashf("unknown unary operator %v", op);
	}

	desc, ok := unaryOpDescs[op];
 	if !ok {
		desc = "unary " + op.String() + " expression";
		unaryOpDescs[op] = desc;
	}

	// Compile
	expr := a.newExpr(t, desc);
	switch op {
	case token.ADD:
		// Just compile it out
		expr = v;
		expr.desc = desc;

	case token.SUB:
		expr.genUnaryOpNeg(v);

	case token.NOT:
		expr.genUnaryOpNot(v);

	case token.XOR:
		expr.genUnaryOpXor(v);

	case token.AND:
		vf := v.evalAddr;
		expr.evalPtr = func(f *Frame) Value { return vf(f) };

	default:
		log.Crashf("Compilation of unary op %v not implemented", op);
	}

	return expr;
}

var binOpDescs = make(map[token.Token] string)

func (a *exprInfo) compileBinaryExpr(op token.Token, l, r *expr) *expr {
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
			return nil;
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
				return nil;
			}
		}
	}

	// Useful type predicates
	// TODO(austin) CL 33668 mandates identical types except for comparisons.
	compat := func() bool {
		return l.t.compat(r.t, false);
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
	var t Type;
	switch op {
	case token.ADD:
		if !compat() || (!integers() && !floats() && !strings()) {
			a.diagOpTypes(op, origlt, origrt);
			return nil;
		}
		t = l.t;

	case token.SUB, token.MUL, token.QUO:
		if !compat() || (!integers() && !floats()) {
			a.diagOpTypes(op, origlt, origrt);
			return nil;
		}
		t = l.t;

	case token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
		if !compat() || !integers() {
			a.diagOpTypes(op, origlt, origrt);
			return nil;
		}
		t = l.t;

	case token.SHL, token.SHR:
		// XXX(Spec) Is it okay for the right operand to be an
		// ideal float with no fractional part?  "The right
		// operand in a shift operation must be always be of
		// unsigned integer type or an ideal number that can
		// be safely converted into an unsigned integer type
		// (§Arithmetic operators)" suggests so and 6g agrees.

		if !l.t.isInteger() || !(r.t.isInteger() || r.t.isIdeal()) {
			a.diagOpTypes(op, origlt, origrt);
			return nil;
		}

		// The right operand in a shift operation must be
		// always be of unsigned integer type or an ideal
		// number that can be safely converted into an
		// unsigned integer type.
		if r.t.isIdeal() {
			r2 := r.convertTo(UintType);
			if r2 == nil {
				return nil;
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
		} else if _, ok := r.t.lit().(*uintType); !ok {
			a.diag("right operand of shift must be unsigned");
			return nil;
		}

		if l.t.isIdeal() && !r.t.isIdeal() {
			// XXX(Spec) What is the meaning of "ideal >>
			// non-ideal"?  Russ says the ideal should be
			// converted to an int.  6g propagates the
			// type down from assignments as a hint.

			l = l.convertTo(IntType);
			if l == nil {
				return nil;
			}
		}

		// At this point, we should have one of three cases:
		// 1) uint SHIFT uint
		// 2) int SHIFT uint
		// 3) ideal int SHIFT ideal int

		t = l.t;

	case token.LOR, token.LAND:
		if !booleans() {
			return nil;
		}
		// XXX(Spec) There's no mention of *which* boolean
		// type the logical operators return.  From poking at
		// 6g, it appears to be the named boolean type, NOT
		// the type of the left operand, and NOT an unnamed
		// boolean type.

		t = BoolType;

	case token.ARROW:
		// The operands in channel sends differ in type: one
		// is always a channel and the other is a variable or
		// value of the channel's element type.
		log.Crash("Binary op <- not implemented");
		t = BoolType;

	case token.LSS, token.GTR, token.LEQ, token.GEQ:
		// XXX(Spec) It's really unclear what types which
		// comparison operators apply to.  I feel like the
		// text is trying to paint a Venn diagram for me,
		// which it's really pretty simple: <, <=, >, >= apply
		// only to numeric types and strings.  == and != apply
		// to everything except arrays and structs, and there
		// are some restrictions on when it applies to slices.

		if !compat() || (!integers() && !floats() && !strings()) {
			a.diagOpTypes(op, origlt, origrt);
			return nil;
		}
		t = BoolType;

	case token.EQL, token.NEQ:
		// XXX(Spec) The rules for type checking comparison
		// operators are spread across three places that all
		// partially overlap with each other: the Comparison
		// Compatibility section, the Operators section, and
		// the Comparison Operators section.  The Operators
		// section should just say that operators require
		// identical types (as it does currently) except that
		// there a few special cases for comparison, which are
		// described in section X.  Currently it includes just
		// one of the four special cases.  The Comparison
		// Compatibility section and the Comparison Operators
		// section should either be merged, or at least the
		// Comparison Compatibility section should be
		// exclusively about type checking and the Comparison
		// Operators section should be exclusively about
		// semantics.

		// XXX(Spec) Comparison operators: "All comparison
		// operators apply to basic types except bools."  This
		// is very difficult to parse.  It's explained much
		// better in the Comparison Compatibility section.

		// XXX(Spec) Comparison compatibility: "Function
		// values are equal if they refer to the same
		// function." is rather vague.  It should probably be
		// similar to the way the rule for map values is
		// written: Function values are equal if they were
		// created by the same execution of a function literal
		// or refer to the same function declaration.  This is
		// *almost* but not quite waht 6g implements.  If a
		// function literals does not capture any variables,
		// then multiple executions of it will result in the
		// same closure.  Russ says he'll change that.

		// TODO(austin) Deal with remaining special cases

		if !compat() {
			a.diagOpTypes(op, origlt, origrt);
			return nil;
		}
		// Arrays and structs may not be compared to anything.
		// TODO(austin) Use a multi-type switch
		if _, ok := l.t.(*ArrayType); ok {
			a.diagOpTypes(op, origlt, origrt);
			return nil;
		}
		if _, ok := l.t.(*StructType); ok {
			a.diagOpTypes(op, origlt, origrt);
			return nil;
		}
		t = BoolType;

	default:
		log.Crashf("unknown binary operator %v", op);
	}

	desc, ok := binOpDescs[op];
	if !ok {
		desc = op.String() + " expression";
		binOpDescs[op] = desc;
	}

	// Check for ideal divide by zero
	switch op {
	case token.QUO, token.REM:
		if r.t.isIdeal() {
			if (r.t.isInteger() && r.asIdealInt()().IsZero()) ||
				(r.t.isFloat() && r.asIdealFloat()().IsZero()) {
				a.diag("divide by zero");
				return nil;
			}
		}
	}

	// Compile
	expr := a.newExpr(t, desc);
	switch op {
	case token.ADD:
		expr.genBinOpAdd(l, r);

	case token.SUB:
		expr.genBinOpSub(l, r);

	case token.MUL:
		expr.genBinOpMul(l, r);

	case token.QUO:
		// TODO(austin) Clear higher bits that may have
		// accumulated in our temporary.
		expr.genBinOpQuo(l, r);

	case token.REM:
		// TODO(austin) Clear higher bits that may have
		// accumulated in our temporary.
		expr.genBinOpRem(l, r);

	case token.AND:
		expr.genBinOpAnd(l, r);

	case token.OR:
		expr.genBinOpOr(l, r);

	case token.XOR:
		expr.genBinOpXor(l, r);

	case token.AND_NOT:
		expr.genBinOpAndNot(l, r);

	case token.SHL:
		if l.t.isIdeal() {
			lv := l.asIdealInt()();
			rv := r.asIdealInt()();
			const maxShift = 99999;
			if rv.Cmp(bignum.Int(maxShift)) > 0 {
				a.diag("left shift by %v; exceeds implementation limit of %v", rv, maxShift);
				expr.t = nil;
				return nil;
			}
			val := lv.Shl(uint(rv.Value()));
			expr.evalIdealInt = func() *bignum.Integer { return val };
		} else {
			expr.genBinOpShl(l, r);
		}

	case token.SHR:
		if l.t.isIdeal() {
			lv := l.asIdealInt()();
			rv := r.asIdealInt()();
			val := lv.Shr(uint(rv.Value()));
			expr.evalIdealInt = func() *bignum.Integer { return val };
		} else {
			expr.genBinOpShr(l, r);
		}

	case token.LSS:
		expr.genBinOpLss(l, r);

	case token.GTR:
		expr.genBinOpGtr(l, r);

	case token.LEQ:
		expr.genBinOpLeq(l, r);

	case token.GEQ:
		expr.genBinOpGeq(l, r);

	case token.EQL:
		expr.genBinOpEql(l, r);

	case token.NEQ:
		expr.genBinOpNeq(l, r);

	default:
		log.Crashf("Compilation of binary op %v not implemented", op);
	}

	return expr;
}

// TODO(austin) This is a hack to eliminate a circular dependency
// between type.go and expr.go
func (a *compiler) compileArrayLen(b *block, expr ast.Expr) (int64, bool) {
	lenExpr := a.compileExpr(b, true, expr);
	if lenExpr == nil {
		return 0, false;
	}

	// XXX(Spec) Are ideal floats with no fractional part okay?
	if lenExpr.t.isIdeal() {
		lenExpr = lenExpr.convertTo(IntType);
		if lenExpr == nil {
			return 0, false;
		}
	}

	if !lenExpr.t.isInteger() {
		a.diagAt(expr, "array size must be an integer");
		return 0, false;
	}

	switch _ := lenExpr.t.lit().(type) {
	case *intType:
		return lenExpr.evalInt(nil), true;
	case *uintType:
		return int64(lenExpr.evalUint(nil)), true;
	}
	log.Crashf("unexpected integer type %T", lenExpr.t);
	return 0, false;
}

func (a *compiler) compileExpr(b *block, constant bool, expr ast.Expr) *expr {
	ec := &exprCompiler{a, b, constant};
	nerr := a.numError();
	e := ec.compile(expr);
	if e == nil && nerr == a.numError() {
		log.Crashf("expression compilation failed without reporting errors");
	}
	return e;
}

// extractEffect separates out any effects that the expression may
// have, returning a function that will perform those effects and a
// new exprCompiler that is guaranteed to be side-effect free.  These
// are the moral equivalents of "temp := expr" and "temp" (or "temp :=
// &expr" and "*temp" for addressable exprs).  Because this creates a
// temporary variable, the caller should create a temporary block for
// the compilation of this expression and the evaluation of the
// results.
func (a *expr) extractEffect(b *block, errOp string) (func(f *Frame), *expr) {
	// Create "&a" if a is addressable
	rhs := a;
	if a.evalAddr != nil {
		rhs = a.compileUnaryExpr(token.AND, rhs);
	}

	// Create temp
	ac, ok := a.checkAssign(a.pos, []*expr{rhs}, errOp, "");
	if !ok {
		return nil, nil;
	}
	if len(ac.rmt.Elems) != 1 {
		a.diag("multi-valued expression not allowed in %s", errOp);
		return nil, nil;
	}
	tempType := ac.rmt.Elems[0];
	if tempType.isIdeal() {
		// It's too bad we have to duplicate this rule.
		switch {
		case tempType.isInteger():
			tempType = IntType;
		case tempType.isFloat():
			tempType = FloatType;
		default:
			log.Crashf("unexpected ideal type %v", tempType);
		}
	}
	temp := b.DefineSlot(tempType);
	tempIdx := temp.Index;

	// Create "temp := rhs"
	assign := ac.compile(b, tempType);
	if assign == nil {
		log.Crashf("compileAssign type check failed");
	}

	effect := func(f *Frame) {
		tempVal := tempType.Zero();
		f.Vars[tempIdx] = tempVal;
		assign(tempVal, f);
	};

	// Generate "temp" or "*temp"
	getTemp := a.compileVariable(0, temp);
	if a.evalAddr == nil {
		return effect, getTemp;
	}

	deref := a.compileStarExpr(getTemp);
	if deref == nil {
		return nil, nil;
	}
	return effect, deref;
}

/*
 * Testing interface
 */

type Expr struct {
	t Type;
	f func(f *Frame, out Value);
}

func (expr *Expr) Eval(f *Frame) (Value, os.Error) {
	v := expr.t.Zero();
	err := Try(func() {expr.f(f, v)});
	return v, err;
}

func CompileExpr(scope *Scope, expr ast.Expr) (*Expr, os.Error) {
	errors := scanner.NewErrorVector();
	cc := &compiler{errors, 0, 0};

	ec := cc.compileExpr(scope.block, false, expr);
	if ec == nil {
		return nil, errors.GetError(scanner.Sorted);
	}
	switch t := ec.t.lit().(type) {
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
	case *ArrayType:
		return &Expr{t, func(f *Frame, out Value) { out.(ArrayValue).Assign(ec.evalArray(f)) }}, nil;
	case *PtrType:
		return &Expr{t, func(f *Frame, out Value) { out.(PtrValue).Set(ec.evalPtr(f)) }}, nil;
	case *FuncType:
		return &Expr{t, func(f *Frame, out Value) { out.(FuncValue).Set(ec.evalFunc(f)) }}, nil;
	case *SliceType:
		return &Expr{t, func(f *Frame, out Value) { out.(SliceValue).Set(ec.evalSlice(f)) }}, nil;
	}
	log.Crashf("unexpected type %v", ec.t);
	panic();
}

/*
 * Operator generators
 * Everything below here is MACHINE GENERATED by gen.py genOps
 */

func (a *expr) genConstant(v Value) {
	switch _ := a.t.lit().(type) {
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
	case *StructType:
		val := v.(StructValue).Get();
		a.evalStruct = func(f *Frame) StructValue { return val };
	case *PtrType:
		val := v.(PtrValue).Get();
		a.evalPtr = func(f *Frame) Value { return val };
	case *FuncType:
		val := v.(FuncValue).Get();
		a.evalFunc = func(f *Frame) Func { return val };
	case *SliceType:
		val := v.(SliceValue).Get();
		a.evalSlice = func(f *Frame) Slice { return val };
	case *MapType:
		val := v.(MapValue).Get();
		a.evalMap = func(f *Frame) Map { return val };
	default:
		log.Crashf("unexpected constant type %v at %v", a.t, a.pos);
	}
}

func (a *expr) genIdentOp(level int, index int) {
	a.evalAddr = func(f *Frame) Value { return f.Get(level, index) };
	switch _ := a.t.lit().(type) {
	case *boolType:
		a.evalBool = func(f *Frame) bool { return f.Get(level, index).(BoolValue).Get() };
	case *uintType:
		a.evalUint = func(f *Frame) uint64 { return f.Get(level, index).(UintValue).Get() };
	case *intType:
		a.evalInt = func(f *Frame) int64 { return f.Get(level, index).(IntValue).Get() };
	case *floatType:
		a.evalFloat = func(f *Frame) float64 { return f.Get(level, index).(FloatValue).Get() };
	case *stringType:
		a.evalString = func(f *Frame) string { return f.Get(level, index).(StringValue).Get() };
	case *ArrayType:
		a.evalArray = func(f *Frame) ArrayValue { return f.Get(level, index).(ArrayValue).Get() };
	case *StructType:
		a.evalStruct = func(f *Frame) StructValue { return f.Get(level, index).(StructValue).Get() };
	case *PtrType:
		a.evalPtr = func(f *Frame) Value { return f.Get(level, index).(PtrValue).Get() };
	case *FuncType:
		a.evalFunc = func(f *Frame) Func { return f.Get(level, index).(FuncValue).Get() };
	case *SliceType:
		a.evalSlice = func(f *Frame) Slice { return f.Get(level, index).(SliceValue).Get() };
	case *MapType:
		a.evalMap = func(f *Frame) Map { return f.Get(level, index).(MapValue).Get() };
	default:
		log.Crashf("unexpected identifier type %v at %v", a.t, a.pos);
	}
}

func (a *expr) genFuncCall(call func(f *Frame) []Value) {
	a.exec = func(f *Frame) { call(f) };
	switch _ := a.t.lit().(type) {
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
	case *StructType:
		a.evalStruct = func(f *Frame) StructValue { return call(f)[0].(StructValue).Get() };
	case *PtrType:
		a.evalPtr = func(f *Frame) Value { return call(f)[0].(PtrValue).Get() };
	case *FuncType:
		a.evalFunc = func(f *Frame) Func { return call(f)[0].(FuncValue).Get() };
	case *SliceType:
		a.evalSlice = func(f *Frame) Slice { return call(f)[0].(SliceValue).Get() };
	case *MapType:
		a.evalMap = func(f *Frame) Map { return call(f)[0].(MapValue).Get() };
	case *MultiType:
		a.evalMulti = func(f *Frame) []Value { return call(f) };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *expr) genValue(vf func(*Frame) Value) {
	a.evalAddr = vf;
	switch _ := a.t.lit().(type) {
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
	case *StructType:
		a.evalStruct = func(f *Frame) StructValue { return vf(f).(StructValue).Get() };
	case *PtrType:
		a.evalPtr = func(f *Frame) Value { return vf(f).(PtrValue).Get() };
	case *FuncType:
		a.evalFunc = func(f *Frame) Func { return vf(f).(FuncValue).Get() };
	case *SliceType:
		a.evalSlice = func(f *Frame) Slice { return vf(f).(SliceValue).Get() };
	case *MapType:
		a.evalMap = func(f *Frame) Map { return vf(f).(MapValue).Get() };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *expr) genUnaryOpNeg(v *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genUnaryOpNot(v *expr) {
	switch _ := a.t.lit().(type) {
	case *boolType:
		vf := v.asBool();
		a.evalBool = func(f *Frame) bool { return !vf(f) };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *expr) genUnaryOpXor(v *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpAdd(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpSub(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpMul(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpQuo(l, r *expr) {
	switch _ := a.t.lit().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { l, r := lf(f), rf(f); if r == 0 { Abort(DivByZero{}) }; return l / r };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { l, r := lf(f), rf(f); if r == 0 { Abort(DivByZero{}) }; return l / r };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Quo(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalFloat = func(f *Frame) float64 { l, r := lf(f), rf(f); if r == 0 { Abort(DivByZero{}) }; return l / r };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Quo(rf());
		a.evalIdealFloat = func() *bignum.Rational { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *expr) genBinOpRem(l, r *expr) {
	switch _ := a.t.lit().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalUint = func(f *Frame) uint64 { l, r := lf(f), rf(f); if r == 0 { Abort(DivByZero{}) }; return l % r };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalInt = func(f *Frame) int64 { l, r := lf(f), rf(f); if r == 0 { Abort(DivByZero{}) }; return l % r };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Rem(rf());
		a.evalIdealInt = func() *bignum.Integer { return val };
	default:
		log.Crashf("unexpected result type %v at %v", a.t, a.pos);
	}
}

func (a *expr) genBinOpAnd(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpOr(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpXor(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpAndNot(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpShl(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpShr(l, r *expr) {
	switch _ := a.t.lit().(type) {
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

func (a *expr) genBinOpLss(l, r *expr) {
	switch _ := l.t.lit().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalBool = func(f *Frame) bool { return lf(f) < rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalBool = func(f *Frame) bool { return lf(f) < rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Cmp(rf()) < 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalBool = func(f *Frame) bool { return lf(f) < rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Cmp(rf()) < 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *stringType:
		lf := l.asString();
		rf := r.asString();
		a.evalBool = func(f *Frame) bool { return lf(f) < rf(f) };
	default:
		log.Crashf("unexpected left operand type %v at %v", l.t, a.pos);
	}
}

func (a *expr) genBinOpGtr(l, r *expr) {
	switch _ := l.t.lit().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalBool = func(f *Frame) bool { return lf(f) > rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalBool = func(f *Frame) bool { return lf(f) > rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Cmp(rf()) > 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalBool = func(f *Frame) bool { return lf(f) > rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Cmp(rf()) > 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *stringType:
		lf := l.asString();
		rf := r.asString();
		a.evalBool = func(f *Frame) bool { return lf(f) > rf(f) };
	default:
		log.Crashf("unexpected left operand type %v at %v", l.t, a.pos);
	}
}

func (a *expr) genBinOpLeq(l, r *expr) {
	switch _ := l.t.lit().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalBool = func(f *Frame) bool { return lf(f) <= rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalBool = func(f *Frame) bool { return lf(f) <= rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Cmp(rf()) <= 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalBool = func(f *Frame) bool { return lf(f) <= rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Cmp(rf()) <= 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *stringType:
		lf := l.asString();
		rf := r.asString();
		a.evalBool = func(f *Frame) bool { return lf(f) <= rf(f) };
	default:
		log.Crashf("unexpected left operand type %v at %v", l.t, a.pos);
	}
}

func (a *expr) genBinOpGeq(l, r *expr) {
	switch _ := l.t.lit().(type) {
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalBool = func(f *Frame) bool { return lf(f) >= rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalBool = func(f *Frame) bool { return lf(f) >= rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Cmp(rf()) >= 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalBool = func(f *Frame) bool { return lf(f) >= rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Cmp(rf()) >= 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *stringType:
		lf := l.asString();
		rf := r.asString();
		a.evalBool = func(f *Frame) bool { return lf(f) >= rf(f) };
	default:
		log.Crashf("unexpected left operand type %v at %v", l.t, a.pos);
	}
}

func (a *expr) genBinOpEql(l, r *expr) {
	switch _ := l.t.lit().(type) {
	case *boolType:
		lf := l.asBool();
		rf := r.asBool();
		a.evalBool = func(f *Frame) bool { return lf(f) == rf(f) };
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalBool = func(f *Frame) bool { return lf(f) == rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalBool = func(f *Frame) bool { return lf(f) == rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Cmp(rf()) == 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalBool = func(f *Frame) bool { return lf(f) == rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Cmp(rf()) == 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *stringType:
		lf := l.asString();
		rf := r.asString();
		a.evalBool = func(f *Frame) bool { return lf(f) == rf(f) };
	case *PtrType:
		lf := l.asPtr();
		rf := r.asPtr();
		a.evalBool = func(f *Frame) bool { return lf(f) == rf(f) };
	case *FuncType:
		lf := l.asFunc();
		rf := r.asFunc();
		a.evalBool = func(f *Frame) bool { return lf(f) == rf(f) };
	case *MapType:
		lf := l.asMap();
		rf := r.asMap();
		a.evalBool = func(f *Frame) bool { return lf(f) == rf(f) };
	default:
		log.Crashf("unexpected left operand type %v at %v", l.t, a.pos);
	}
}

func (a *expr) genBinOpNeq(l, r *expr) {
	switch _ := l.t.lit().(type) {
	case *boolType:
		lf := l.asBool();
		rf := r.asBool();
		a.evalBool = func(f *Frame) bool { return lf(f) != rf(f) };
	case *uintType:
		lf := l.asUint();
		rf := r.asUint();
		a.evalBool = func(f *Frame) bool { return lf(f) != rf(f) };
	case *intType:
		lf := l.asInt();
		rf := r.asInt();
		a.evalBool = func(f *Frame) bool { return lf(f) != rf(f) };
	case *idealIntType:
		lf := l.asIdealInt();
		rf := r.asIdealInt();
		val := lf().Cmp(rf()) != 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *floatType:
		lf := l.asFloat();
		rf := r.asFloat();
		a.evalBool = func(f *Frame) bool { return lf(f) != rf(f) };
	case *idealFloatType:
		lf := l.asIdealFloat();
		rf := r.asIdealFloat();
		val := lf().Cmp(rf()) != 0;
		a.evalBool = func(f *Frame) bool { return val };
	case *stringType:
		lf := l.asString();
		rf := r.asString();
		a.evalBool = func(f *Frame) bool { return lf(f) != rf(f) };
	case *PtrType:
		lf := l.asPtr();
		rf := r.asPtr();
		a.evalBool = func(f *Frame) bool { return lf(f) != rf(f) };
	case *FuncType:
		lf := l.asFunc();
		rf := r.asFunc();
		a.evalBool = func(f *Frame) bool { return lf(f) != rf(f) };
	case *MapType:
		lf := l.asMap();
		rf := r.asMap();
		a.evalBool = func(f *Frame) bool { return lf(f) != rf(f) };
	default:
		log.Crashf("unexpected left operand type %v at %v", l.t, a.pos);
	}
}

func genAssign(lt Type, r *expr) (func(lv Value, f *Frame)) {
	switch _ := lt.lit().(type) {
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
	case *StructType:
		rf := r.asStruct();
		return func(lv Value, f *Frame) { lv.Assign(rf(f)) };
	case *PtrType:
		rf := r.asPtr();
		return func(lv Value, f *Frame) { lv.(PtrValue).Set(rf(f)) };
	case *FuncType:
		rf := r.asFunc();
		return func(lv Value, f *Frame) { lv.(FuncValue).Set(rf(f)) };
	case *SliceType:
		rf := r.asSlice();
		return func(lv Value, f *Frame) { lv.(SliceValue).Set(rf(f)) };
	case *MapType:
		rf := r.asMap();
		return func(lv Value, f *Frame) { lv.(MapValue).Set(rf(f)) };
	default:
		log.Crashf("unexpected left operand type %v at %v", lt, r.pos);
	}
	panic();
}
