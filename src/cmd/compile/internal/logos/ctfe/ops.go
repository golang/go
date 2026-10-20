// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

// This file implements M5-S2 expression evaluation and value
// operations (SPEC Sec 2.1, T-201): literals, operators with constant
// versus variable semantics, conversions with target widths, composite
// literals, indexing, slicing, builtins, and size accounting.

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
)

// evalExpr evaluates a single-valued expression.
func (e *evaluator) evalExpr(fr *frame, x syntax.Expr) (Value, error) {
	switch x := syntax.Unparen(x).(type) {
	case *syntax.BasicLit:
		return e.evalLit(x)
	case *syntax.Name:
		return e.evalName(fr, x)
	case *syntax.Operation:
		return e.evalOperation(fr, x)
	case *syntax.CallExpr:
		vs, multi, err := e.evalCallMulti(fr, x)
		if err != nil {
			return Value{}, err
		}
		if multi || len(vs) != 1 {
			return Value{}, e.fail(posOf(x), "multiple values in single-value context")
		}
		return vs[0], nil
	case *syntax.SelectorExpr:
		return e.evalSelector(fr, x)
	case *syntax.IndexExpr:
		return e.evalIndex(fr, x)
	case *syntax.SliceExpr:
		return e.evalSlice(fr, x)
	case *syntax.CompositeLit:
		return e.evalComposite(fr, x, nil)
	case *syntax.CoalesceExpr:
		return e.evalCoalesce(fr, x)
	case *syntax.AssertExpr:
		return Value{}, e.fail(posOf(x), "type assertion in const func not supported")
	case *syntax.FuncLit:
		return Value{}, e.impurity(fr, posOf(x), "function literal in const func")
	case *syntax.ParenExpr:
		return e.evalExpr(fr, x.X)
	}
	return Value{}, e.fail(posOf(x), "unsupported expression %T in const evaluation", x)
}

// evalLit evaluates a basic literal.
func (e *evaluator) evalLit(lit *syntax.BasicLit) (Value, error) {
	pos := posOf(lit)
	switch lit.Kind {
	case syntax.IntLit:
		s := strings.ReplaceAll(lit.Value, "_", "")
		n, ok := new(big.Int).SetString(s, 0)
		if !ok {
			return Value{}, e.fail(pos, "bad integer literal in const evaluation")
		}
		return intValue(n, UntypedInt(), true), nil
	case syntax.FloatLit:
		s := strings.ReplaceAll(lit.Value, "_", "")
		r, ok := new(big.Rat).SetString(s)
		if !ok {
			f, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return Value{}, e.fail(pos, "bad float literal in const evaluation")
			}
			r = new(big.Rat).SetFloat64(f)
		}
		return Value{typ: UntypedFloat(), constant: true, fval: r}, nil
	case syntax.RuneLit:
		r, _, _, err := strconv.UnquoteChar(lit.Value[1:len(lit.Value)-1], '\'')
		if err != nil {
			// Raw single byte.
			if len(lit.Value) >= 3 {
				r = rune(lit.Value[1])
			} else {
				return Value{}, e.fail(pos, "bad rune literal in const evaluation")
			}
		}
		return intValue(big.NewInt(int64(r)), UntypedRune(), true), nil
	case syntax.StringLit:
		s := lit.Value
		if strings.HasPrefix(s, "`") {
			s = s[1 : len(s)-1]
		} else {
			u, err := strconv.Unquote(s)
			if err != nil {
				return Value{}, e.fail(pos, "bad string literal in const evaluation")
			}
			s = u
		}
		return Value{typ: UntypedString(), constant: true, sval: s}, nil
	}
	return Value{}, e.fail(pos, "complex constants not supported")
}

// evalName resolves an identifier: frame bindings first, then the host.
func (e *evaluator) evalName(fr *frame, id *syntax.Name) (Value, error) {
	if id.Value == "_" {
		return Value{}, e.fail(posOf(id), "cannot use _ as value in const evaluation")
	}
	switch id.Value {
	case "nil":
		return Value{isNil: true, typ: &Type{Kind: KUntypedNil}}, nil
	case "true":
		return Value{typ: UntypedBool(), constant: true, bval: true}, nil
	case "false":
		return Value{typ: UntypedBool(), constant: true, bval: false}, nil
	}
	if v, ok := e.lookup(fr, id.Value); ok {
		return v, nil
	}
	ent := e.h.Resolve(id)
	switch ent.Kind {
	case EntConst:
		return ent.Const, nil
	case EntPackageVar:
		return Value{}, e.impurity(fr, posOf(id), "read of package variable %s in const func", ent.Name)
	case EntImport:
		return Value{}, e.impurity(fr, posOf(id), "use of imported %s in const func", ent.Name)
	case EntBuiltin, EntConstFunc, EntPlainFunc:
		return Value{}, e.fail(posOf(id), "function %s used as value in const evaluation", id.Value)
	}
	return Value{}, e.fail(posOf(id), "cannot resolve %s in const evaluation", id.Value)
}

// evalCoalesce evaluates `x ?? y`: a present map read yields its
// value, a missing key (or any evaluated left side that is not an
// absent read) falls back per presence.
func (e *evaluator) evalCoalesce(fr *frame, c *syntax.CoalesceExpr) (Value, error) {
	if ix, ok := syntax.Unparen(c.X).(*syntax.IndexExpr); ok {
		if _, isList := syntax.Unparen(ix.Index).(*syntax.ListExpr); !isList {
			base, err := e.evalExpr(fr, ix.X)
			if err != nil {
				return Value{}, err
			}
			if base.mval != nil {
				key, err := e.evalExpr(fr, ix.Index)
				if err != nil {
					return Value{}, err
				}
				v, present, err := e.mapLookup(base, key, posOf(ix))
				if err != nil {
					return Value{}, err
				}
				if present {
					return withVar(v), nil
				}
				return e.evalExpr(fr, c.Y)
			}
		}
	}
	v, err := e.evalExpr(fr, c.X)
	if err != nil {
		return Value{}, err
	}
	return v, nil
}

// evalOperation evaluates unary and binary operations.
func (e *evaluator) evalOperation(fr *frame, o *syntax.Operation) (Value, error) {
	if o.Y == nil {
		return e.evalUnary(fr, o)
	}
	// Short-circuit logical operators.
	if o.Op == syntax.AndAnd || o.Op == syntax.OrOr {
		x, err := e.evalExpr(fr, o.X)
		if err != nil {
			return Value{}, err
		}
		b, err := e.truth(x, posOf(o.X))
		if err != nil {
			return Value{}, err
		}
		if o.Op == syntax.AndAnd && !b {
			return Value{typ: UntypedBool(), constant: x.constant, bval: false}, nil
		}
		if o.Op == syntax.OrOr && b {
			return Value{typ: UntypedBool(), constant: x.constant, bval: true}, nil
		}
		y, err := e.evalExpr(fr, o.Y)
		if err != nil {
			return Value{}, err
		}
		by, err := e.truth(y, posOf(o.Y))
		if err != nil {
			return Value{}, err
		}
		return Value{typ: UntypedBool(), constant: x.constant && y.constant, bval: by}, nil
	}
	x, err := e.evalExpr(fr, o.X)
	if err != nil {
		return Value{}, err
	}
	y, err := e.evalExpr(fr, o.Y)
	if err != nil {
		return Value{}, err
	}
	return e.applyOp(o.Op, x, y, posOf(o))
}

func (e *evaluator) applyOp(op syntax.Operator, x, y Value, pos syntax.Pos) (Value, error) {
	switch op {
	case syntax.Eql, syntax.Neq:
		eq, err := e.equal(x, y, pos)
		if err != nil {
			return Value{}, err
		}
		if op == syntax.Neq {
			eq = !eq
		}
		return Value{typ: UntypedBool(), constant: x.constant && y.constant, bval: eq}, nil
	case syntax.Lss, syntax.Leq, syntax.Gtr, syntax.Geq:
		return e.compare(op, x, y, pos)
	case syntax.Add:
		return e.add(x, y, pos)
	case syntax.Sub:
		return e.sub(x, y, pos)
	case syntax.Mul:
		return e.mul(x, y, pos)
	case syntax.Div:
		return e.quo(x, y, pos)
	case syntax.Rem:
		return e.rem(x, y, pos)
	case syntax.And, syntax.Or, syntax.Xor, syntax.AndNot:
		return e.bitwise(op, x, y, pos)
	case syntax.Shl, syntax.Shr:
		return e.shift(op, x, y, pos)
	}
	return Value{}, e.fail(pos, "unsupported operator in const evaluation")
}

func (e *evaluator) evalUnary(fr *frame, o *syntax.Operation) (Value, error) {
	switch o.Op {
	case syntax.Mul:
		return Value{}, e.impurity(fr, posOf(o), "pointer dereference in const func")
	case syntax.And:
		return Value{}, e.impurity(fr, posOf(o), "address-of in const func")
	case syntax.Recv:
		return Value{}, e.impurity(fr, posOf(o), "channel receive in const func")
	}
	x, err := e.evalExpr(fr, o.X)
	if err != nil {
		return Value{}, err
	}
	pos := posOf(o)
	switch o.Op {
	case syntax.Add:
		return x, nil
	case syntax.Sub:
		return e.neg(x, pos)
	case syntax.Not:
		b, err := e.truth(x, pos)
		if err != nil {
			return Value{}, err
		}
		return Value{typ: x.typ, constant: x.constant, bval: !b}, nil
	case syntax.Xor:
		if !isInteger(x.typ.Kind) {
			return Value{}, e.fail(pos, "^ of non-integer in const evaluation")
		}
		n := new(big.Int).Not(x.ival)
		if x.typ.Kind == KUntypedInt || x.typ.Kind == KUntypedRune {
			return intValue(n, x.typ, x.constant), nil
		}
		return e.wrapInt(n, x.typ, x.constant, pos)
	}
	return Value{}, e.fail(pos, "unsupported unary operator in const evaluation")
}

func (e *evaluator) truth(v Value, pos syntax.Pos) (bool, error) {
	if v.typ == nil || (v.typ.Kind != KBool && v.typ.Kind != KUntypedBool) {
		return false, e.fail(pos, "non-boolean condition in const evaluation")
	}
	return v.bval, nil
}

// numPair prepares two numeric operands: untyped operands convert to
// the typed side; two typed operands must share a kind.
func (e *evaluator) numPair(x, y Value, pos syntax.Pos) (Value, Value, *Type, error) {
	xk, yk := kindOf(x), kindOf(y)
	switch {
	case isUntyped(xk) && isUntyped(yk):
		if isFloat(xk) || isFloat(yk) {
			return x, y, UntypedFloat(), nil
		}
		if xk == KUntypedRune && yk == KUntypedRune {
			return x, y, UntypedRune(), nil
		}
		return x, y, UntypedInt(), nil
	case isUntyped(xk):
		c, err := e.coerceUntyped(x, y.typ, pos)
		if err != nil {
			return Value{}, Value{}, nil, err
		}
		return c, y, y.typ, nil
	case isUntyped(yk):
		c, err := e.coerceUntyped(y, x.typ, pos)
		if err != nil {
			return Value{}, Value{}, nil, err
		}
		return x, c, x.typ, nil
	case xk == yk:
		return x, y, x.typ, nil
	}
	// Named types share kinds; identical kinds convert freely.
	if x.typ != nil && y.typ != nil && x.typ.Kind == y.typ.Kind {
		return x, y, x.typ, nil
	}
	return Value{}, Value{}, nil, e.fail(pos, "mismatched types in const evaluation")
}

func kindOf(v Value) Kind {
	if v.typ == nil {
		return KInvalid
	}
	return v.typ.Kind
}

// coerceUntyped converts an untyped operand to a typed target for
// arithmetic, with range checks for constant values.
func (e *evaluator) coerceUntyped(v Value, t *Type, pos syntax.Pos) (Value, error) {
	return e.convert(v, t, pos)
}

func (e *evaluator) add(x, y Value, pos syntax.Pos) (Value, error) {
	if isStringLike(kindOf(x)) || isStringLike(kindOf(y)) {
		if !isStringLike(kindOf(x)) || !isStringLike(kindOf(y)) {
			return Value{}, e.fail(pos, "mismatched types in const evaluation")
		}
		return Value{typ: &Type{Kind: KString}, constant: x.constant && y.constant, sval: x.sval + y.sval}, nil
	}
	x, y, t, err := e.numPair(x, y, pos)
	if err != nil {
		return Value{}, err
	}
	cst := x.constant && y.constant
	if isFloat(t.Kind) {
		return e.floatOp(syntax.Add, x, y, t, cst, pos)
	}
	n := new(big.Int).Add(x.ival, y.ival)
	return e.finishInt(n, t, cst, pos)
}

func (e *evaluator) sub(x, y Value, pos syntax.Pos) (Value, error) {
	x, y, t, err := e.numPair(x, y, pos)
	if err != nil {
		return Value{}, err
	}
	cst := x.constant && y.constant
	if isFloat(t.Kind) {
		return e.floatOp(syntax.Sub, x, y, t, cst, pos)
	}
	n := new(big.Int).Sub(x.ival, y.ival)
	return e.finishInt(n, t, cst, pos)
}

func (e *evaluator) mul(x, y Value, pos syntax.Pos) (Value, error) {
	x, y, t, err := e.numPair(x, y, pos)
	if err != nil {
		return Value{}, err
	}
	cst := x.constant && y.constant
	if isFloat(t.Kind) {
		return e.floatOp(syntax.Mul, x, y, t, cst, pos)
	}
	n := new(big.Int).Mul(x.ival, y.ival)
	return e.finishInt(n, t, cst, pos)
}

func (e *evaluator) quo(x, y Value, pos syntax.Pos) (Value, error) {
	x, y, t, err := e.numPair(x, y, pos)
	if err != nil {
		return Value{}, err
	}
	cst := x.constant && y.constant
	if isFloat(t.Kind) {
		if y.constant {
			if isRatZero(yFloat(y)) {
				return Value{}, e.fail(pos, "division by zero in const evaluation")
			}
			if cst {
				return Value{typ: t, constant: true, fval: new(big.Rat).Quo(xFloat(x), yFloat(y))}, nil
			}
		}
		xf, yf := toFloat64(x, t), toFloat64(y, t)
		return floatResult(t, xf/yf, false, pos)
	}
	if y.ival.Sign() == 0 {
		return Value{}, e.fail(pos, "integer divide by zero in const evaluation")
	}
	n := new(big.Int).Quo(x.ival, y.ival)
	return e.finishInt(n, t, cst, pos)
}

func (e *evaluator) rem(x, y Value, pos syntax.Pos) (Value, error) {
	x, y, t, err := e.numPair(x, y, pos)
	if err != nil {
		return Value{}, err
	}
	if !isInteger(t.Kind) {
		return Value{}, e.fail(pos, "%% of non-integers in const evaluation")
	}
	if y.ival.Sign() == 0 {
		return Value{}, e.fail(pos, "integer divide by zero in const evaluation")
	}
	n := new(big.Int).Rem(x.ival, y.ival)
	return e.finishInt(n, t, x.constant && y.constant, pos)
}

func (e *evaluator) bitwise(op syntax.Operator, x, y Value, pos syntax.Pos) (Value, error) {
	x, y, t, err := e.numPair(x, y, pos)
	if err != nil {
		return Value{}, err
	}
	if !isInteger(t.Kind) {
		return Value{}, e.fail(pos, "bitwise op on non-integers in const evaluation")
	}
	n := new(big.Int)
	switch op {
	case syntax.And:
		n.And(x.ival, y.ival)
	case syntax.Or:
		n.Or(x.ival, y.ival)
	case syntax.Xor:
		n.Xor(x.ival, y.ival)
	case syntax.AndNot:
		n.AndNot(x.ival, y.ival)
	}
	return e.finishInt(n, t, x.constant && y.constant, pos)
}

func (e *evaluator) shift(op syntax.Operator, x, y Value, pos syntax.Pos) (Value, error) {
	if !isInteger(kindOf(x)) || !isInteger(kindOf(y)) {
		return Value{}, e.fail(pos, "shift of non-integers in const evaluation")
	}
	if y.ival.Sign() < 0 {
		return Value{}, e.fail(pos, "negative shift count in const evaluation")
	}
	t := x.typ
	if isUntyped(kindOf(x)) && isUntyped(kindOf(y)) {
		t = UntypedInt()
	} else if isUntyped(kindOf(x)) {
		var err error
		x, err = e.coerceUntyped(x, y.typ, pos)
		if err != nil {
			return Value{}, err
		}
		t = y.typ
	}
	n := y.ival
	if !n.IsUint64() {
		// Huge counts shift everything out; model as zero/one-fill.
		if op == syntax.Shl {
			return e.finishInt(new(big.Int), t, x.constant && y.constant, pos)
		}
		if isSigned(t.Kind) && x.ival.Sign() < 0 {
			return e.finishInt(new(big.Int).SetInt64(-1), t, x.constant && y.constant, pos)
		}
		return e.finishInt(new(big.Int), t, x.constant && y.constant, pos)
	}
	r := new(big.Int)
	if op == syntax.Shl {
		r.Lsh(x.ival, uint(n.Uint64()))
	} else if isSigned(t.Kind) {
		r.Rsh(x.ival, uint(n.Uint64()))
	} else {
		r.Rsh(unsigned(x.ival, t, e.cfg.WordSize), uint(n.Uint64()))
	}
	return e.finishInt(r, t, x.constant && y.constant, pos)
}

// unsigned reinterprets a wrapped value's bits as unsigned for logical
// right shifts.
func unsigned(n *big.Int, t *Type, word int64) *big.Int {
	if n.Sign() >= 0 {
		return n
	}
	w := intWidth(t.Kind, word)
	m := new(big.Int).Lsh(big.NewInt(1), uint(w))
	return new(big.Int).Add(n, m)
}

func (e *evaluator) neg(x Value, pos syntax.Pos) (Value, error) {
	if isFloat(kindOf(x)) {
		if x.constant {
			return Value{typ: x.typ, constant: true, fval: new(big.Rat).Neg(xFloat(x))}, nil
		}
		return floatResult(x.typ, -toFloat64(x, x.typ), false, pos)
	}
	if !isInteger(kindOf(x)) {
		return Value{}, e.fail(pos, "negation of non-number in const evaluation")
	}
	return e.finishInt(new(big.Int).Neg(x.ival), x.typ, x.constant, pos)
}

// finishInt applies constant range checks or variable wrapping.
func (e *evaluator) finishInt(n *big.Int, t *Type, cst bool, pos syntax.Pos) (Value, error) {
	if t == nil || isUntyped(t.Kind) {
		return intValue(n, t, cst), nil
	}
	if cst {
		if !inRange(n, t, e.cfg.WordSize) {
			return Value{}, e.fail(pos, "constant %v overflows %s in const evaluation", n, typeName(t))
		}
		return intValue(n, t, true), nil
	}
	return e.wrapInt(n, t, false, pos)
}

// wrapInt reduces n modulo the type width (variable semantics).
func (e *evaluator) wrapInt(n *big.Int, t *Type, cst bool, pos syntax.Pos) (Value, error) {
	w := intWidth(t.Kind, e.cfg.WordSize)
	m := new(big.Int).Lsh(big.NewInt(1), uint(w))
	n = new(big.Int).Mod(n, m)
	if isSigned(t.Kind) {
		half := new(big.Int).Lsh(big.NewInt(1), uint(w-1))
		if n.Cmp(half) >= 0 {
			n.Sub(n, m)
		}
	}
	return intValue(n, t, cst), nil
}

// inRange reports whether n fits in t.
func inRange(n *big.Int, t *Type, word int64) bool {
	w := intWidth(t.Kind, word)
	if isSigned(t.Kind) {
		lo := new(big.Int).Neg(new(big.Int).Lsh(big.NewInt(1), uint(w-1)))
		hi := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(w-1)), big.NewInt(1))
		return n.Cmp(lo) >= 0 && n.Cmp(hi) <= 0
	}
	return n.Sign() >= 0 && n.BitLen() <= int(w)
}

// xFloat returns the exact float operand.
func xFloat(v Value) *big.Rat {
	if v.fval != nil {
		return v.fval
	}
	if v.typ != nil && v.typ.Kind == KFloat32 {
		return new(big.Rat).SetFloat64(float64(float32(v.f64)))
	}
	return new(big.Rat).SetFloat64(v.f64)
}

// yFloat mirrors xFloat for the second operand.
func yFloat(v Value) *big.Rat { return xFloat(v) }

func toFloat64(v Value, t *Type) float64 {
	if v.fval != nil {
		f, _ := v.fval.Float64()
		if t.Kind == KFloat32 {
			return float64(float32(f))
		}
		return f
	}
	if t.Kind == KFloat32 {
		return float64(float32(v.f64))
	}
	return v.f64
}

func isRatZero(r *big.Rat) bool { return r.Sign() == 0 }

// floatOp evaluates a float binary op.
func (e *evaluator) floatOp(op syntax.Operator, x, y Value, t *Type, cst bool, pos syntax.Pos) (Value, error) {
	if cst {
		var r *big.Rat
		switch op {
		case syntax.Add:
			r = new(big.Rat).Add(xFloat(x), yFloat(y))
		case syntax.Sub:
			r = new(big.Rat).Sub(xFloat(x), yFloat(y))
		case syntax.Mul:
			r = new(big.Rat).Mul(xFloat(x), yFloat(y))
		case syntax.Div:
			r = new(big.Rat).Quo(xFloat(x), yFloat(y))
		}
		if t.Kind == KFloat32 && !ratFitsFloat32(r) {
			return Value{}, e.fail(pos, "constant overflows float32 in const evaluation")
		}
		if t.Kind == KFloat64 && !ratFitsFloat64(r) {
			return Value{}, e.fail(pos, "constant overflows float64 in const evaluation")
		}
		return Value{typ: t, constant: true, fval: r}, nil
	}
	xf, yf := toFloat64(x, t), toFloat64(y, t)
	var r float64
	switch op {
	case syntax.Add:
		r = xf + yf
	case syntax.Sub:
		r = xf - yf
	case syntax.Mul:
		r = xf * yf
	case syntax.Div:
		r = xf / yf
	}
	return floatResult(t, r, false, pos)
}

func ratFitsFloat32(r *big.Rat) bool {
	f, _ := r.Float64()
	return !math.IsInf(float64(float32(f)), 0) || r.Sign() == 0
}

func ratFitsFloat64(r *big.Rat) bool {
	f, _ := r.Float64()
	return !math.IsInf(f, 0) || r.Sign() == 0
}

func floatResult(t *Type, f float64, cst bool, pos syntax.Pos) (Value, error) {
	if t.Kind == KFloat32 {
		f = float64(float32(f))
	}
	return Value{typ: t, constant: cst, f64: f}, nil
}

func (e *evaluator) compare(op syntax.Operator, x, y Value, pos syntax.Pos) (Value, error) {
	if isNumeric(kindOf(x)) && isNumeric(kindOf(y)) && (isFloat(kindOf(x)) || isFloat(kindOf(y))) {
		fx, fy, t, err := e.numPair(x, y, pos)
		if err != nil {
			return Value{}, err
		}
		var c int
		if fx.constant && fy.constant {
			c = xFloat(fx).Cmp(yFloat(fy))
		} else {
			ax, ay := toFloat64(fx, t), toFloat64(fy, t)
			switch {
			case ax < ay:
				c = -1
			case ax > ay:
				c = 1
			}
		}
		return Value{typ: UntypedBool(), constant: fx.constant && fy.constant, bval: cmpOp(op, c)}, nil
	}
	if isStringLike(kindOf(x)) && isStringLike(kindOf(y)) {
		return Value{typ: UntypedBool(), constant: x.constant && y.constant, bval: cmpOp(op, strings.Compare(x.sval, y.sval))}, nil
	}
	ix, iy, _, err := e.numPair(x, y, pos)
	if err != nil {
		return Value{}, err
	}
	return Value{typ: UntypedBool(), constant: ix.constant && iy.constant, bval: cmpOp(op, ix.ival.Cmp(iy.ival))}, nil
}

func cmpOp(op syntax.Operator, c int) bool {
	switch op {
	case syntax.Lss:
		return c < 0
	case syntax.Leq:
		return c <= 0
	case syntax.Gtr:
		return c > 0
	}
	return c >= 0
}

// equal implements == over const values after -0.0 canonicalization.
func (e *evaluator) equal(x, y Value, pos syntax.Pos) (bool, error) {
	x = e.canon(x)
	y = e.canon(y)
	if x.isNil || y.isNil {
		if !isNilable(kindOf(x)) || !isNilable(kindOf(y)) {
			return false, e.fail(pos, "mismatched nil comparison in const evaluation")
		}
		return x.isNil && y.isNil, nil
	}
	switch {
	case isInteger(kindOf(x)) && isInteger(kindOf(y)):
		return x.ival.Cmp(y.ival) == 0, nil
	case isFloat(kindOf(x)) && isFloat(kindOf(y)):
		return x.Float64() == y.Float64(), nil
	case isStringLike(kindOf(x)) && isStringLike(kindOf(y)):
		return x.sval == y.sval, nil
	case (kindOf(x) == KBool || kindOf(x) == KUntypedBool) && (kindOf(y) == KBool || kindOf(y) == KUntypedBool):
		return x.bval == y.bval, nil
	case kindOf(x) == KArray && kindOf(y) == KArray:
		if len(x.array) != len(y.array) {
			return false, e.fail(pos, "mismatched array lengths in const comparison")
		}
		for i := range x.array {
			eq, err := e.equal(x.array[i], y.array[i], pos)
			if err != nil || !eq {
				return eq, err
			}
		}
		return true, nil
	case kindOf(x) == KStruct && kindOf(y) == KStruct:
		if len(x.strct) != len(y.strct) {
			return false, nil
		}
		for i := range x.strct {
			eq, err := e.equal(x.strct[i], y.strct[i], pos)
			if err != nil || !eq {
				return eq, err
			}
		}
		return true, nil
	}
	return false, e.fail(pos, "values not comparable in const evaluation")
}

func isNilable(k Kind) bool {
	switch k {
	case KSlice, KMap, KUntypedNil:
		return true
	}
	return false
}

// canon canonicalizes -0.0 to +0.0 for key comparison (M5-S4 rule:
// equality is Go == after canonicalization).
func (e *evaluator) canon(v Value) Value {
	if v.typ != nil && v.typ.Kind == KFloat64 && !v.constant && v.f64 == 0 {
		v.f64 = 0
	}
	if v.typ != nil && v.typ.Kind == KStruct {
		out := slices.Clone(v.strct)
		for i := range out {
			out[i] = e.canon(out[i])
		}
		v.strct = out
	}
	if v.typ != nil && v.typ.Kind == KArray {
		out := slices.Clone(v.array)
		for i := range out {
			out[i] = e.canon(out[i])
		}
		v.array = out
	}
	return v
}

// floatBits returns the float64 bits of a float value.
func (v Value) floatBits() uint64 {
	if v.fval != nil {
		f, _ := v.fval.Float64()
		return math.Float64bits(f)
	}
	return math.Float64bits(v.f64)
}

func typeName(t *Type) string {
	if t == nil {
		return "unknown"
	}
	if t.Named != "" {
		return t.Named
	}
	switch t.Kind {
	case KBool:
		return "bool"
	case KInt:
		return "int"
	case KInt8:
		return "int8"
	case KInt16:
		return "int16"
	case KInt32:
		return "int32"
	case KInt64:
		return "int64"
	case KUint:
		return "uint"
	case KUint8:
		return "uint8"
	case KUint16:
		return "uint16"
	case KUint32:
		return "uint32"
	case KUint64:
		return "uint64"
	case KUintptr:
		return "uintptr"
	case KFloat32:
		return "float32"
	case KFloat64:
		return "float64"
	case KString:
		return "string"
	case KSlice:
		return "[]" + typeName(t.Elem)
	case KArray:
		return fmt.Sprintf("[%d]%s", t.Len, typeName(t.Elem))
	case KMap:
		return fmt.Sprintf("map[%s]%s", typeName(t.Key), typeName(t.Elem))
	}
	return "composite"
}
