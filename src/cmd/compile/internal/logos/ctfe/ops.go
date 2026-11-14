// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
)

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
	case syntax.ImagLit:
		s := strings.TrimSuffix(strings.ReplaceAll(lit.Value, "_", ""), "i")
		if !strings.ContainsAny(s, ".eEpPxXoObB") {
			s = strings.TrimLeft(s, "0")
			if s == "" {
				s = "0"
			}
		}
		r, ok := new(big.Rat).SetString(s)
		if !ok {
			f, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return Value{}, e.fail(pos, "bad imaginary literal in const evaluation")
			}
			r = new(big.Rat).SetFloat64(f)
		}
		parts := new([2]big.Rat)
		parts[1].Set(r)
		return Value{typ: UntypedComplex(), constant: true, cparts: parts}, nil
	case syntax.RuneLit:
		r, _, _, err := strconv.UnquoteChar(lit.Value[1:len(lit.Value)-1], '\'')
		if err != nil {
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
			s = strings.ReplaceAll(s, "\r", "")
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

func (e *evaluator) evalName(fr *frame, id *syntax.Name) (Value, error) {
	if id.Value == "_" {
		return Value{}, e.fail(posOf(id), "cannot use _ as value in const evaluation")
	}
	if v, ok := e.lookup(fr, id.Value); ok {
		return v, nil
	}
	ent := e.h.Resolve(id)
	switch ent.Kind {
	case EntConst:
		return e.ingest(ent.Const, posOf(id))
	case EntPackageVar:
		return Value{}, e.impurity(fr, posOf(id), "read of package variable %s in const func", ent.Name)
	case EntImport:
		return Value{}, e.impurity(fr, posOf(id), "use of imported %s in const func", ent.Name)
	case EntBuiltin, EntConstFunc, EntPlainFunc:
		return Value{}, e.fail(posOf(id), "function %s used as value in const evaluation", id.Value)
	}
	switch id.Value {
	case "nil":
		return Value{isNil: true, typ: &Type{Kind: KUntypedNil}}, nil
	case "true":
		return Value{typ: UntypedBool(), constant: true, bval: true}, nil
	case "false":
		return Value{typ: UntypedBool(), constant: true, bval: false}, nil
	}
	return Value{}, e.fail(posOf(id), "cannot resolve %s in const evaluation", id.Value)
}

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

func (e *evaluator) evalOperation(fr *frame, o *syntax.Operation) (Value, error) {
	if o.Y == nil {
		return e.evalUnary(fr, o)
	}
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

func (e *evaluator) numPair(x, y Value, pos syntax.Pos) (Value, Value, *Type, error) {
	xk, yk := kindOf(x), kindOf(y)
	switch {
	case isUntyped(xk) && isUntyped(yk):
		if isComplex(xk) || isComplex(yk) {
			return x, y, UntypedComplex(), nil
		}
		if isFloat(xk) || isFloat(yk) {
			return x, y, UntypedFloat(), nil
		}
		if xk == KUntypedRune && yk == KUntypedRune {
			return x, y, UntypedRune(), nil
		}
		return x, y, UntypedInt(), nil
	case isUntyped(xk):
		c, err := e.convert(x, y.typ, pos)
		if err != nil {
			return Value{}, Value{}, nil, err
		}
		return c, y, y.typ, nil
	case isUntyped(yk):
		c, err := e.convert(y, x.typ, pos)
		if err != nil {
			return Value{}, Value{}, nil, err
		}
		return x, c, x.typ, nil
	case xk == yk:
		return x, y, x.typ, nil
	}
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

func (e *evaluator) complexOp(op syntax.Operator, x, y Value, t *Type, cst bool, pos syntax.Pos) (Value, error) {
	if cst && t.Kind == KUntypedComplex {
		xr, xi := complexParts(x)
		yr, yi := complexParts(y)
		parts := new([2]big.Rat)
		re, im := &parts[0], &parts[1]
		switch op {
		case syntax.Add:
			re.Add(xr, yr)
			im.Add(xi, yi)
		case syntax.Sub:
			re.Sub(xr, yr)
			im.Sub(xi, yi)
		case syntax.Mul:
			var a, b big.Rat
			re.Sub(a.Mul(xr, yr), b.Mul(xi, yi))
			im.Add(a.Mul(xr, yi), b.Mul(xi, yr))
		case syntax.Div:
			if yr.Sign() == 0 && yi.Sign() == 0 {
				return Value{}, e.fail(pos, "division by zero in const evaluation")
			}
			var a, b, den big.Rat
			den.Add(a.Mul(yr, yr), b.Mul(yi, yi))
			re.Quo(re.Add(a.Mul(xr, yr), b.Mul(xi, yi)), &den)
			im.Quo(im.Sub(a.Mul(xi, yr), b.Mul(xr, yi)), &den)
		default:
			return Value{}, e.fail(pos, "unsupported complex operator in const evaluation")
		}
		return Value{typ: t, constant: true, cparts: parts}, nil
	}
	xc, yc := toComplex(x), toComplex(y)
	if op == syntax.Div && yc == 0 && cst {
		return Value{}, e.fail(pos, "division by zero in const evaluation")
	}
	var r complex128
	switch op {
	case syntax.Add:
		r = xc + yc
	case syntax.Sub:
		r = xc - yc
	case syntax.Mul:
		r = xc * yc
	case syntax.Div:
		r = xc / yc
	default:
		return Value{}, e.fail(pos, "unsupported complex operator in const evaluation")
	}
	if cst {
		if t.Kind == KComplex64 {
			if math.IsInf(float64(float32(real(r))), 0) || math.IsInf(float64(float32(imag(r))), 0) {
				return Value{}, e.fail(pos, "constant overflows complex64 in const evaluation")
			}
		} else if math.IsInf(real(r), 0) || math.IsInf(imag(r), 0) {
			return Value{}, e.fail(pos, "constant overflows %s in const evaluation", typeName(t))
		}
		return complexConst(t, r), nil
	}
	return complexDyn(t, r), nil
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
	if isComplex(t.Kind) {
		return e.complexOp(syntax.Add, x, y, t, cst, pos)
	}
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
	if isComplex(t.Kind) {
		return e.complexOp(syntax.Sub, x, y, t, cst, pos)
	}
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
	if isComplex(t.Kind) {
		return e.complexOp(syntax.Mul, x, y, t, cst, pos)
	}
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
	if isComplex(t.Kind) {
		return e.complexOp(syntax.Div, x, y, t, cst, pos)
	}
	if isFloat(t.Kind) {
		if y.constant {
			if xFloat(y).Sign() == 0 {
				return Value{}, e.fail(pos, "division by zero in const evaluation")
			}
			if cst {
				return Value{typ: t, constant: true, fval: new(big.Rat).Quo(xFloat(x), xFloat(y))}, nil
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
	xk, yk := kindOf(x), kindOf(y)
	t := x.typ
	if isUntyped(xk) {
		switch {
		case isUntyped(yk):
			t = UntypedInt()
		case y.constant:
			t = x.typ
		default:
			t = defaulted(x).typ
		}
	}
	cst := x.constant && y.constant
	n := y.ival
	if !n.IsUint64() || n.Uint64() > uint64(^uint(0)) {
		if op == syntax.Shl {
			if x.ival.Sign() == 0 {
				return e.finishInt(new(big.Int), t, cst, pos)
			}
			if !isUntyped(t.Kind) {
				if cst {
					return Value{}, e.fail(pos, "constant shift result overflows %s in const evaluation", typeName(t))
				}
				return e.finishInt(new(big.Int), t, cst, pos)
			}
			return Value{}, &Error{Code: CodeTooLarge, Detail: fmt.Sprintf("shift result over %d byte limit", e.cfg.MaxValueBytes), Pos: pos}
		}
		if (isSigned(t.Kind) || isUntyped(t.Kind)) && x.ival.Sign() < 0 {
			return e.finishInt(big.NewInt(-1), t, cst, pos)
		}
		return e.finishInt(new(big.Int), t, cst, pos)
	}
	n64 := n.Uint64()
	if op == syntax.Shl {
		if x.ival.Sign() == 0 {
			return e.finishInt(new(big.Int), t, cst, pos)
		}
		if !isUntyped(t.Kind) {
			if w := intWidth(t.Kind, e.cfg.WordSize); w > 0 && n64 >= uint64(w) {
				if cst {
					return Value{}, e.fail(pos, "constant shift result overflows %s in const evaluation", typeName(t))
				}
				return e.finishInt(new(big.Int), t, cst, pos)
			}
		} else if e.cfg.MaxValueBytes > 0 {
			limit := uint64(e.cfg.MaxValueBytes)
			xBits := uint64(x.ival.BitLen())
			sum := xBits + n64
			if sum < n64 {
				return Value{}, &Error{
					Code:   CodeTooLarge,
					Detail: fmt.Sprintf("shift result over %d byte limit", e.cfg.MaxValueBytes),
					Pos:    pos,
				}
			}
			bytes := sum / 8
			if sum%8 != 0 {
				bytes++
			}
			if bytes > limit {
				return Value{}, &Error{
					Code:   CodeTooLarge,
					Detail: fmt.Sprintf("shift result over %d byte limit", e.cfg.MaxValueBytes),
					Pos:    pos,
				}
			}
		}
		return e.finishInt(new(big.Int).Lsh(x.ival, uint(n64)), t, cst, pos)
	}
	if !isUntyped(t.Kind) {
		if w := intWidth(t.Kind, e.cfg.WordSize); w > 0 && n64 >= uint64(w) {
			if isSigned(t.Kind) && x.ival.Sign() < 0 {
				return e.finishInt(big.NewInt(-1), t, cst, pos)
			}
			return e.finishInt(new(big.Int), t, cst, pos)
		}
	}
	if isSigned(t.Kind) || isUntyped(t.Kind) {
		return e.finishInt(new(big.Int).Rsh(x.ival, uint(n64)), t, cst, pos)
	}
	return e.finishInt(new(big.Int).Rsh(unsigned(x.ival, t, e.cfg.WordSize), uint(n64)), t, cst, pos)
}

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
	if isComplex(kindOf(x)) {
		if x.constant && kindOf(x) == KUntypedComplex {
			re, im := complexParts(x)
			parts := new([2]big.Rat)
			parts[0].Neg(re)
			parts[1].Neg(im)
			return Value{typ: x.typ, constant: true, cparts: parts}, nil
		}
		c := x.Complex()
		if x.constant {
			return complexConst(x.typ, complex(-real(c), -imag(c))), nil
		}
		return complexDyn(x.typ, complex(-real(c), -imag(c))), nil
	}
	if !isInteger(kindOf(x)) {
		return Value{}, e.fail(pos, "negation of non-number in const evaluation")
	}
	return e.finishInt(new(big.Int).Neg(x.ival), x.typ, x.constant, pos)
}

func (e *evaluator) finishInt(n *big.Int, t *Type, cst bool, pos syntax.Pos) (Value, error) {
	if t == nil || isUntyped(t.Kind) {
		if cst && n.BitLen() > 512 {
			return Value{}, e.fail(pos, "constant overflows %s in const evaluation", typeName(t))
		}
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

func inRange(n *big.Int, t *Type, word int64) bool {
	w := intWidth(t.Kind, word)
	if isSigned(t.Kind) {
		lo := new(big.Int).Neg(new(big.Int).Lsh(big.NewInt(1), uint(w-1)))
		hi := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(w-1)), big.NewInt(1))
		return n.Cmp(lo) >= 0 && n.Cmp(hi) <= 0
	}
	return n.Sign() >= 0 && n.BitLen() <= int(w)
}

func xFloat(v Value) *big.Rat {
	if v.fval != nil {
		return v.fval
	}
	if v.ival != nil {
		return new(big.Rat).SetInt(v.ival)
	}
	if v.typ != nil && v.typ.Kind == KFloat32 {
		return new(big.Rat).SetFloat64(float64(float32(v.f64)))
	}
	return new(big.Rat).SetFloat64(v.f64)
}

func toFloat64(v Value, t *Type) float64 {
	if v.fval != nil {
		f, _ := v.fval.Float64()
		if t != nil && t.Kind == KFloat32 {
			return float64(float32(f))
		}
		return f
	}
	if v.ival != nil {
		f, _ := new(big.Rat).SetInt(v.ival).Float64()
		if t != nil && t.Kind == KFloat32 {
			return float64(float32(f))
		}
		return f
	}
	if t != nil && t.Kind == KFloat32 {
		return float64(float32(v.f64))
	}
	return v.f64
}

func toComplex(v Value) complex128 {
	if isComplex(kindOf(v)) {
		return v.Complex()
	}
	if isFloat(kindOf(v)) {
		return complex(v.Float64(), 0)
	}
	f, _ := new(big.Rat).SetInt(v.ival).Float64()
	return complex(f, 0)
}

func complexParts(v Value) (*big.Rat, *big.Rat) {
	if v.cparts != nil {
		return &v.cparts[0], &v.cparts[1]
	}
	if isComplex(kindOf(v)) {
		c := v.Complex()
		return new(big.Rat).SetFloat64(real(c)), new(big.Rat).SetFloat64(imag(c))
	}
	return xFloat(v), new(big.Rat)
}

func (e *evaluator) floatOp(op syntax.Operator, x, y Value, t *Type, cst bool, pos syntax.Pos) (Value, error) {
	if cst {
		var r *big.Rat
		switch op {
		case syntax.Add:
			r = new(big.Rat).Add(xFloat(x), xFloat(y))
		case syntax.Sub:
			r = new(big.Rat).Sub(xFloat(x), xFloat(y))
		case syntax.Mul:
			r = new(big.Rat).Mul(xFloat(x), xFloat(y))
		case syntax.Div:
			r = new(big.Rat).Quo(xFloat(x), xFloat(y))
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

func complexResult(t *Type, c complex128, cst bool) Value {
	if t.Kind == KComplex64 {
		c = complex(float64(float32(real(c))), float64(float32(imag(c))))
	}
	return Value{typ: t, constant: cst, cval: c}
}

func complexConst(t *Type, c complex128) Value { return complexResult(t, c, true) }

func complexDyn(t *Type, c complex128) Value { return complexResult(t, c, false) }

func (e *evaluator) compare(op syntax.Operator, x, y Value, pos syntax.Pos) (Value, error) {
	if isComplex(kindOf(x)) || isComplex(kindOf(y)) {
		return Value{}, e.fail(pos, "ordered comparison of complex in const evaluation")
	}
	if isNumeric(kindOf(x)) && isNumeric(kindOf(y)) && (isFloat(kindOf(x)) || isFloat(kindOf(y))) {
		fx, fy, t, err := e.numPair(x, y, pos)
		if err != nil {
			return Value{}, err
		}
		var c int
		if fx.constant && fy.constant {
			c = xFloat(fx).Cmp(xFloat(fy))
		} else {
			ax, ay := toFloat64(fx, t), toFloat64(fy, t)
			if math.IsNaN(ax) || math.IsNaN(ay) {
				return Value{typ: UntypedBool(), constant: false, bval: false}, nil
			}
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
	case isComplex(kindOf(x)) || isComplex(kindOf(y)):
		if !isNumeric(kindOf(x)) || !isNumeric(kindOf(y)) {
			return false, e.fail(pos, "values not comparable in const evaluation")
		}
		if isUntyped(kindOf(x)) != isUntyped(kindOf(y)) {
			cx, cy, _, err := e.numPair(x, y, pos)
			if err != nil {
				return false, err
			}
			x, y = cx, cy
		}
		if x.constant && y.constant && isUntyped(kindOf(x)) && isUntyped(kindOf(y)) {
			xr, xi := complexParts(x)
			yr, yi := complexParts(y)
			return xr.Cmp(yr) == 0 && xi.Cmp(yi) == 0, nil
		}
		return toComplex(x) == toComplex(y), nil
	case isInteger(kindOf(x)) && isInteger(kindOf(y)):
		return x.ival.Cmp(y.ival) == 0, nil
	case isNumeric(kindOf(x)) && isNumeric(kindOf(y)) && (isFloat(kindOf(x)) || isFloat(kindOf(y))):
		if isUntyped(kindOf(x)) != isUntyped(kindOf(y)) {
			cx, cy, _, err := e.numPair(x, y, pos)
			if err != nil {
				return false, err
			}
			x, y = cx, cy
		}
		if x.constant && y.constant {
			return xFloat(x).Cmp(xFloat(y)) == 0, nil
		}
		if isInteger(kindOf(x)) && !isFloat(kindOf(x)) {
			xf, yf, t, err := e.numPair(x, y, pos)
			if err != nil {
				return false, err
			}
			return toFloat64(xf, t) == toFloat64(yf, t), nil
		}
		if isInteger(kindOf(y)) && !isFloat(kindOf(y)) {
			xf, yf, t, err := e.numPair(x, y, pos)
			if err != nil {
				return false, err
			}
			return toFloat64(xf, t) == toFloat64(yf, t), nil
		}
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

func (e *evaluator) canon(v Value) Value {
	if v.typ != nil && v.typ.Kind == KFloat64 && !v.constant && v.f64 == 0 {
		v.f64 = 0
	}
	if v.typ != nil && isComplex(v.typ.Kind) && v.cparts == nil {
		if real(v.cval) == 0 {
			v.cval = complex(0, imag(v.cval))
		}
		if imag(v.cval) == 0 {
			v.cval = complex(real(v.cval), 0)
		}
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
	case KComplex64:
		return "complex64"
	case KComplex128:
		return "complex128"
	case KUntypedBool:
		return "untyped bool"
	case KUntypedInt:
		return "untyped int"
	case KUntypedRune:
		return "untyped rune"
	case KUntypedString:
		return "untyped string"
	case KUntypedFloat:
		return "untyped float"
	case KUntypedComplex:
		return "untyped complex"
	case KUntypedNil:
		return "untyped nil"
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
