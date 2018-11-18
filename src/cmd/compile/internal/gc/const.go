// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"math/big"
	"strings"
)

// Ctype describes the constant kind of an "ideal" (untyped) constant.
type Ctype uint8

const (
	CTxxx Ctype = iota

	CTINT
	CTRUNE
	CTFLT
	CTCPLX
	CTSTR
	CTBOOL
	CTNIL
)

type Val struct {
	// U contains one of:
	// bool     bool when n.ValCtype() == CTBOOL
	// *Mpint   int when n.ValCtype() == CTINT, rune when n.ValCtype() == CTRUNE
	// *Mpflt   float when n.ValCtype() == CTFLT
	// *Mpcplx  pair of floats when n.ValCtype() == CTCPLX
	// string   string when n.ValCtype() == CTSTR
	// *Nilval  when n.ValCtype() == CTNIL
	U interface{}
}

func (v Val) Ctype() Ctype {
	switch x := v.U.(type) {
	default:
		Fatalf("unexpected Ctype for %T", v.U)
		panic("unreachable")
	case nil:
		return 0
	case *NilVal:
		return CTNIL
	case bool:
		return CTBOOL
	case *Mpint:
		if x.Rune {
			return CTRUNE
		}
		return CTINT
	case *Mpflt:
		return CTFLT
	case *Mpcplx:
		return CTCPLX
	case string:
		return CTSTR
	}
}

func eqval(a, b Val) bool {
	if a.Ctype() != b.Ctype() {
		return false
	}
	switch x := a.U.(type) {
	default:
		Fatalf("unexpected Ctype for %T", a.U)
		panic("unreachable")
	case *NilVal:
		return true
	case bool:
		y := b.U.(bool)
		return x == y
	case *Mpint:
		y := b.U.(*Mpint)
		return x.Cmp(y) == 0
	case *Mpflt:
		y := b.U.(*Mpflt)
		return x.Cmp(y) == 0
	case *Mpcplx:
		y := b.U.(*Mpcplx)
		return x.Real.Cmp(&y.Real) == 0 && x.Imag.Cmp(&y.Imag) == 0
	case string:
		y := b.U.(string)
		return x == y
	}
}

// Interface returns the constant value stored in v as an interface{}.
// It returns int64s for ints and runes, float64s for floats,
// complex128s for complex values, and nil for constant nils.
func (v Val) Interface() interface{} {
	switch x := v.U.(type) {
	default:
		Fatalf("unexpected Interface for %T", v.U)
		panic("unreachable")
	case *NilVal:
		return nil
	case bool, string:
		return x
	case *Mpint:
		return x.Int64()
	case *Mpflt:
		return x.Float64()
	case *Mpcplx:
		return complex(x.Real.Float64(), x.Imag.Float64())
	}
}

type NilVal struct{}

// Int64 returns n as an int64.
// n must be an integer or rune constant.
func (n *Node) Int64() int64 {
	if !Isconst(n, CTINT) {
		Fatalf("Int64(%v)", n)
	}
	return n.Val().U.(*Mpint).Int64()
}

// CanInt64 reports whether it is safe to call Int64() on n.
func (n *Node) CanInt64() bool {
	if !Isconst(n, CTINT) {
		return false
	}

	// if the value inside n cannot be represented as an int64, the
	// return value of Int64 is undefined
	return n.Val().U.(*Mpint).CmpInt64(n.Int64()) == 0
}

// Bool returns n as a bool.
// n must be a boolean constant.
func (n *Node) Bool() bool {
	if !Isconst(n, CTBOOL) {
		Fatalf("Bool(%v)", n)
	}
	return n.Val().U.(bool)
}

// truncate float literal fv to 32-bit or 64-bit precision
// according to type; return truncated value.
func truncfltlit(oldv *Mpflt, t *types.Type) *Mpflt {
	if t == nil {
		return oldv
	}

	if overflow(Val{oldv}, t) {
		// If there was overflow, simply continuing would set the
		// value to Inf which in turn would lead to spurious follow-on
		// errors. Avoid this by returning the existing value.
		return oldv
	}

	fv := newMpflt()

	// convert large precision literal floating
	// into limited precision (float64 or float32)
	switch t.Etype {
	case types.TFLOAT32:
		fv.SetFloat64(oldv.Float32())
	case types.TFLOAT64:
		fv.SetFloat64(oldv.Float64())
	default:
		Fatalf("truncfltlit: unexpected Etype %v", t.Etype)
	}

	return fv
}

// truncate Real and Imag parts of Mpcplx to 32-bit or 64-bit
// precision, according to type; return truncated value. In case of
// overflow, calls yyerror but does not truncate the input value.
func trunccmplxlit(oldv *Mpcplx, t *types.Type) *Mpcplx {
	if t == nil {
		return oldv
	}

	if overflow(Val{oldv}, t) {
		// If there was overflow, simply continuing would set the
		// value to Inf which in turn would lead to spurious follow-on
		// errors. Avoid this by returning the existing value.
		return oldv
	}

	cv := newMpcmplx()

	switch t.Etype {
	case types.TCOMPLEX64:
		cv.Real.SetFloat64(oldv.Real.Float32())
		cv.Imag.SetFloat64(oldv.Imag.Float32())
	case types.TCOMPLEX128:
		cv.Real.SetFloat64(oldv.Real.Float64())
		cv.Imag.SetFloat64(oldv.Imag.Float64())
	default:
		Fatalf("trunccplxlit: unexpected Etype %v", t.Etype)
	}

	return cv
}

// canReuseNode indicates whether it is known to be safe
// to reuse a Node.
type canReuseNode bool

const (
	noReuse canReuseNode = false // not necessarily safe to reuse
	reuseOK canReuseNode = true  // safe to reuse
)

// convert n, if literal, to type t.
// implicit conversion.
// The result of convlit MUST be assigned back to n, e.g.
// 	n.Left = convlit(n.Left, t)
func convlit(n *Node, t *types.Type) *Node {
	return convlit1(n, t, false, noReuse)
}

// convlit1 converts n, if literal, to type t.
// It returns a new node if necessary.
// The result of convlit1 MUST be assigned back to n, e.g.
// 	n.Left = convlit1(n.Left, t, explicit, reuse)
func convlit1(n *Node, t *types.Type, explicit bool, reuse canReuseNode) *Node {
	if n == nil || t == nil || n.Type == nil || t.IsUntyped() || n.Type == t {
		return n
	}
	if !explicit && !n.Type.IsUntyped() {
		return n
	}

	if n.Op == OLITERAL && !reuse {
		// Can't always set n.Type directly on OLITERAL nodes.
		// See discussion on CL 20813.
		n = n.rawcopy()
		reuse = true
	}

	switch n.Op {
	default:
		if n.Type == types.Idealbool {
			if !t.IsBoolean() {
				t = types.Types[TBOOL]
			}
			switch n.Op {
			case ONOT:
				n.Left = convlit(n.Left, t)
			case OANDAND, OOROR:
				n.Left = convlit(n.Left, t)
				n.Right = convlit(n.Right, t)
			}
			n.Type = t
		}

		if n.Type.IsUntyped() {
			if t.IsInterface() {
				n.Left, n.Right = defaultlit2(n.Left, n.Right, true)
				n.Type = n.Left.Type // same as n.Right.Type per defaultlit2
			} else {
				n.Left = convlit(n.Left, t)
				n.Right = convlit(n.Right, t)
				n.Type = t
			}
		}

		return n

	// target is invalid type for a constant? leave alone.
	case OLITERAL:
		if !okforconst[t.Etype] && n.Type.Etype != TNIL {
			return defaultlitreuse(n, nil, reuse)
		}

	case OLSH, ORSH:
		n.Left = convlit1(n.Left, t, explicit && n.Left.Type.IsUntyped(), noReuse)
		t = n.Left.Type
		if t != nil && t.Etype == TIDEAL && n.Val().Ctype() != CTINT {
			n.SetVal(toint(n.Val()))
		}
		if t != nil && !t.IsInteger() {
			yyerror("invalid operation: %v (shift of type %v)", n, t)
			t = nil
		}

		n.Type = t
		return n

	case OCOMPLEX:
		if n.Type.Etype == TIDEAL {
			switch t.Etype {
			default:
				// If trying to convert to non-complex type,
				// leave as complex128 and let typechecker complain.
				t = types.Types[TCOMPLEX128]
				fallthrough
			case types.TCOMPLEX128:
				n.Type = t
				n.Left = convlit(n.Left, types.Types[TFLOAT64])
				n.Right = convlit(n.Right, types.Types[TFLOAT64])

			case TCOMPLEX64:
				n.Type = t
				n.Left = convlit(n.Left, types.Types[TFLOAT32])
				n.Right = convlit(n.Right, types.Types[TFLOAT32])
			}
		}

		return n
	}

	// avoid repeated calculations, errors
	if types.Identical(n.Type, t) {
		return n
	}

	ct := consttype(n)
	var et types.EType
	if ct == 0 {
		goto bad
	}

	et = t.Etype
	if et == TINTER {
		if ct == CTNIL && n.Type == types.Types[TNIL] {
			n.Type = t
			return n
		}
		return defaultlitreuse(n, nil, reuse)
	}

	switch ct {
	default:
		goto bad

	case CTNIL:
		switch et {
		default:
			n.Type = nil
			goto bad

			// let normal conversion code handle it
		case TSTRING:
			return n

		case TARRAY:
			goto bad

		case TPTR, TUNSAFEPTR:
			n.SetVal(Val{new(Mpint)})

		case TCHAN, TFUNC, TINTER, TMAP, TSLICE:
			break
		}

	case CTSTR, CTBOOL:
		if et != n.Type.Etype {
			goto bad
		}

	case CTINT, CTRUNE, CTFLT, CTCPLX:
		if n.Type.Etype == TUNSAFEPTR && t.Etype != TUINTPTR {
			goto bad
		}
		ct := n.Val().Ctype()
		if isInt[et] {
			switch ct {
			default:
				goto bad

			case CTCPLX, CTFLT, CTRUNE:
				n.SetVal(toint(n.Val()))
				fallthrough

			case CTINT:
				overflow(n.Val(), t)
			}
		} else if isFloat[et] {
			switch ct {
			default:
				goto bad

			case CTCPLX, CTINT, CTRUNE:
				n.SetVal(toflt(n.Val()))
				fallthrough

			case CTFLT:
				n.SetVal(Val{truncfltlit(n.Val().U.(*Mpflt), t)})
			}
		} else if isComplex[et] {
			switch ct {
			default:
				goto bad

			case CTFLT, CTINT, CTRUNE:
				n.SetVal(tocplx(n.Val()))
				fallthrough

			case CTCPLX:
				n.SetVal(Val{trunccmplxlit(n.Val().U.(*Mpcplx), t)})
			}
		} else if et == types.TSTRING && (ct == CTINT || ct == CTRUNE) && explicit {
			n.SetVal(tostr(n.Val()))
		} else {
			goto bad
		}
	}

	n.Type = t
	return n

bad:
	if !n.Diag() {
		if !t.Broke() {
			yyerror("cannot convert %L to type %v", n, t)
		}
		n.SetDiag(true)
	}

	if n.Type.IsUntyped() {
		n = defaultlitreuse(n, nil, reuse)
	}
	return n
}

func tocplx(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		c := new(Mpcplx)
		c.Real.SetInt(u)
		c.Imag.SetFloat64(0.0)
		v.U = c

	case *Mpflt:
		c := new(Mpcplx)
		c.Real.Set(u)
		c.Imag.SetFloat64(0.0)
		v.U = c
	}

	return v
}

func toflt(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		f := newMpflt()
		f.SetInt(u)
		v.U = f

	case *Mpcplx:
		f := newMpflt()
		f.Set(&u.Real)
		if u.Imag.CmpFloat64(0) != 0 {
			yyerror("constant %v truncated to real", u.GoString())
		}
		v.U = f
	}

	return v
}

func toint(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		if u.Rune {
			i := new(Mpint)
			i.Set(u)
			v.U = i
		}

	case *Mpflt:
		i := new(Mpint)
		if !i.SetFloat(u) {
			if i.checkOverflow(0) {
				yyerror("integer too large")
			} else {
				// The value of u cannot be represented as an integer;
				// so we need to print an error message.
				// Unfortunately some float values cannot be
				// reasonably formatted for inclusion in an error
				// message (example: 1 + 1e-100), so first we try to
				// format the float; if the truncation resulted in
				// something that looks like an integer we omit the
				// value from the error message.
				// (See issue #11371).
				var t big.Float
				t.Parse(u.GoString(), 10)
				if t.IsInt() {
					yyerror("constant truncated to integer")
				} else {
					yyerror("constant %v truncated to integer", u.GoString())
				}
			}
		}
		v.U = i

	case *Mpcplx:
		i := new(Mpint)
		if !i.SetFloat(&u.Real) || u.Imag.CmpFloat64(0) != 0 {
			yyerror("constant %v truncated to integer", u.GoString())
		}

		v.U = i
	}

	return v
}

func doesoverflow(v Val, t *types.Type) bool {
	switch u := v.U.(type) {
	case *Mpint:
		if !t.IsInteger() {
			Fatalf("overflow: %v integer constant", t)
		}
		return u.Cmp(minintval[t.Etype]) < 0 || u.Cmp(maxintval[t.Etype]) > 0

	case *Mpflt:
		if !t.IsFloat() {
			Fatalf("overflow: %v floating-point constant", t)
		}
		return u.Cmp(minfltval[t.Etype]) <= 0 || u.Cmp(maxfltval[t.Etype]) >= 0

	case *Mpcplx:
		if !t.IsComplex() {
			Fatalf("overflow: %v complex constant", t)
		}
		return u.Real.Cmp(minfltval[t.Etype]) <= 0 || u.Real.Cmp(maxfltval[t.Etype]) >= 0 ||
			u.Imag.Cmp(minfltval[t.Etype]) <= 0 || u.Imag.Cmp(maxfltval[t.Etype]) >= 0
	}

	return false
}

func overflow(v Val, t *types.Type) bool {
	// v has already been converted
	// to appropriate form for t.
	if t == nil || t.Etype == TIDEAL {
		return false
	}

	// Only uintptrs may be converted to pointers, which cannot overflow.
	if t.IsPtr() || t.IsUnsafePtr() {
		return false
	}

	if doesoverflow(v, t) {
		yyerror("constant %v overflows %v", v, t)
		return true
	}

	return false

}

func tostr(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		var i int64 = 0xFFFD
		if u.Cmp(minintval[TUINT32]) >= 0 && u.Cmp(maxintval[TUINT32]) <= 0 {
			i = u.Int64()
		}
		v.U = string(i)
	}

	return v
}

func consttype(n *Node) Ctype {
	if n == nil || n.Op != OLITERAL {
		return 0
	}
	return n.Val().Ctype()
}

func Isconst(n *Node, ct Ctype) bool {
	t := consttype(n)

	// If the caller is asking for CTINT, allow CTRUNE too.
	// Makes life easier for back ends.
	return t == ct || (ct == CTINT && t == CTRUNE)
}

// evconst rewrites constant expressions into OLITERAL nodes.
func evconst(n *Node) {
	nl, nr := n.Left, n.Right

	// Pick off just the opcodes that can be constant evaluated.
	switch op := n.Op; op {
	case OPLUS, ONEG, OBITNOT, ONOT:
		if nl.Op == OLITERAL {
			setconst(n, unaryOp(op, nl.Val(), n.Type))
		}

	case OADD, OSUB, OMUL, ODIV, OMOD, OOR, OXOR, OAND, OANDNOT, OOROR, OANDAND:
		if nl.Op == OLITERAL && nr.Op == OLITERAL {
			setconst(n, binaryOp(nl.Val(), op, nr.Val()))
		}

	case OEQ, ONE, OLT, OLE, OGT, OGE:
		if nl.Op == OLITERAL && nr.Op == OLITERAL {
			if nl.Type.IsInterface() != nr.Type.IsInterface() {
				// Mixed interface/non-interface
				// constant comparison means comparing
				// nil interface with some typed
				// constant, which is always unequal.
				// E.g., interface{}(nil) == (*int)(nil).
				setboolconst(n, op == ONE)
			} else {
				setboolconst(n, compareOp(nl.Val(), op, nr.Val()))
			}
		}

	case OLSH, ORSH:
		if nl.Op == OLITERAL && nr.Op == OLITERAL {
			setconst(n, shiftOp(nl.Val(), op, nr.Val()))
		}

	case OCONV:
		if n.Type != nil && okforconst[n.Type.Etype] && nl.Op == OLITERAL {
			// TODO(mdempsky): There should be a convval function.
			setconst(n, convlit1(nl, n.Type, true, false).Val())
		}

	case OBYTES2STR:
		// string([]byte(nil)) or string([]rune(nil))
		if nl.Op == OLITERAL && nl.Val().Ctype() == CTNIL {
			setconst(n, Val{U: ""})
		}

	case OADDSTR:
		// Merge adjacent constants in the argument list.
		s := n.List.Slice()
		for i1 := 0; i1 < len(s); i1++ {
			if Isconst(s[i1], CTSTR) && i1+1 < len(s) && Isconst(s[i1+1], CTSTR) {
				// merge from i1 up to but not including i2
				var strs []string
				i2 := i1
				for i2 < len(s) && Isconst(s[i2], CTSTR) {
					strs = append(strs, s[i2].Val().U.(string))
					i2++
				}

				nl := *s[i1]
				nl.Orig = &nl
				nl.SetVal(Val{strings.Join(strs, "")})
				s[i1] = &nl
				s = append(s[:i1+1], s[i2:]...)
			}
		}

		if len(s) == 1 && Isconst(s[0], CTSTR) {
			n.Op = OLITERAL
			n.SetVal(s[0].Val())
		} else {
			n.List.Set(s)
		}
	}
}

func match(x, y Val) (Val, Val) {
	switch {
	case x.Ctype() == CTCPLX || y.Ctype() == CTCPLX:
		return tocplx(x), tocplx(y)
	case x.Ctype() == CTFLT || y.Ctype() == CTFLT:
		return toflt(x), toflt(y)
	}

	// Mixed int/rune are fine.
	return x, y
}

func compareOp(x Val, op Op, y Val) bool {
	x, y = match(x, y)

	switch x.Ctype() {
	case CTNIL:
		_, _ = x.U.(*NilVal), y.U.(*NilVal) // assert dynamic types match
		switch op {
		case OEQ:
			return true
		case ONE:
			return false
		}

	case CTBOOL:
		x, y := x.U.(bool), y.U.(bool)
		switch op {
		case OEQ:
			return x == y
		case ONE:
			return x != y
		}

	case CTINT, CTRUNE:
		x, y := x.U.(*Mpint), y.U.(*Mpint)
		return cmpZero(x.Cmp(y), op)

	case CTFLT:
		x, y := x.U.(*Mpflt), y.U.(*Mpflt)
		return cmpZero(x.Cmp(y), op)

	case CTCPLX:
		x, y := x.U.(*Mpcplx), y.U.(*Mpcplx)
		eq := x.Real.Cmp(&y.Real) == 0 && x.Imag.Cmp(&y.Imag) == 0
		switch op {
		case OEQ:
			return eq
		case ONE:
			return !eq
		}

	case CTSTR:
		x, y := x.U.(string), y.U.(string)
		switch op {
		case OEQ:
			return x == y
		case ONE:
			return x != y
		case OLT:
			return x < y
		case OLE:
			return x <= y
		case OGT:
			return x > y
		case OGE:
			return x >= y
		}
	}

	Fatalf("compareOp: bad comparison: %v %v %v", x, op, y)
	panic("unreachable")
}

func cmpZero(x int, op Op) bool {
	switch op {
	case OEQ:
		return x == 0
	case ONE:
		return x != 0
	case OLT:
		return x < 0
	case OLE:
		return x <= 0
	case OGT:
		return x > 0
	case OGE:
		return x >= 0
	}

	Fatalf("cmpZero: want comparison operator, got %v", op)
	panic("unreachable")
}

func binaryOp(x Val, op Op, y Val) Val {
	x, y = match(x, y)

Outer:
	switch x.Ctype() {
	case CTBOOL:
		x, y := x.U.(bool), y.U.(bool)
		switch op {
		case OANDAND:
			return Val{U: x && y}
		case OOROR:
			return Val{U: x || y}
		}

	case CTINT, CTRUNE:
		x, y := x.U.(*Mpint), y.U.(*Mpint)

		u := new(Mpint)
		u.Rune = x.Rune || y.Rune
		u.Set(x)
		switch op {
		case OADD:
			u.Add(y)
		case OSUB:
			u.Sub(y)
		case OMUL:
			u.Mul(y)
		case ODIV:
			if y.CmpInt64(0) == 0 {
				yyerror("division by zero")
				u.SetOverflow()
				break
			}
			u.Quo(y)
		case OMOD:
			if y.CmpInt64(0) == 0 {
				yyerror("division by zero")
				u.SetOverflow()
				break
			}
			u.Rem(y)
		case OOR:
			u.Or(y)
		case OAND:
			u.And(y)
		case OANDNOT:
			u.AndNot(y)
		case OXOR:
			u.Xor(y)
		default:
			break Outer
		}
		return Val{U: u}

	case CTFLT:
		x, y := x.U.(*Mpflt), y.U.(*Mpflt)

		u := newMpflt()
		u.Set(x)
		switch op {
		case OADD:
			u.Add(y)
		case OSUB:
			u.Sub(y)
		case OMUL:
			u.Mul(y)
		case ODIV:
			if y.CmpFloat64(0) == 0 {
				yyerror("division by zero")
				u.SetFloat64(1)
				break
			}
			u.Quo(y)
		case OMOD:
			// TODO(mdempsky): Move to typecheck.
			yyerror("illegal constant expression: floating-point %% operation")
		default:
			break Outer
		}
		return Val{U: u}

	case CTCPLX:
		x, y := x.U.(*Mpcplx), y.U.(*Mpcplx)

		u := new(Mpcplx)
		u.Real.Set(&x.Real)
		u.Imag.Set(&x.Imag)
		switch op {
		case OADD:
			u.Real.Add(&y.Real)
			u.Imag.Add(&y.Imag)
		case OSUB:
			u.Real.Sub(&y.Real)
			u.Imag.Sub(&y.Imag)
		case OMUL:
			u.Mul(y)
		case ODIV:
			if !u.Div(y) {
				yyerror("complex division by zero")
				u.Real.SetFloat64(1)
				u.Imag.SetFloat64(0)
			}
		default:
			break Outer
		}
		return Val{U: u}
	}

	Fatalf("binaryOp: bad operation: %v %v %v", x, op, y)
	panic("unreachable")
}

func unaryOp(op Op, x Val, t *types.Type) Val {
	switch op {
	case OPLUS:
		switch x.Ctype() {
		case CTINT, CTRUNE, CTFLT, CTCPLX:
			return x
		}

	case ONEG:
		switch x.Ctype() {
		case CTINT, CTRUNE:
			x := x.U.(*Mpint)
			u := new(Mpint)
			u.Rune = x.Rune
			u.Set(x)
			u.Neg()
			return Val{U: u}

		case CTFLT:
			x := x.U.(*Mpflt)
			u := newMpflt()
			u.Set(x)
			u.Neg()
			return Val{U: u}

		case CTCPLX:
			x := x.U.(*Mpcplx)
			u := new(Mpcplx)
			u.Real.Set(&x.Real)
			u.Imag.Set(&x.Imag)
			u.Real.Neg()
			u.Imag.Neg()
			return Val{U: u}
		}

	case OBITNOT:
		x := x.U.(*Mpint)

		u := new(Mpint)
		u.Rune = x.Rune
		if t.IsSigned() || t.IsUntyped() {
			// Signed values change sign.
			u.SetInt64(-1)
		} else {
			// Unsigned values invert their bits.
			u.Set(maxintval[t.Etype])
		}
		u.Xor(x)
		return Val{U: u}

	case ONOT:
		return Val{U: !x.U.(bool)}
	}

	Fatalf("unaryOp: bad operation: %v %v", op, x)
	panic("unreachable")
}

func shiftOp(x Val, op Op, y Val) Val {
	if x.Ctype() != CTRUNE {
		x = toint(x)
	}
	y = toint(y)

	u := new(Mpint)
	u.Set(x.U.(*Mpint))
	u.Rune = x.U.(*Mpint).Rune
	switch op {
	case OLSH:
		u.Lsh(y.U.(*Mpint))
	case ORSH:
		u.Rsh(y.U.(*Mpint))
	default:
		Fatalf("shiftOp: bad operator: %v", op)
		panic("unreachable")
	}
	return Val{U: u}
}

// setconst rewrites n as an OLITERAL with value v.
func setconst(n *Node, v Val) {
	// Ensure n.Orig still points to a semantically-equivalent
	// expression after we rewrite n into a constant.
	if n.Orig == n {
		n.Orig = n.sepcopy()
	}

	*n = Node{
		Op:      OLITERAL,
		Pos:     n.Pos,
		Orig:    n.Orig,
		Type:    n.Type,
		Xoffset: BADWIDTH,
	}
	n.SetVal(v)

	// Check range.
	lno := setlineno(n)
	overflow(v, n.Type)
	lineno = lno

	// Truncate precision for non-ideal float.
	if v.Ctype() == CTFLT && n.Type.Etype != TIDEAL {
		n.SetVal(Val{truncfltlit(v.U.(*Mpflt), n.Type)})
	}
}

func setboolconst(n *Node, v bool) {
	setconst(n, Val{U: v})
}

func setintconst(n *Node, v int64) {
	u := new(Mpint)
	u.SetInt64(v)
	setconst(n, Val{u})
}

// nodlit returns a new untyped constant with value v.
func nodlit(v Val) *Node {
	n := nod(OLITERAL, nil, nil)
	n.SetVal(v)
	switch v.Ctype() {
	default:
		Fatalf("nodlit ctype %d", v.Ctype())

	case CTSTR:
		n.Type = types.Idealstring

	case CTBOOL:
		n.Type = types.Idealbool

	case CTINT, CTRUNE, CTFLT, CTCPLX:
		n.Type = types.Types[TIDEAL]

	case CTNIL:
		n.Type = types.Types[TNIL]
	}

	return n
}

// idealkind returns a constant kind like consttype
// but for an arbitrary "ideal" (untyped constant) expression.
func idealkind(n *Node) Ctype {
	if n == nil || !n.Type.IsUntyped() {
		return CTxxx
	}

	switch n.Op {
	default:
		return CTxxx

	case OLITERAL:
		return n.Val().Ctype()

		// numeric kinds.
	case OADD,
		OAND,
		OANDNOT,
		OBITNOT,
		ODIV,
		ONEG,
		OMOD,
		OMUL,
		OSUB,
		OXOR,
		OOR,
		OPLUS:
		k1 := idealkind(n.Left)
		k2 := idealkind(n.Right)
		if k1 > k2 {
			return k1
		} else {
			return k2
		}

	case OREAL, OIMAG:
		return CTFLT

	case OCOMPLEX:
		return CTCPLX

	case OADDSTR:
		return CTSTR

	case OANDAND,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLT,
		ONE,
		ONOT,
		OOROR:
		return CTBOOL

		// shifts (beware!).
	case OLSH, ORSH:
		return idealkind(n.Left)
	}
}

// The result of defaultlit MUST be assigned back to n, e.g.
// 	n.Left = defaultlit(n.Left, t)
func defaultlit(n *Node, t *types.Type) *Node {
	return defaultlitreuse(n, t, noReuse)
}

// The result of defaultlitreuse MUST be assigned back to n, e.g.
// 	n.Left = defaultlitreuse(n.Left, t, reuse)
func defaultlitreuse(n *Node, t *types.Type, reuse canReuseNode) *Node {
	if n == nil || !n.Type.IsUntyped() {
		return n
	}

	if n.Op == OLITERAL && !reuse {
		n = n.rawcopy()
		reuse = true
	}

	lno := setlineno(n)
	ctype := idealkind(n)
	var t1 *types.Type
	switch ctype {
	default:
		if t != nil {
			return convlit(n, t)
		}

		switch n.Val().Ctype() {
		case CTNIL:
			lineno = lno
			if !n.Diag() {
				yyerror("use of untyped nil")
				n.SetDiag(true)
			}

			n.Type = nil
		case CTSTR:
			t1 := types.Types[TSTRING]
			n = convlit1(n, t1, false, reuse)
		default:
			yyerror("defaultlit: unknown literal: %v", n)
		}
		lineno = lno
		return n

	case CTxxx:
		Fatalf("defaultlit: idealkind is CTxxx: %+v", n)

	case CTBOOL:
		t1 := types.Types[TBOOL]
		if t != nil && t.IsBoolean() {
			t1 = t
		}
		n = convlit1(n, t1, false, reuse)
		lineno = lno
		return n

	case CTINT:
		t1 = types.Types[TINT]
	case CTRUNE:
		t1 = types.Runetype
	case CTFLT:
		t1 = types.Types[TFLOAT64]
	case CTCPLX:
		t1 = types.Types[TCOMPLEX128]
	}

	// Note: n.Val().Ctype() can be CTxxx (not a constant) here
	// in the case of an untyped non-constant value, like 1<<i.
	v1 := n.Val()
	if t != nil {
		if t.IsInteger() {
			t1 = t
			v1 = toint(n.Val())
		} else if t.IsFloat() {
			t1 = t
			v1 = toflt(n.Val())
		} else if t.IsComplex() {
			t1 = t
			v1 = tocplx(n.Val())
		}
		if n.Val().Ctype() != CTxxx {
			n.SetVal(v1)
		}
	}

	if n.Val().Ctype() != CTxxx {
		overflow(n.Val(), t1)
	}
	n = convlit1(n, t1, false, reuse)
	lineno = lno
	return n
}

// defaultlit on both nodes simultaneously;
// if they're both ideal going in they better
// get the same type going out.
// force means must assign concrete (non-ideal) type.
// The results of defaultlit2 MUST be assigned back to l and r, e.g.
// 	n.Left, n.Right = defaultlit2(n.Left, n.Right, force)
func defaultlit2(l *Node, r *Node, force bool) (*Node, *Node) {
	if l.Type == nil || r.Type == nil {
		return l, r
	}
	if !l.Type.IsUntyped() {
		r = convlit(r, l.Type)
		return l, r
	}

	if !r.Type.IsUntyped() {
		l = convlit(l, r.Type)
		return l, r
	}

	if !force {
		return l, r
	}

	if l.Type.IsBoolean() {
		l = convlit(l, types.Types[TBOOL])
		r = convlit(r, types.Types[TBOOL])
	}

	lkind := idealkind(l)
	rkind := idealkind(r)
	if lkind == CTCPLX || rkind == CTCPLX {
		l = convlit(l, types.Types[TCOMPLEX128])
		r = convlit(r, types.Types[TCOMPLEX128])
		return l, r
	}

	if lkind == CTFLT || rkind == CTFLT {
		l = convlit(l, types.Types[TFLOAT64])
		r = convlit(r, types.Types[TFLOAT64])
		return l, r
	}

	if lkind == CTRUNE || rkind == CTRUNE {
		l = convlit(l, types.Runetype)
		r = convlit(r, types.Runetype)
		return l, r
	}

	l = convlit(l, types.Types[TINT])
	r = convlit(r, types.Types[TINT])

	return l, r
}

// strlit returns the value of a literal string Node as a string.
func strlit(n *Node) string {
	return n.Val().U.(string)
}

func smallintconst(n *Node) bool {
	if n.Op == OLITERAL && Isconst(n, CTINT) && n.Type != nil {
		switch simtype[n.Type.Etype] {
		case TINT8,
			TUINT8,
			TINT16,
			TUINT16,
			TINT32,
			TUINT32,
			TBOOL:
			return true

		case TIDEAL, TINT64, TUINT64, TPTR:
			v, ok := n.Val().U.(*Mpint)
			if ok && v.Cmp(minintval[TINT32]) > 0 && v.Cmp(maxintval[TINT32]) < 0 {
				return true
			}
		}
	}

	return false
}

// nonnegintconst checks if Node n contains a constant expression
// representable as a non-negative small integer, and returns its
// (integer) value if that's the case. Otherwise, it returns -1.
func nonnegintconst(n *Node) int64 {
	if n.Op != OLITERAL {
		return -1
	}

	// toint will leave n.Val unchanged if it's not castable to an
	// Mpint, so we still have to guard the conversion.
	v := toint(n.Val())
	vi, ok := v.U.(*Mpint)
	if !ok || vi.CmpInt64(0) < 0 || vi.Cmp(maxintval[TINT32]) > 0 {
		return -1
	}

	return vi.Int64()
}

// isGoConst reports whether n is a Go language constant (as opposed to a
// compile-time constant).
//
// Expressions derived from nil, like string([]byte(nil)), while they
// may be known at compile time, are not Go language constants.
// Only called for expressions known to evaluated to compile-time
// constants.
func (n *Node) isGoConst() bool {
	if n.Orig != nil {
		n = n.Orig
	}

	switch n.Op {
	case OADD,
		OADDSTR,
		OAND,
		OANDAND,
		OANDNOT,
		OBITNOT,
		ODIV,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLSH,
		OLT,
		ONEG,
		OMOD,
		OMUL,
		ONE,
		ONOT,
		OOR,
		OOROR,
		OPLUS,
		ORSH,
		OSUB,
		OXOR,
		OIOTA,
		OCOMPLEX,
		OREAL,
		OIMAG:
		if n.Left.isGoConst() && (n.Right == nil || n.Right.isGoConst()) {
			return true
		}

	case OCONV:
		if okforconst[n.Type.Etype] && n.Left.isGoConst() {
			return true
		}

	case OLEN, OCAP:
		l := n.Left
		if l.isGoConst() {
			return true
		}

		// Special case: len/cap is constant when applied to array or
		// pointer to array when the expression does not contain
		// function calls or channel receive operations.
		t := l.Type

		if t != nil && t.IsPtr() {
			t = t.Elem()
		}
		if t != nil && t.IsArray() && !hascallchan(l) {
			return true
		}

	case OLITERAL:
		if n.Val().Ctype() != CTNIL {
			return true
		}

	case ONAME:
		l := asNode(n.Sym.Def)
		if l != nil && l.Op == OLITERAL && n.Val().Ctype() != CTNIL {
			return true
		}

	case ONONAME:
		if asNode(n.Sym.Def) != nil && asNode(n.Sym.Def).Op == OIOTA {
			return true
		}

	case OALIGNOF, OOFFSETOF, OSIZEOF:
		return true
	}

	//dump("nonconst", n);
	return false
}

func hascallchan(n *Node) bool {
	if n == nil {
		return false
	}
	switch n.Op {
	case OAPPEND,
		OCALL,
		OCALLFUNC,
		OCALLINTER,
		OCALLMETH,
		OCAP,
		OCLOSE,
		OCOMPLEX,
		OCOPY,
		ODELETE,
		OIMAG,
		OLEN,
		OMAKE,
		ONEW,
		OPANIC,
		OPRINT,
		OPRINTN,
		OREAL,
		ORECOVER,
		ORECV:
		return true
	}

	if hascallchan(n.Left) || hascallchan(n.Right) {
		return true
	}
	for _, n1 := range n.List.Slice() {
		if hascallchan(n1) {
			return true
		}
	}
	for _, n2 := range n.Rlist.Slice() {
		if hascallchan(n2) {
			return true
		}
	}

	return false
}
