// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"go/constant"
	"go/token"
	"math"
	"math/big"
	"strings"
	"unicode"
)

const (
	// Maximum size in bits for big.Ints before signalling
	// overflow and also mantissa precision for big.Floats.
	Mpprec = 512
)

func bigFloatVal(v constant.Value) *big.Float {
	f := new(big.Float)
	f.SetPrec(Mpprec)
	switch u := constant.Val(v).(type) {
	case int64:
		f.SetInt64(u)
	case *big.Int:
		f.SetInt(u)
	case *big.Float:
		f.Set(u)
	case *big.Rat:
		f.SetRat(u)
	default:
		base.Fatalf("unexpected: %v", u)
	}
	return f
}

func roundFloat(v constant.Value, sz int64) constant.Value {
	switch sz {
	case 4:
		f, _ := constant.Float32Val(v)
		return makeFloat64(float64(f))
	case 8:
		f, _ := constant.Float64Val(v)
		return makeFloat64(f)
	}
	base.Fatalf("unexpected size: %v", sz)
	panic("unreachable")
}

// truncate float literal fv to 32-bit or 64-bit precision
// according to type; return truncated value.
func truncfltlit(v constant.Value, t *types.Type) constant.Value {
	if t.IsUntyped() || overflow(v, t) {
		// If there was overflow, simply continuing would set the
		// value to Inf which in turn would lead to spurious follow-on
		// errors. Avoid this by returning the existing value.
		return v
	}

	return roundFloat(v, t.Size())
}

// truncate Real and Imag parts of Mpcplx to 32-bit or 64-bit
// precision, according to type; return truncated value. In case of
// overflow, calls Errorf but does not truncate the input value.
func trunccmplxlit(v constant.Value, t *types.Type) constant.Value {
	if t.IsUntyped() || overflow(v, t) {
		// If there was overflow, simply continuing would set the
		// value to Inf which in turn would lead to spurious follow-on
		// errors. Avoid this by returning the existing value.
		return v
	}

	fsz := t.Size() / 2
	return makeComplex(roundFloat(constant.Real(v), fsz), roundFloat(constant.Imag(v), fsz))
}

// TODO(mdempsky): Replace these with better APIs.
func convlit(n ir.Node, t *types.Type) ir.Node    { return convlit1(n, t, false, nil) }
func defaultlit(n ir.Node, t *types.Type) ir.Node { return convlit1(n, t, false, nil) }

// convlit1 converts an untyped expression n to type t. If n already
// has a type, convlit1 has no effect.
//
// For explicit conversions, t must be non-nil, and integer-to-string
// conversions are allowed.
//
// For implicit conversions (e.g., assignments), t may be nil; if so,
// n is converted to its default type.
//
// If there's an error converting n to t, context is used in the error
// message.
func convlit1(n ir.Node, t *types.Type, explicit bool, context func() string) ir.Node {
	if explicit && t == nil {
		base.Fatalf("explicit conversion missing type")
	}
	if t != nil && t.IsUntyped() {
		base.Fatalf("bad conversion to untyped: %v", t)
	}

	if n == nil || n.Type() == nil {
		// Allow sloppy callers.
		return n
	}
	if !n.Type().IsUntyped() {
		// Already typed; nothing to do.
		return n
	}

	// Nil is technically not a constant, so handle it specially.
	if n.Type().Kind() == types.TNIL {
		if n.Op() != ir.ONIL {
			base.Fatalf("unexpected op: %v (%v)", n, n.Op())
		}
		n = ir.Copy(n)
		if t == nil {
			base.Errorf("use of untyped nil")
			n.SetDiag(true)
			n.SetType(nil)
			return n
		}

		if !t.HasNil() {
			// Leave for caller to handle.
			return n
		}

		n.SetType(t)
		return n
	}

	if t == nil || !ir.OKForConst[t.Kind()] {
		t = defaultType(n.Type())
	}

	switch n.Op() {
	default:
		base.Fatalf("unexpected untyped expression: %v", n)

	case ir.OLITERAL:
		v := convertVal(n.Val(), t, explicit)
		if v.Kind() == constant.Unknown {
			n = ir.NewConstExpr(n.Val(), n)
			break
		}
		n = ir.NewConstExpr(v, n)
		n.SetType(t)
		return n

	case ir.OPLUS, ir.ONEG, ir.OBITNOT, ir.ONOT, ir.OREAL, ir.OIMAG:
		ot := operandType(n.Op(), t)
		if ot == nil {
			n = defaultlit(n, nil)
			break
		}

		n.SetLeft(convlit(n.Left(), ot))
		if n.Left().Type() == nil {
			n.SetType(nil)
			return n
		}
		n.SetType(t)
		return n

	case ir.OADD, ir.OSUB, ir.OMUL, ir.ODIV, ir.OMOD, ir.OOR, ir.OXOR, ir.OAND, ir.OANDNOT, ir.OOROR, ir.OANDAND, ir.OCOMPLEX:
		ot := operandType(n.Op(), t)
		if ot == nil {
			n = defaultlit(n, nil)
			break
		}

		n.SetLeft(convlit(n.Left(), ot))
		n.SetRight(convlit(n.Right(), ot))
		if n.Left().Type() == nil || n.Right().Type() == nil {
			n.SetType(nil)
			return n
		}
		if !types.Identical(n.Left().Type(), n.Right().Type()) {
			base.Errorf("invalid operation: %v (mismatched types %v and %v)", n, n.Left().Type(), n.Right().Type())
			n.SetType(nil)
			return n
		}

		n.SetType(t)
		return n

	case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		if !t.IsBoolean() {
			break
		}
		n.SetType(t)
		return n

	case ir.OLSH, ir.ORSH:
		n.SetLeft(convlit1(n.Left(), t, explicit, nil))
		n.SetType(n.Left().Type())
		if n.Type() != nil && !n.Type().IsInteger() {
			base.Errorf("invalid operation: %v (shift of type %v)", n, n.Type())
			n.SetType(nil)
		}
		return n
	}

	if !n.Diag() {
		if !t.Broke() {
			if explicit {
				base.Errorf("cannot convert %L to type %v", n, t)
			} else if context != nil {
				base.Errorf("cannot use %L as type %v in %s", n, t, context())
			} else {
				base.Errorf("cannot use %L as type %v", n, t)
			}
		}
		n.SetDiag(true)
	}
	n.SetType(nil)
	return n
}

func operandType(op ir.Op, t *types.Type) *types.Type {
	switch op {
	case ir.OCOMPLEX:
		if t.IsComplex() {
			return floatForComplex(t)
		}
	case ir.OREAL, ir.OIMAG:
		if t.IsFloat() {
			return complexForFloat(t)
		}
	default:
		if okfor[op][t.Kind()] {
			return t
		}
	}
	return nil
}

// convertVal converts v into a representation appropriate for t. If
// no such representation exists, it returns Val{} instead.
//
// If explicit is true, then conversions from integer to string are
// also allowed.
func convertVal(v constant.Value, t *types.Type, explicit bool) constant.Value {
	switch ct := v.Kind(); ct {
	case constant.Bool:
		if t.IsBoolean() {
			return v
		}

	case constant.String:
		if t.IsString() {
			return v
		}

	case constant.Int:
		if explicit && t.IsString() {
			return tostr(v)
		}
		fallthrough
	case constant.Float, constant.Complex:
		switch {
		case t.IsInteger():
			v = toint(v)
			overflow(v, t)
			return v
		case t.IsFloat():
			v = toflt(v)
			v = truncfltlit(v, t)
			return v
		case t.IsComplex():
			v = tocplx(v)
			v = trunccmplxlit(v, t)
			return v
		}
	}

	return constant.MakeUnknown()
}

func tocplx(v constant.Value) constant.Value {
	return constant.ToComplex(v)
}

func toflt(v constant.Value) constant.Value {
	if v.Kind() == constant.Complex {
		if constant.Sign(constant.Imag(v)) != 0 {
			base.Errorf("constant %v truncated to real", v)
		}
		v = constant.Real(v)
	}

	return constant.ToFloat(v)
}

func toint(v constant.Value) constant.Value {
	if v.Kind() == constant.Complex {
		if constant.Sign(constant.Imag(v)) != 0 {
			base.Errorf("constant %v truncated to integer", v)
		}
		v = constant.Real(v)
	}

	if v := constant.ToInt(v); v.Kind() == constant.Int {
		return v
	}

	// The value of v cannot be represented as an integer;
	// so we need to print an error message.
	// Unfortunately some float values cannot be
	// reasonably formatted for inclusion in an error
	// message (example: 1 + 1e-100), so first we try to
	// format the float; if the truncation resulted in
	// something that looks like an integer we omit the
	// value from the error message.
	// (See issue #11371).
	f := bigFloatVal(v)
	if f.MantExp(nil) > 2*Mpprec {
		base.Errorf("integer too large")
	} else {
		var t big.Float
		t.Parse(fmt.Sprint(v), 0)
		if t.IsInt() {
			base.Errorf("constant truncated to integer")
		} else {
			base.Errorf("constant %v truncated to integer", v)
		}
	}

	// Prevent follow-on errors.
	// TODO(mdempsky): Use constant.MakeUnknown() instead.
	return constant.MakeInt64(1)
}

// doesoverflow reports whether constant value v is too large
// to represent with type t.
func doesoverflow(v constant.Value, t *types.Type) bool {
	switch {
	case t.IsInteger():
		bits := uint(8 * t.Size())
		if t.IsUnsigned() {
			x, ok := constant.Uint64Val(v)
			return !ok || x>>bits != 0
		}
		x, ok := constant.Int64Val(v)
		if x < 0 {
			x = ^x
		}
		return !ok || x>>(bits-1) != 0
	case t.IsFloat():
		switch t.Size() {
		case 4:
			f, _ := constant.Float32Val(v)
			return math.IsInf(float64(f), 0)
		case 8:
			f, _ := constant.Float64Val(v)
			return math.IsInf(f, 0)
		}
	case t.IsComplex():
		ft := floatForComplex(t)
		return doesoverflow(constant.Real(v), ft) || doesoverflow(constant.Imag(v), ft)
	}
	base.Fatalf("doesoverflow: %v, %v", v, t)
	panic("unreachable")
}

// overflow reports whether constant value v is too large
// to represent with type t, and emits an error message if so.
func overflow(v constant.Value, t *types.Type) bool {
	// v has already been converted
	// to appropriate form for t.
	if t.IsUntyped() {
		return false
	}
	if v.Kind() == constant.Int && constant.BitLen(v) > Mpprec {
		base.Errorf("integer too large")
		return true
	}
	if doesoverflow(v, t) {
		base.Errorf("constant %v overflows %v", types.FmtConst(v, false), t)
		return true
	}
	return false
}

func tostr(v constant.Value) constant.Value {
	if v.Kind() == constant.Int {
		r := unicode.ReplacementChar
		if x, ok := constant.Uint64Val(v); ok && x <= unicode.MaxRune {
			r = rune(x)
		}
		v = constant.MakeString(string(r))
	}
	return v
}

var tokenForOp = [...]token.Token{
	ir.OPLUS:   token.ADD,
	ir.ONEG:    token.SUB,
	ir.ONOT:    token.NOT,
	ir.OBITNOT: token.XOR,

	ir.OADD:    token.ADD,
	ir.OSUB:    token.SUB,
	ir.OMUL:    token.MUL,
	ir.ODIV:    token.QUO,
	ir.OMOD:    token.REM,
	ir.OOR:     token.OR,
	ir.OXOR:    token.XOR,
	ir.OAND:    token.AND,
	ir.OANDNOT: token.AND_NOT,
	ir.OOROR:   token.LOR,
	ir.OANDAND: token.LAND,

	ir.OEQ: token.EQL,
	ir.ONE: token.NEQ,
	ir.OLT: token.LSS,
	ir.OLE: token.LEQ,
	ir.OGT: token.GTR,
	ir.OGE: token.GEQ,

	ir.OLSH: token.SHL,
	ir.ORSH: token.SHR,
}

// evalConst returns a constant-evaluated expression equivalent to n.
// If n is not a constant, evalConst returns n.
// Otherwise, evalConst returns a new OLITERAL with the same value as n,
// and with .Orig pointing back to n.
func evalConst(n ir.Node) ir.Node {
	nl, nr := n.Left(), n.Right()

	// Pick off just the opcodes that can be constant evaluated.
	switch op := n.Op(); op {
	case ir.OPLUS, ir.ONEG, ir.OBITNOT, ir.ONOT:
		if nl.Op() == ir.OLITERAL {
			var prec uint
			if n.Type().IsUnsigned() {
				prec = uint(n.Type().Size() * 8)
			}
			return origConst(n, constant.UnaryOp(tokenForOp[op], nl.Val(), prec))
		}

	case ir.OADD, ir.OSUB, ir.OMUL, ir.ODIV, ir.OMOD, ir.OOR, ir.OXOR, ir.OAND, ir.OANDNOT, ir.OOROR, ir.OANDAND:
		if nl.Op() == ir.OLITERAL && nr.Op() == ir.OLITERAL {
			rval := nr.Val()

			// check for divisor underflow in complex division (see issue 20227)
			if op == ir.ODIV && n.Type().IsComplex() && constant.Sign(square(constant.Real(rval))) == 0 && constant.Sign(square(constant.Imag(rval))) == 0 {
				base.Errorf("complex division by zero")
				n.SetType(nil)
				return n
			}
			if (op == ir.ODIV || op == ir.OMOD) && constant.Sign(rval) == 0 {
				base.Errorf("division by zero")
				n.SetType(nil)
				return n
			}

			tok := tokenForOp[op]
			if op == ir.ODIV && n.Type().IsInteger() {
				tok = token.QUO_ASSIGN // integer division
			}
			return origConst(n, constant.BinaryOp(nl.Val(), tok, rval))
		}

	case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		if nl.Op() == ir.OLITERAL && nr.Op() == ir.OLITERAL {
			return origBoolConst(n, constant.Compare(nl.Val(), tokenForOp[op], nr.Val()))
		}

	case ir.OLSH, ir.ORSH:
		if nl.Op() == ir.OLITERAL && nr.Op() == ir.OLITERAL {
			// shiftBound from go/types; "so we can express smallestFloat64"
			const shiftBound = 1023 - 1 + 52
			s, ok := constant.Uint64Val(nr.Val())
			if !ok || s > shiftBound {
				base.Errorf("invalid shift count %v", nr)
				n.SetType(nil)
				break
			}
			return origConst(n, constant.Shift(toint(nl.Val()), tokenForOp[op], uint(s)))
		}

	case ir.OCONV, ir.ORUNESTR:
		if ir.OKForConst[n.Type().Kind()] && nl.Op() == ir.OLITERAL {
			return origConst(n, convertVal(nl.Val(), n.Type(), true))
		}

	case ir.OCONVNOP:
		if ir.OKForConst[n.Type().Kind()] && nl.Op() == ir.OLITERAL {
			// set so n.Orig gets OCONV instead of OCONVNOP
			n.SetOp(ir.OCONV)
			return origConst(n, nl.Val())
		}

	case ir.OADDSTR:
		// Merge adjacent constants in the argument list.
		s := n.List().Slice()
		need := 0
		for i := 0; i < len(s); i++ {
			if i == 0 || !ir.IsConst(s[i-1], constant.String) || !ir.IsConst(s[i], constant.String) {
				// Can't merge s[i] into s[i-1]; need a slot in the list.
				need++
			}
		}
		if need == len(s) {
			return n
		}
		if need == 1 {
			var strs []string
			for _, c := range s {
				strs = append(strs, ir.StringVal(c))
			}
			return origConst(n, constant.MakeString(strings.Join(strs, "")))
		}
		newList := make([]ir.Node, 0, need)
		for i := 0; i < len(s); i++ {
			if ir.IsConst(s[i], constant.String) && i+1 < len(s) && ir.IsConst(s[i+1], constant.String) {
				// merge from i up to but not including i2
				var strs []string
				i2 := i
				for i2 < len(s) && ir.IsConst(s[i2], constant.String) {
					strs = append(strs, ir.StringVal(s[i2]))
					i2++
				}

				nl := ir.Copy(n)
				nl.PtrList().Set(s[i:i2])
				nl = origConst(nl, constant.MakeString(strings.Join(strs, "")))
				newList = append(newList, nl)
				i = i2 - 1
			} else {
				newList = append(newList, s[i])
			}
		}

		n = ir.Copy(n)
		n.PtrList().Set(newList)
		return n

	case ir.OCAP, ir.OLEN:
		switch nl.Type().Kind() {
		case types.TSTRING:
			if ir.IsConst(nl, constant.String) {
				return origIntConst(n, int64(len(ir.StringVal(nl))))
			}
		case types.TARRAY:
			if !hasCallOrChan(nl) {
				return origIntConst(n, nl.Type().NumElem())
			}
		}

	case ir.OALIGNOF, ir.OOFFSETOF, ir.OSIZEOF:
		return origIntConst(n, evalunsafe(n))

	case ir.OREAL:
		if nl.Op() == ir.OLITERAL {
			return origConst(n, constant.Real(nl.Val()))
		}

	case ir.OIMAG:
		if nl.Op() == ir.OLITERAL {
			return origConst(n, constant.Imag(nl.Val()))
		}

	case ir.OCOMPLEX:
		if nl.Op() == ir.OLITERAL && nr.Op() == ir.OLITERAL {
			return origConst(n, makeComplex(nl.Val(), nr.Val()))
		}
	}

	return n
}

func makeInt(i *big.Int) constant.Value {
	if i.IsInt64() {
		return constant.Make(i.Int64()) // workaround #42640 (Int64Val(Make(big.NewInt(10))) returns (10, false), not (10, true))
	}
	return constant.Make(i)
}

func makeFloat64(f float64) constant.Value {
	if math.IsInf(f, 0) {
		base.Fatalf("infinity is not a valid constant")
	}
	v := constant.MakeFloat64(f)
	v = constant.ToFloat(v) // workaround #42641 (MakeFloat64(0).Kind() returns Int, not Float)
	return v
}

func makeComplex(real, imag constant.Value) constant.Value {
	return constant.BinaryOp(constant.ToFloat(real), token.ADD, constant.MakeImag(constant.ToFloat(imag)))
}

func square(x constant.Value) constant.Value {
	return constant.BinaryOp(x, token.MUL, x)
}

// For matching historical "constant OP overflow" error messages.
// TODO(mdempsky): Replace with error messages like go/types uses.
var overflowNames = [...]string{
	ir.OADD:    "addition",
	ir.OSUB:    "subtraction",
	ir.OMUL:    "multiplication",
	ir.OLSH:    "shift",
	ir.OXOR:    "bitwise XOR",
	ir.OBITNOT: "bitwise complement",
}

// origConst returns an OLITERAL with orig n and value v.
func origConst(n ir.Node, v constant.Value) ir.Node {
	lno := setlineno(n)
	v = convertVal(v, n.Type(), false)
	base.Pos = lno

	switch v.Kind() {
	case constant.Int:
		if constant.BitLen(v) <= Mpprec {
			break
		}
		fallthrough
	case constant.Unknown:
		what := overflowNames[n.Op()]
		if what == "" {
			base.Fatalf("unexpected overflow: %v", n.Op())
		}
		base.ErrorfAt(n.Pos(), "constant %v overflow", what)
		n.SetType(nil)
		return n
	}

	return ir.NewConstExpr(v, n)
}

func origBoolConst(n ir.Node, v bool) ir.Node {
	return origConst(n, constant.MakeBool(v))
}

func origIntConst(n ir.Node, v int64) ir.Node {
	return origConst(n, constant.MakeInt64(v))
}

// defaultlit on both nodes simultaneously;
// if they're both ideal going in they better
// get the same type going out.
// force means must assign concrete (non-ideal) type.
// The results of defaultlit2 MUST be assigned back to l and r, e.g.
// 	n.Left, n.Right = defaultlit2(n.Left, n.Right, force)
func defaultlit2(l ir.Node, r ir.Node, force bool) (ir.Node, ir.Node) {
	if l.Type() == nil || r.Type() == nil {
		return l, r
	}
	if !l.Type().IsUntyped() {
		r = convlit(r, l.Type())
		return l, r
	}

	if !r.Type().IsUntyped() {
		l = convlit(l, r.Type())
		return l, r
	}

	if !force {
		return l, r
	}

	// Can't mix bool with non-bool, string with non-string, or nil with anything (untyped).
	if l.Type().IsBoolean() != r.Type().IsBoolean() {
		return l, r
	}
	if l.Type().IsString() != r.Type().IsString() {
		return l, r
	}
	if ir.IsNil(l) || ir.IsNil(r) {
		return l, r
	}

	t := defaultType(mixUntyped(l.Type(), r.Type()))
	l = convlit(l, t)
	r = convlit(r, t)
	return l, r
}

func mixUntyped(t1, t2 *types.Type) *types.Type {
	if t1 == t2 {
		return t1
	}

	rank := func(t *types.Type) int {
		switch t {
		case types.UntypedInt:
			return 0
		case types.UntypedRune:
			return 1
		case types.UntypedFloat:
			return 2
		case types.UntypedComplex:
			return 3
		}
		base.Fatalf("bad type %v", t)
		panic("unreachable")
	}

	if rank(t2) > rank(t1) {
		return t2
	}
	return t1
}

func defaultType(t *types.Type) *types.Type {
	if !t.IsUntyped() || t.Kind() == types.TNIL {
		return t
	}

	switch t {
	case types.UntypedBool:
		return types.Types[types.TBOOL]
	case types.UntypedString:
		return types.Types[types.TSTRING]
	case types.UntypedInt:
		return types.Types[types.TINT]
	case types.UntypedRune:
		return types.RuneType
	case types.UntypedFloat:
		return types.Types[types.TFLOAT64]
	case types.UntypedComplex:
		return types.Types[types.TCOMPLEX128]
	}

	base.Fatalf("bad type %v", t)
	return nil
}

func smallintconst(n ir.Node) bool {
	if n.Op() == ir.OLITERAL {
		v, ok := constant.Int64Val(n.Val())
		return ok && int64(int32(v)) == v
	}
	return false
}

// indexconst checks if Node n contains a constant expression
// representable as a non-negative int and returns its value.
// If n is not a constant expression, not representable as an
// integer, or negative, it returns -1. If n is too large, it
// returns -2.
func indexconst(n ir.Node) int64 {
	if n.Op() != ir.OLITERAL {
		return -1
	}
	if !n.Type().IsInteger() && n.Type().Kind() != types.TIDEAL {
		return -1
	}

	v := toint(n.Val())
	if v.Kind() != constant.Int || constant.Sign(v) < 0 {
		return -1
	}
	if doesoverflow(v, types.Types[types.TINT]) {
		return -2
	}
	return ir.IntVal(types.Types[types.TINT], v)
}

// isGoConst reports whether n is a Go language constant (as opposed to a
// compile-time constant).
//
// Expressions derived from nil, like string([]byte(nil)), while they
// may be known at compile time, are not Go language constants.
func isGoConst(n ir.Node) bool {
	return n.Op() == ir.OLITERAL
}

// hasCallOrChan reports whether n contains any calls or channel operations.
func hasCallOrChan(n ir.Node) bool {
	return ir.Find(n, func(n ir.Node) bool {
		switch n.Op() {
		case ir.OAPPEND,
			ir.OCALL,
			ir.OCALLFUNC,
			ir.OCALLINTER,
			ir.OCALLMETH,
			ir.OCAP,
			ir.OCLOSE,
			ir.OCOMPLEX,
			ir.OCOPY,
			ir.ODELETE,
			ir.OIMAG,
			ir.OLEN,
			ir.OMAKE,
			ir.ONEW,
			ir.OPANIC,
			ir.OPRINT,
			ir.OPRINTN,
			ir.OREAL,
			ir.ORECOVER,
			ir.ORECV:
			return true
		}
		return false
	})
}

// A constSet represents a set of Go constant expressions.
type constSet struct {
	m map[constSetKey]src.XPos
}

type constSetKey struct {
	typ *types.Type
	val interface{}
}

// add adds constant expression n to s. If a constant expression of
// equal value and identical type has already been added, then add
// reports an error about the duplicate value.
//
// pos provides position information for where expression n occurred
// (in case n does not have its own position information). what and
// where are used in the error message.
//
// n must not be an untyped constant.
func (s *constSet) add(pos src.XPos, n ir.Node, what, where string) {
	if n.Op() == ir.OCONVIFACE && n.Implicit() {
		n = n.Left()
	}

	if !isGoConst(n) {
		return
	}
	if n.Type().IsUntyped() {
		base.Fatalf("%v is untyped", n)
	}

	// Consts are only duplicates if they have the same value and
	// identical types.
	//
	// In general, we have to use types.Identical to test type
	// identity, because == gives false negatives for anonymous
	// types and the byte/uint8 and rune/int32 builtin type
	// aliases.  However, this is not a problem here, because
	// constant expressions are always untyped or have a named
	// type, and we explicitly handle the builtin type aliases
	// below.
	//
	// This approach may need to be revisited though if we fix
	// #21866 by treating all type aliases like byte/uint8 and
	// rune/int32.

	typ := n.Type()
	switch typ {
	case types.ByteType:
		typ = types.Types[types.TUINT8]
	case types.RuneType:
		typ = types.Types[types.TINT32]
	}
	k := constSetKey{typ, ir.ConstValue(n)}

	if hasUniquePos(n) {
		pos = n.Pos()
	}

	if s.m == nil {
		s.m = make(map[constSetKey]src.XPos)
	}

	if prevPos, isDup := s.m[k]; isDup {
		base.ErrorfAt(pos, "duplicate %s %s in %s\n\tprevious %s at %v",
			what, nodeAndVal(n), where,
			what, base.FmtPos(prevPos))
	} else {
		s.m[k] = pos
	}
}

// nodeAndVal reports both an expression and its constant value, if
// the latter is non-obvious.
//
// TODO(mdempsky): This could probably be a fmt.go flag.
func nodeAndVal(n ir.Node) string {
	show := fmt.Sprint(n)
	val := ir.ConstValue(n)
	if s := fmt.Sprintf("%#v", val); show != s {
		show += " (value " + s + ")"
	}
	return show
}
