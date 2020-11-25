// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
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

// ValueInterface returns the constant value stored in n as an interface{}.
// It returns int64s for ints and runes, float64s for floats,
// and complex128s for complex values.
func (n *Node) ValueInterface() interface{} {
	switch v := n.Val(); v.Kind() {
	default:
		Fatalf("unexpected constant: %v", v)
		panic("unreachable")
	case constant.Bool:
		return constant.BoolVal(v)
	case constant.String:
		return constant.StringVal(v)
	case constant.Int:
		return int64Val(n.Type, v)
	case constant.Float:
		return float64Val(v)
	case constant.Complex:
		return complex(float64Val(constant.Real(v)), float64Val(constant.Imag(v)))
	}
}

// int64Val returns v converted to int64.
// Note: if t is uint64, very large values will be converted to negative int64.
func int64Val(t *types.Type, v constant.Value) int64 {
	if t.IsUnsigned() {
		if x, ok := constant.Uint64Val(v); ok {
			return int64(x)
		}
	} else {
		if x, ok := constant.Int64Val(v); ok {
			return x
		}
	}
	Fatalf("%v out of range for %v", v, t)
	panic("unreachable")
}

func float64Val(v constant.Value) float64 {
	if x, _ := constant.Float64Val(v); !math.IsInf(x, 0) {
		return x + 0 // avoid -0 (should not be needed, but be conservative)
	}
	Fatalf("bad float64 value: %v", v)
	panic("unreachable")
}

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
		Fatalf("unexpected: %v", u)
	}
	return f
}

// Int64Val returns n as an int64.
// n must be an integer or rune constant.
func (n *Node) Int64Val() int64 {
	if !Isconst(n, constant.Int) {
		Fatalf("Int64Val(%v)", n)
	}
	x, ok := constant.Int64Val(n.Val())
	if !ok {
		Fatalf("Int64Val(%v)", n)
	}
	return x
}

// CanInt64 reports whether it is safe to call Int64Val() on n.
func (n *Node) CanInt64() bool {
	if !Isconst(n, constant.Int) {
		return false
	}

	// if the value inside n cannot be represented as an int64, the
	// return value of Int64 is undefined
	_, ok := constant.Int64Val(n.Val())
	return ok
}

// Uint64Val returns n as an uint64.
// n must be an integer or rune constant.
func (n *Node) Uint64Val() uint64 {
	if !Isconst(n, constant.Int) {
		Fatalf("Uint64Val(%v)", n)
	}
	x, ok := constant.Uint64Val(n.Val())
	if !ok {
		Fatalf("Uint64Val(%v)", n)
	}
	return x
}

// BoolVal returns n as a bool.
// n must be a boolean constant.
func (n *Node) BoolVal() bool {
	if !Isconst(n, constant.Bool) {
		Fatalf("BoolVal(%v)", n)
	}
	return constant.BoolVal(n.Val())
}

// StringVal returns the value of a literal string Node as a string.
// n must be a string constant.
func (n *Node) StringVal() string {
	if !Isconst(n, constant.String) {
		Fatalf("StringVal(%v)", n)
	}
	return constant.StringVal(n.Val())
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
	Fatalf("unexpected size: %v", sz)
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
// overflow, calls yyerror but does not truncate the input value.
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
func convlit(n *Node, t *types.Type) *Node    { return convlit1(n, t, false, nil) }
func defaultlit(n *Node, t *types.Type) *Node { return convlit1(n, t, false, nil) }

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
func convlit1(n *Node, t *types.Type, explicit bool, context func() string) *Node {
	if explicit && t == nil {
		Fatalf("explicit conversion missing type")
	}
	if t != nil && t.IsUntyped() {
		Fatalf("bad conversion to untyped: %v", t)
	}

	if n == nil || n.Type == nil {
		// Allow sloppy callers.
		return n
	}
	if !n.Type.IsUntyped() {
		// Already typed; nothing to do.
		return n
	}

	if n.Op == OLITERAL || n.Op == ONIL {
		// Can't always set n.Type directly on OLITERAL nodes.
		// See discussion on CL 20813.
		n = n.rawcopy()
	}

	// Nil is technically not a constant, so handle it specially.
	if n.Type.Etype == TNIL {
		if n.Op != ONIL {
			Fatalf("unexpected op: %v (%v)", n, n.Op)
		}
		if t == nil {
			yyerror("use of untyped nil")
			n.SetDiag(true)
			n.Type = nil
			return n
		}

		if !t.HasNil() {
			// Leave for caller to handle.
			return n
		}

		n.Type = t
		return n
	}

	if t == nil || !okforconst[t.Etype] {
		t = defaultType(n.Type)
	}

	switch n.Op {
	default:
		Fatalf("unexpected untyped expression: %v", n)

	case OLITERAL:
		v := convertVal(n.Val(), t, explicit)
		if v.Kind() == constant.Unknown {
			break
		}
		n.Type = t
		n.SetVal(v)
		return n

	case OPLUS, ONEG, OBITNOT, ONOT, OREAL, OIMAG:
		ot := operandType(n.Op, t)
		if ot == nil {
			n = defaultlit(n, nil)
			break
		}

		n.Left = convlit(n.Left, ot)
		if n.Left.Type == nil {
			n.Type = nil
			return n
		}
		n.Type = t
		return n

	case OADD, OSUB, OMUL, ODIV, OMOD, OOR, OXOR, OAND, OANDNOT, OOROR, OANDAND, OCOMPLEX:
		ot := operandType(n.Op, t)
		if ot == nil {
			n = defaultlit(n, nil)
			break
		}

		n.Left = convlit(n.Left, ot)
		n.Right = convlit(n.Right, ot)
		if n.Left.Type == nil || n.Right.Type == nil {
			n.Type = nil
			return n
		}
		if !types.Identical(n.Left.Type, n.Right.Type) {
			yyerror("invalid operation: %v (mismatched types %v and %v)", n, n.Left.Type, n.Right.Type)
			n.Type = nil
			return n
		}

		n.Type = t
		return n

	case OEQ, ONE, OLT, OLE, OGT, OGE:
		if !t.IsBoolean() {
			break
		}
		n.Type = t
		return n

	case OLSH, ORSH:
		n.Left = convlit1(n.Left, t, explicit, nil)
		n.Type = n.Left.Type
		if n.Type != nil && !n.Type.IsInteger() {
			yyerror("invalid operation: %v (shift of type %v)", n, n.Type)
			n.Type = nil
		}
		return n
	}

	if !n.Diag() {
		if !t.Broke() {
			if explicit {
				yyerror("cannot convert %L to type %v", n, t)
			} else if context != nil {
				yyerror("cannot use %L as type %v in %s", n, t, context())
			} else {
				yyerror("cannot use %L as type %v", n, t)
			}
		}
		n.SetDiag(true)
	}
	n.Type = nil
	return n
}

func operandType(op Op, t *types.Type) *types.Type {
	switch op {
	case OCOMPLEX:
		if t.IsComplex() {
			return floatForComplex(t)
		}
	case OREAL, OIMAG:
		if t.IsFloat() {
			return complexForFloat(t)
		}
	default:
		if okfor[op][t.Etype] {
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
			yyerror("constant %v truncated to real", v)
		}
		v = constant.Real(v)
	}

	return constant.ToFloat(v)
}

func toint(v constant.Value) constant.Value {
	if v.Kind() == constant.Complex {
		if constant.Sign(constant.Imag(v)) != 0 {
			yyerror("constant %v truncated to integer", v)
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
		yyerror("integer too large")
	} else {
		var t big.Float
		t.Parse(fmt.Sprint(v), 0)
		if t.IsInt() {
			yyerror("constant truncated to integer")
		} else {
			yyerror("constant %v truncated to integer", v)
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
	Fatalf("doesoverflow: %v, %v", v, t)
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
		yyerror("integer too large")
		return true
	}
	if doesoverflow(v, t) {
		yyerror("constant %v overflows %v", vconv(v, 0), t)
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

func consttype(n *Node) constant.Kind {
	if n == nil || n.Op != OLITERAL {
		return constant.Unknown
	}
	return n.Val().Kind()
}

func Isconst(n *Node, ct constant.Kind) bool {
	return consttype(n) == ct
}

var tokenForOp = [...]token.Token{
	OPLUS:   token.ADD,
	ONEG:    token.SUB,
	ONOT:    token.NOT,
	OBITNOT: token.XOR,

	OADD:    token.ADD,
	OSUB:    token.SUB,
	OMUL:    token.MUL,
	ODIV:    token.QUO,
	OMOD:    token.REM,
	OOR:     token.OR,
	OXOR:    token.XOR,
	OAND:    token.AND,
	OANDNOT: token.AND_NOT,
	OOROR:   token.LOR,
	OANDAND: token.LAND,

	OEQ: token.EQL,
	ONE: token.NEQ,
	OLT: token.LSS,
	OLE: token.LEQ,
	OGT: token.GTR,
	OGE: token.GEQ,

	OLSH: token.SHL,
	ORSH: token.SHR,
}

// evalConst returns a constant-evaluated expression equivalent to n.
// If n is not a constant, evalConst returns n.
// Otherwise, evalConst returns a new OLITERAL with the same value as n,
// and with .Orig pointing back to n.
func evalConst(n *Node) *Node {
	nl, nr := n.Left, n.Right

	// Pick off just the opcodes that can be constant evaluated.
	switch op := n.Op; op {
	case OPLUS, ONEG, OBITNOT, ONOT:
		if nl.Op == OLITERAL {
			var prec uint
			if n.Type.IsUnsigned() {
				prec = uint(n.Type.Size() * 8)
			}
			return origConst(n, constant.UnaryOp(tokenForOp[op], nl.Val(), prec))
		}

	case OADD, OSUB, OMUL, ODIV, OMOD, OOR, OXOR, OAND, OANDNOT, OOROR, OANDAND:
		if nl.Op == OLITERAL && nr.Op == OLITERAL {
			rval := nr.Val()

			// check for divisor underflow in complex division (see issue 20227)
			if op == ODIV && n.Type.IsComplex() && constant.Sign(square(constant.Real(rval))) == 0 && constant.Sign(square(constant.Imag(rval))) == 0 {
				yyerror("complex division by zero")
				n.Type = nil
				return n
			}
			if (op == ODIV || op == OMOD) && constant.Sign(rval) == 0 {
				yyerror("division by zero")
				n.Type = nil
				return n
			}

			tok := tokenForOp[op]
			if op == ODIV && n.Type.IsInteger() {
				tok = token.QUO_ASSIGN // integer division
			}
			return origConst(n, constant.BinaryOp(nl.Val(), tok, rval))
		}

	case OEQ, ONE, OLT, OLE, OGT, OGE:
		if nl.Op == OLITERAL && nr.Op == OLITERAL {
			return origBoolConst(n, constant.Compare(nl.Val(), tokenForOp[op], nr.Val()))
		}

	case OLSH, ORSH:
		if nl.Op == OLITERAL && nr.Op == OLITERAL {
			// shiftBound from go/types; "so we can express smallestFloat64"
			const shiftBound = 1023 - 1 + 52
			s, ok := constant.Uint64Val(nr.Val())
			if !ok || s > shiftBound {
				yyerror("invalid shift count %v", nr)
				n.Type = nil
				break
			}
			return origConst(n, constant.Shift(toint(nl.Val()), tokenForOp[op], uint(s)))
		}

	case OCONV, ORUNESTR:
		if okforconst[n.Type.Etype] && nl.Op == OLITERAL {
			return origConst(n, convertVal(nl.Val(), n.Type, true))
		}

	case OCONVNOP:
		if okforconst[n.Type.Etype] && nl.Op == OLITERAL {
			// set so n.Orig gets OCONV instead of OCONVNOP
			n.Op = OCONV
			return origConst(n, nl.Val())
		}

	case OADDSTR:
		// Merge adjacent constants in the argument list.
		s := n.List.Slice()
		need := 0
		for i := 0; i < len(s); i++ {
			if i == 0 || !Isconst(s[i-1], constant.String) || !Isconst(s[i], constant.String) {
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
				strs = append(strs, c.StringVal())
			}
			return origConst(n, constant.MakeString(strings.Join(strs, "")))
		}
		newList := make([]*Node, 0, need)
		for i := 0; i < len(s); i++ {
			if Isconst(s[i], constant.String) && i+1 < len(s) && Isconst(s[i+1], constant.String) {
				// merge from i up to but not including i2
				var strs []string
				i2 := i
				for i2 < len(s) && Isconst(s[i2], constant.String) {
					strs = append(strs, s[i2].StringVal())
					i2++
				}

				nl := origConst(s[i], constant.MakeString(strings.Join(strs, "")))
				nl.Orig = nl // it's bigger than just s[i]
				newList = append(newList, nl)
				i = i2 - 1
			} else {
				newList = append(newList, s[i])
			}
		}

		n = n.copy()
		n.List.Set(newList)
		return n

	case OCAP, OLEN:
		switch nl.Type.Etype {
		case TSTRING:
			if Isconst(nl, constant.String) {
				return origIntConst(n, int64(len(nl.StringVal())))
			}
		case TARRAY:
			if !hascallchan(nl) {
				return origIntConst(n, nl.Type.NumElem())
			}
		}

	case OALIGNOF, OOFFSETOF, OSIZEOF:
		return origIntConst(n, evalunsafe(n))

	case OREAL:
		if nl.Op == OLITERAL {
			return origConst(n, constant.Real(nl.Val()))
		}

	case OIMAG:
		if nl.Op == OLITERAL {
			return origConst(n, constant.Imag(nl.Val()))
		}

	case OCOMPLEX:
		if nl.Op == OLITERAL && nr.Op == OLITERAL {
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
		Fatalf("infinity is not a valid constant")
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
	OADD:    "addition",
	OSUB:    "subtraction",
	OMUL:    "multiplication",
	OLSH:    "shift",
	OXOR:    "bitwise XOR",
	OBITNOT: "bitwise complement",
}

// origConst returns an OLITERAL with orig n and value v.
func origConst(n *Node, v constant.Value) *Node {
	lno := setlineno(n)
	v = convertVal(v, n.Type, false)
	lineno = lno

	switch v.Kind() {
	case constant.Int:
		if constant.BitLen(v) <= Mpprec {
			break
		}
		fallthrough
	case constant.Unknown:
		what := overflowNames[n.Op]
		if what == "" {
			Fatalf("unexpected overflow: %v", n.Op)
		}
		yyerrorl(n.Pos, "constant %v overflow", what)
		n.Type = nil
		return n
	}

	orig := n
	n = nodl(orig.Pos, OLITERAL, nil, nil)
	n.Orig = orig
	n.Type = orig.Type
	n.SetVal(v)
	return n
}

func assertRepresents(t *types.Type, v constant.Value) {
	if !represents(t, v) {
		Fatalf("%v does not represent %v", t, v)
	}
}

func represents(t *types.Type, v constant.Value) bool {
	switch v.Kind() {
	case constant.Unknown:
		return okforconst[t.Etype]
	case constant.Bool:
		return t.IsBoolean()
	case constant.String:
		return t.IsString()
	case constant.Int:
		return t.IsInteger()
	case constant.Float:
		return t.IsFloat()
	case constant.Complex:
		return t.IsComplex()
	}

	Fatalf("unexpected constant kind: %v", v)
	panic("unreachable")
}

func origBoolConst(n *Node, v bool) *Node {
	return origConst(n, constant.MakeBool(v))
}

func origIntConst(n *Node, v int64) *Node {
	return origConst(n, constant.MakeInt64(v))
}

// nodlit returns a new untyped constant with value v.
func nodlit(v constant.Value) *Node {
	n := nod(OLITERAL, nil, nil)
	if k := v.Kind(); k != constant.Unknown {
		n.Type = idealType(k)
		n.SetVal(v)
	}
	return n
}

func idealType(ct constant.Kind) *types.Type {
	switch ct {
	case constant.String:
		return types.UntypedString
	case constant.Bool:
		return types.UntypedBool
	case constant.Int:
		return types.UntypedInt
	case constant.Float:
		return types.UntypedFloat
	case constant.Complex:
		return types.UntypedComplex
	}
	Fatalf("unexpected Ctype: %v", ct)
	return nil
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

	// Can't mix bool with non-bool, string with non-string, or nil with anything (untyped).
	if l.Type.IsBoolean() != r.Type.IsBoolean() {
		return l, r
	}
	if l.Type.IsString() != r.Type.IsString() {
		return l, r
	}
	if l.isNil() || r.isNil() {
		return l, r
	}

	t := defaultType(mixUntyped(l.Type, r.Type))
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
		Fatalf("bad type %v", t)
		panic("unreachable")
	}

	if rank(t2) > rank(t1) {
		return t2
	}
	return t1
}

func defaultType(t *types.Type) *types.Type {
	if !t.IsUntyped() || t.Etype == TNIL {
		return t
	}

	switch t {
	case types.UntypedBool:
		return types.Types[TBOOL]
	case types.UntypedString:
		return types.Types[TSTRING]
	case types.UntypedInt:
		return types.Types[TINT]
	case types.UntypedRune:
		return types.Runetype
	case types.UntypedFloat:
		return types.Types[TFLOAT64]
	case types.UntypedComplex:
		return types.Types[TCOMPLEX128]
	}

	Fatalf("bad type %v", t)
	return nil
}

func smallintconst(n *Node) bool {
	if n.Op == OLITERAL {
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
func indexconst(n *Node) int64 {
	if n.Op != OLITERAL {
		return -1
	}
	if !n.Type.IsInteger() && n.Type.Etype != TIDEAL {
		return -1
	}

	v := toint(n.Val())
	if v.Kind() != constant.Int || constant.Sign(v) < 0 {
		return -1
	}
	if doesoverflow(v, types.Types[TINT]) {
		return -2
	}
	return int64Val(types.Types[TINT], v)
}

// isGoConst reports whether n is a Go language constant (as opposed to a
// compile-time constant).
//
// Expressions derived from nil, like string([]byte(nil)), while they
// may be known at compile time, are not Go language constants.
func (n *Node) isGoConst() bool {
	return n.Op == OLITERAL
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
func (s *constSet) add(pos src.XPos, n *Node, what, where string) {
	if n.Op == OCONVIFACE && n.Implicit() {
		n = n.Left
	}

	if !n.isGoConst() {
		return
	}
	if n.Type.IsUntyped() {
		Fatalf("%v is untyped", n)
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

	typ := n.Type
	switch typ {
	case types.Bytetype:
		typ = types.Types[TUINT8]
	case types.Runetype:
		typ = types.Types[TINT32]
	}
	k := constSetKey{typ, n.ValueInterface()}

	if hasUniquePos(n) {
		pos = n.Pos
	}

	if s.m == nil {
		s.m = make(map[constSetKey]src.XPos)
	}

	if prevPos, isDup := s.m[k]; isDup {
		yyerrorl(pos, "duplicate %s %s in %s\n\tprevious %s at %v",
			what, nodeAndVal(n), where,
			what, linestr(prevPos))
	} else {
		s.m[k] = pos
	}
}

// nodeAndVal reports both an expression and its constant value, if
// the latter is non-obvious.
//
// TODO(mdempsky): This could probably be a fmt.go flag.
func nodeAndVal(n *Node) string {
	show := n.String()
	val := n.ValueInterface()
	if s := fmt.Sprintf("%#v", val); show != s {
		show += " (value " + s + ")"
	}
	return show
}
